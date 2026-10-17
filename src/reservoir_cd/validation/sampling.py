"""Stratified random sampling of validation points + KML / CSV export."""
import csv
from pathlib import Path

import numpy as np
from pyproj import Transformer

from reservoir_cd.grid import Grid
from reservoir_cd.validation.strata import STRATA

CHANGE = ["S6", "S7", "S8"]


def draw_sample(strata_maps: list[np.ndarray], allocation: dict[str, int], grid: Grid,
                min_spacing_m: float, seed: int) -> tuple[list[dict], dict]:
    """Pool (pair, row, col) candidates per stratum, draw uniformly at random.

    Spacing: points of the same pair must be >= ``min_spacing_m`` apart (Euclidean).
    Shortfalls are redistributed round-robin to other change strata (then, if still
    short, to any stratum with candidates). Returns (points, report).
    """
    rng = np.random.default_rng(seed)
    pools = {}
    for si, name in enumerate(STRATA, start=1):
        parts = []
        for p, sm in enumerate(strata_maps):
            r, c = np.nonzero(sm == si)
            parts.append(np.stack([np.full(r.size, p), r, c], axis=1))
        arr = np.concatenate(parts) if parts else np.zeros((0, 3), int)
        pools[name] = arr[rng.permutation(len(arr))]
    cursor = {k: 0 for k in STRATA}
    chosen: list[dict] = []
    by_pair: dict[int, list[tuple[float, float]]] = {}
    got = {k: 0 for k in STRATA}

    def take(name: str, n: int) -> int:
        taken = 0
        pool = pools[name]
        while taken < n and cursor[name] < len(pool):
            p, r, c = pool[cursor[name]]
            cursor[name] += 1
            xy = grid.center(int(r), int(c))
            near = by_pair.get(int(p), [])
            if any(np.hypot(xy[0] - a, xy[1] - b) < min_spacing_m for a, b in near):
                continue
            by_pair.setdefault(int(p), []).append(xy)
            chosen.append({"stratum": name, "pair": int(p), "row": int(r), "col": int(c),
                           "x": xy[0], "y": xy[1]})
            taken += 1
        got[name] += taken
        return taken

    shortfall = 0
    for name in STRATA:
        shortfall += allocation[name] - take(name, allocation[name])
    redistributed = shortfall
    while shortfall > 0:
        progressed = False
        for name in CHANGE + [s for s in STRATA if s not in CHANGE]:
            if shortfall <= 0:
                break
            if take(name, 1):
                shortfall -= 1
                progressed = True
        if not progressed:
            break
    report = {"requested": allocation, "sampled": got, "shortfall_redistributed": redistributed,
              "unfilled": shortfall,
              "candidates": {k: int(len(v)) for k, v in pools.items()}}
    return chosen, report


def _square_lonlat(x: float, y: float, res: float, to_ll: Transformer) -> list[tuple[float, float]]:
    h = res / 2
    ring = [(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h), (x - h, y - h)]
    return [to_ll.transform(a, b) for a, b in ring]


def write_kml(points: list[dict], path: Path, grid: Grid, title: str = "BHA pilot validation points") -> None:
    """Pins (name = point_id, description with both dates) + 30 m footprint outlines."""
    to_ll = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           f'<name>{title}</name>',
           '<Style id="fp"><LineStyle><color>ff00ffff</color><width>2</width></LineStyle>'
           '<PolyStyle><fill>0</fill><outline>1</outline></PolyStyle></Style>',
           '<Style id="pin"><IconStyle><scale>0.7</scale></IconStyle></Style>',
           '<Folder><name>Pins</name>']
    for p in points:
        out.append(
            f'<Placemark><name>{p["point_id"]}</name>'
            f'<description>T1 image date: {p["date_t1"]}\nT2 image date: {p["date_t2"]}</description>'
            f'<styleUrl>#pin</styleUrl><Point><coordinates>{p["lon"]:.7f},{p["lat"]:.7f},0</coordinates></Point></Placemark>')
    out.append('</Folder><Folder><name>Pixel footprints (30 m)</name>')
    for p in points:
        ring = _square_lonlat(p["x"], p["y"], grid.res, to_ll)
        coords = " ".join(f"{lo:.7f},{la:.7f},0" for lo, la in ring)
        out.append(f'<Placemark><styleUrl>#fp</styleUrl><Polygon><tessellate>1</tessellate>'
                   f'<outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates>'
                   f'</LinearRing></outerBoundaryIs></Polygon></Placemark>')
    out.append('</Folder></Document></kml>')
    path.write_text("\n".join(out))


LABEL_COLS = ["point_id", "site", "lon", "lat", "date_t1", "date_t2",
              "label_t1", "confidence_t1", "mixed_t1", "ref_source_t1", "ref_date_t1",
              "label_t2", "confidence_t2", "mixed_t2", "ref_source_t2", "ref_date_t2",
              "interpreter", "minutes", "notes"]
KEY_COLS = ["point_id", "stratum", "year_t1", "year_t2", "x", "y", "block_id",
            "rule_label_t1", "rule_label_t2", "consensus_label_t1", "consensus_label_t2"]


def write_csvs(points: list[dict], site: str, out_dir: Path, code: str) -> None:
    with open(out_dir / f"{code}_labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLS, extrasaction="ignore", restval="")
        w.writeheader()
        for p in sorted(points, key=lambda q: q["point_id"]):
            w.writerow({**p, "site": site})
    with open(out_dir / f"{code}_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEY_COLS, extrasaction="ignore")
        w.writeheader()
        for p in sorted(points, key=lambda q: q["point_id"]):
            w.writerow(p)


def greedy_spaced_count(mask: np.ndarray, res: float, min_spacing_m: float, rng: np.random.Generator,
                        cap: int | None = None) -> int:
    """Number of candidate pixels that survive a random greedy thinning so that every kept pair is
    >= ``min_spacing_m`` apart (Euclidean between pixel centres). Stops at ``cap`` (returns cap)."""
    r_idx, c_idx = np.nonzero(mask)
    if r_idx.size == 0:
        return 0
    R = int(np.ceil(min_spacing_m / res)) - 1
    oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
    disk = np.hypot(oy, ox) * res < min_spacing_m
    blocked = np.zeros((mask.shape[0] + 2 * R, mask.shape[1] + 2 * R), bool)
    n = 0
    for k in rng.permutation(r_idx.size):
        r, c = int(r_idx[k]), int(c_idx[k])
        if blocked[r + R, c + R]:
            continue
        blocked[r:r + 2 * R + 1, c:c + 2 * R + 1] |= disk
        n += 1
        if cap is not None and n >= cap:
            return n
    return n
