"""v2 full validation sample: pooled stratified draw over all year pairs, S5 absorbs shortfalls."""
import csv
from pathlib import Path

import numpy as np

from reservoir_cd.grid import Grid
from reservoir_cd.validation.sampling import LABEL_COLS, greedy_spaced_count
from reservoir_cd.validation.strata import STRATA_V2

KEY_COLS_V2 = ["point_id", "site", "stratum", "year_t1", "year_t2", "x", "y", "lon", "lat", "block_id",
               "rule_label_t1", "rule_label_t2", "consensus_label_t1", "consensus_label_t2",
               "date_t1", "date_t2", "second_interpreter"]


def draw_sample_v2(strata_maps: list[np.ndarray], allocation: dict[str, int], grid: Grid,
                   min_spacing_m: float, seed: int, spaced_cap: int = 1000) -> tuple[list[dict], dict]:
    """Draw ``allocation`` points per stratum from the pooled (pair,row,col) population.

    Same-pair points are >= ``min_spacing_m`` apart (all strata share the spacing constraint).
    Order: S1-S4, S6, S7 first; whatever they fall short of is moved to S5, which is drawn last.
    Returns (points, report); points carry stratum, pair index, row, col, x, y.
    """
    rng = np.random.default_rng(seed)
    pools, pop = {}, {}
    for si, name in enumerate(STRATA_V2, start=1):
        parts = []
        for p, sm in enumerate(strata_maps):
            r, c = np.nonzero(sm == si)
            parts.append(np.stack([np.full(r.size, p), r, c], axis=1).astype(np.int32))
        arr = np.concatenate(parts)
        pools[name] = arr[rng.permutation(len(arr))]
        pop[name] = [int((sm == si).sum()) for sm in strata_maps]
    spaced = {}
    for si, name in enumerate(STRATA_V2, start=1):
        srng = np.random.default_rng(seed + 1000 + si)
        spaced[name] = [greedy_spaced_count(sm == si, grid.res, min_spacing_m, srng, cap=spaced_cap)
                        for sm in strata_maps]
    placed: dict[int, list[tuple[float, float]]] = {}
    chosen: list[dict] = []
    got = {k: 0 for k in STRATA_V2}

    def take(name: str, n: int) -> int:
        taken = 0
        for p, r, c in pools[name]:
            if taken >= n:
                break
            xy = grid.center(int(r), int(c))
            near = placed.setdefault(int(p), [])
            if near:
                a = np.asarray(near)
                if (np.hypot(a[:, 0] - xy[0], a[:, 1] - xy[1]) < min_spacing_m).any():
                    continue
            near.append(xy)
            chosen.append({"stratum": name, "pair": int(p), "row": int(r), "col": int(c), "x": xy[0], "y": xy[1]})
            taken += 1
        got[name] = taken
        return taken

    moved = 0
    quota = dict(allocation)
    for name in [s for s in STRATA_V2 if s != "S5"]:
        short = quota[name] - take(name, quota[name])
        moved += short
    quota["S5"] += moved
    take("S5", quota["S5"])
    report = {"requested": allocation, "drawn": got, "moved_to_S5": moved,
              "S5_quota_after_move": quota["S5"], "S5_unfilled": quota["S5"] - got["S5"],
              "population_by_pair": pop, "population": {k: int(sum(v)) for k, v in pop.items()},
              "spaced_candidates_by_pair": spaced,
              "spaced_candidates": {k: int(sum(v)) for k, v in spaced.items()},
              "spaced_cap_per_pair": spaced_cap}
    return chosen, report


def write_labels_sheet(points: list[dict], site: str, path: Path) -> None:
    """Blind labeling sheet: identical columns to the pilot sheet, label fields empty."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLS, extrasaction="ignore", restval="")
        w.writeheader()
        for p in sorted(points, key=lambda q: q["point_id"]):
            w.writerow({**p, "site": site})


def write_key(points: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEY_COLS_V2, extrasaction="ignore")
        w.writeheader()
        for p in sorted(points, key=lambda q: q["point_id"]):
            w.writerow(p)
