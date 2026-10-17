"""Draw the full v2 validation sample for all sites and write blind labeling sheets + keys.

Usage: .venv/bin/python scripts/draw_validation_sample.py [configs/validation.yaml]
Reads data/interim/<Site>_labels.zarr (read-only). Writes only under out_dir (data/ground_truth/full).
Never touches the pilot directory. Strata and automatic labels go only in <SITE>_key.csv.
"""
import csv
import json
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

from reservoir_cd.config import load_yaml
from reservoir_cd.datasets.exclusion import exclusion_mask
from reservoir_cd.grid import Grid
from reservoir_cd.preprocessing.labels import CLASS_NAMES
from reservoir_cd.validation import full_sample, sampling, strata

KEY_NAMES = {**CLASS_NAMES, 255: "Unlabeled"}


def fmt_date(d) -> str:
    return "" if np.isnat(d) else str(np.datetime64(d, "D"))


def main() -> None:
    cfg = load_yaml(sys.argv[1] if len(sys.argv) > 1 else "configs/validation.yaml")
    sites = load_yaml(cfg["sites_config"])["sites"]
    years = list(range(cfg["pairs_start"], cfg["pairs_end"] + 1))
    pairs = list(zip(years[:-1], years[1:]))
    out_root = Path(cfg["out_dir"])
    pilot_xy = np.zeros((0, 2))
    keys = sorted(glob(cfg["pilot_key_glob"]))
    if keys:
        pilot_xy = np.concatenate([pd.read_csv(k, usecols=["x", "y"]).to_numpy(float) for k in keys])
    rng2 = np.random.default_rng(cfg["seed"] + 7)
    summary, weights = {}, []
    for si, site in enumerate(sites):
        name, code = site["name"], site["code"]
        zpath = f"{cfg['interim_dir']}/{name}_labels.zarr"
        st = xr.open_zarr(zpath, group="static", consolidated=False)
        grid = Grid(float(st.x.values[0]) - 15.0, float(st.y.values[0]) + 15.0, 30.0, st.sizes["y"], st.sizes["x"])
        study = st.study_area.values.astype(bool)
        rule, cons, sdate = {}, {}, {}
        for y in years:
            g = xr.open_zarr(zpath, group=f"year_{y}", consolidated=False)
            rule[y], cons[y], sdate[y] = g.rule_label.values, g.consensus_label.values, g.source_date.values
        excl = exclusion_mask(pilot_xy, grid.x0, grid.y0, grid.res, grid.shape, cfg["pilot_exclusion_radius_m"])
        smaps_full, smaps = [], []
        for a, b in pairs:
            s = strata.assign_strata_v2(rule[a], rule[b], study, cfg["shoreline_width_px"])
            smaps_full.append(s)
            smaps.append(np.where(excl, 0, s).astype(np.uint8))
        pts, rep = full_sample.draw_sample_v2(smaps, cfg["allocation"], grid, cfg["min_spacing_m"],
                                              cfg["seed"] + si, cfg["spaced_cap_per_pair"])
        for k, stn in enumerate(strata.STRATA_V2, 1):
            tot = [int((s == k).sum()) for s in smaps_full]
            for p, (a, b) in enumerate(pairs):
                weights.append({"site": name, "stratum": stn, "pair": f"{a}->{b}", "n_pixels": tot[p],
                                "n_pixels_after_pilot_exclusion": rep["population_by_pair"][stn][p]})
            weights.append({"site": name, "stratum": stn, "pair": "ALL", "n_pixels": sum(tot),
                            "n_pixels_after_pilot_exclusion": rep["population"][stn]})
        order = np.random.default_rng(cfg["seed"] + 100 + si).permutation(len(pts))
        to_ll = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
        bs = cfg["block_size_m"]
        final = []
        for n, i in enumerate(order, 1):
            p = dict(pts[i])
            a, b = pairs[p["pair"]]
            r, c = p["row"], p["col"]
            p.update(point_id=f"{code}-{n:04d}", site=name, year_t1=a, year_t2=b,
                     date_t1=fmt_date(sdate[a][r, c]), date_t2=fmt_date(sdate[b][r, c]),
                     block_id=f"{int(p['x'] // bs)}_{int(p['y'] // bs)}",
                     rule_label_t1=KEY_NAMES[int(rule[a][r, c])], rule_label_t2=KEY_NAMES[int(rule[b][r, c])],
                     consensus_label_t1=KEY_NAMES[int(cons[a][r, c])], consensus_label_t2=KEY_NAMES[int(cons[b][r, c])],
                     second_interpreter="FALSE")
            p["lon"], p["lat"] = to_ll.transform(p["x"], p["y"])
            assert p["date_t1"] and p["date_t2"], p["point_id"]
            final.append(p)
        second = set(rng2.choice([p["point_id"] for p in final], cfg["second_interpreter_per_site"], replace=False))
        for p in final:
            if p["point_id"] in second:
                p["second_interpreter"] = "TRUE"
        d = out_root / code
        (out_root / "second_interpreter").mkdir(parents=True, exist_ok=True)
        d.mkdir(parents=True, exist_ok=True)
        srt = sorted(final, key=lambda q: q["point_id"])
        sampling.write_kml(srt, d / f"{code}_points.kml", grid, f"{code} validation points")
        full_sample.write_labels_sheet(final, name, d / f"{code}_labels.csv")
        full_sample.write_key(final, d / f"{code}_key.csv")
        sec = [p for p in srt if p["point_id"] in second]
        sampling.write_kml(sec, out_root / "second_interpreter" / f"{code}_points_2nd.kml", grid,
                           f"{code} second-interpreter points")
        full_sample.write_labels_sheet(sec, name, out_root / "second_interpreter" / f"{code}_labels_2nd.csv")
        rep["by_pair"] = {f"{a}->{b}": {s: sum(1 for q in final if q["pair"] == k and q["stratum"] == s)
                                        for s in strata.STRATA_V2} for k, (a, b) in enumerate(pairs)}
        rep["pilot_excluded_px"] = int(excl.sum())
        summary[name] = rep
        print(name, "drawn", rep["drawn"], "moved_to_S5", rep["moved_to_S5"], flush=True)
    pd.DataFrame(weights).to_csv(out_root / "strata_weights.csv", index=False)
    (out_root / "README.md").write_text("The *_key.csv files hold strata and automatic labels: the labeler must NOT open them.\n")
    (out_root / "draw_summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
