"""Baseline A (MNDWI + Otsu) for every site, year and consecutive-year pair.

Usage: .venv/bin/python scripts/run_baseline_a.py [--config configs/chips.yaml]
Reads data/processed/chips/<Site>.zarr (scene stores), writes
data/processed/predictions/baseline_a/<Site>.zarr (water: Y,H,W; change: P,H,W) and summary.json
(water area per year vs consensus / rule label water area, thresholds, fallbacks).
Baseline A predicts Water vs non-water only; land classes are N/A.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from reservoir_cd.config import load_yaml
from reservoir_cd.datasets.chips import open_store
from reservoir_cd.datasets.pairs import consecutive_pairs
from reservoir_cd.models import mndwi_otsu as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/chips.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    pcfg = load_yaml("configs/preprocessing.yaml")
    out_dir = Path("data/processed/predictions/baseline_a")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for s in load_yaml(cfg["sites_config"])["sites"]:
        name = s["name"]
        g = open_store(Path(cfg["out_dir"]) / f"{name}.zarr")
        years, H, W, res = g.attrs["years"], g.attrs["height"], g.attrs["width"], g.attrs["res"]
        study = g["study_area"][:].astype(bool)
        pairs = consecutive_pairs(years, cfg["pair_step_years"])
        o = zarr.open_group(str(out_dir / f"{name}.zarr"), mode="w")
        codec = zarr.codecs.ZstdCodec(level=5)
        water = o.create_array("water", shape=(len(years), H, W), chunks=(1, 256, 256), dtype="u1", compressors=codec)
        change = o.create_array("change", shape=(len(pairs), H, W), chunks=(1, 256, 256), dtype="u1", compressors=codec)
        yi = {y: i for i, y in enumerate(years)}
        info, w = {}, {}
        ha = res * res / 1e4
        for y in years:
            wy, inf = A.predict_year(g["refl"][yi[y]], study, pcfg)
            water[yi[y]] = wy
            w[y] = wy
            lc, lr = g["label_cons"][yi[y]], g["label_rule"][yi[y]]
            inf.update(water_ha=float((wy == 1).sum() * ha),
                       water_ha_in_study=float(((wy == 1) & study).sum() * ha),
                       consensus_water_ha=float((lc == 0).sum() * ha), rule_water_ha=float((lr == 0).sum() * ha),
                       nodata_frac=float((wy == 255).mean()))
            both = (lc != 255) & (wy != 255)  # agreement on pixels with a consensus label
            inf["agreement_with_consensus_water"] = float(((wy[both] == 1) == (lc[both] == 0)).mean()) if both.any() else None
            info[y] = inf
        for p, (a, b) in enumerate(pairs):
            change[p] = A.water_change(w[a], w[b])
        o.attrs.update(years=years, pairs=[list(p) for p in pairs], classes={"0": "non-water", "1": "water", "255": "no data"},
                       change_classes={"0": "no change", "1": "loss", "2": "gain", "255": "no data"},
                       note="Baseline A predicts Water vs non-water only; Forest/Bare/Other are N/A.")
        summary[name] = info
        print(name, {y: (round(i["water_ha"]), round(i["consensus_water_ha"]), round(i["threshold"], 3), i["fallback"]) for y, i in info.items()})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
