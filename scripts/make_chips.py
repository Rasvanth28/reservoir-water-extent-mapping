"""Build per-site scene stores + chip indices + splits from data/interim/<Site>_labels.zarr.

Usage: .venv/bin/python scripts/make_chips.py [--config configs/chips.yaml] [--sites BHA MET PIL]
Writes data/processed/chips/<Site>.zarr, <Site>_index.parquet (chip-pair index),
<Site>_positions.parquet, splits.parquet and chips_summary.json. Read-only on data/interim.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_cd.config import load_yaml
from reservoir_cd.datasets import chips, splits, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/chips.yaml")
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--skip-store", action="store_true", help="reuse existing scene stores")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    sites = [s for s in load_yaml(cfg["sites_config"])["sites"] if not args.sites or s["code"] in args.sites]
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    allpos, allpairs = [], []
    for s in sites:
        name = s["name"]
        sp = out / f"{name}.zarr"
        if not args.skip_store:
            chips.build_scene_store(Path(cfg["labels_dir"]) / f"{name}_labels.zarr", sp, cfg["years"], cfg["chip_size"])
        pos, pairs = chips.build_index(sp, name, cfg)
        pos.to_parquet(out / f"{name}_positions.parquet")
        pairs.to_parquet(out / f"{name}_index.parquet")
        allpos.append(pos); allpairs.append(pairs)
        print(f"[{name}] positions={len(pos)} chip-pairs kept={len(pairs)}", flush=True)
    positions = pd.concat(allpos, ignore_index=True)
    pairs = pd.concat(allpairs, ignore_index=True)
    sp = splits.make_splits(positions, pairs, cfg)
    sp.to_parquet(out / "splits.parquet")
    rep = summary.chip_summary(out, cfg, sites)
    (out / "chips_summary.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["split_counts"], indent=1))


if __name__ == "__main__":
    main()
