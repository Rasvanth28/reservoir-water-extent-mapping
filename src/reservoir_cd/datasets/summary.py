"""Chip / split summary: counts, class balance, store size."""
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_cd.datasets.chips import open_store

CLASSES = ["Water", "Forest", "Bare", "Other"]
CHANGE = ["NoChange", "Loss", "Gain"]


def dir_size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def chip_summary(out: Path, cfg: dict, sites: list[dict]) -> dict:
    sp = pd.read_parquet(out / "splits.parquet")
    size = cfg["chip_size"]
    rep = {"store_mb": {}, "split_counts": {}, "class_balance": {}, "change_balance": {}}
    tot = {}
    for s in sites:
        name = s["name"]
        rep["store_mb"][name] = round(dir_size_mb(out / f"{name}.zarr"), 1)
        idx = pd.read_parquet(out / f"{name}_index.parquet").merge(sp, on=["site", "row0", "col0"])
        g = open_store(out / f"{name}.zarr")
        lab, years = g["label_cons"][:], g.attrs["years"]
        yi = {y: i for i, y in enumerate(years)}
        for col in ["split"] + [c for c in sp.columns if c.startswith("loso_")]:
            for sname, d in idx.groupby(col):
                key = f"{name}/{col}/{sname}"
                rep["split_counts"][key] = {"windows": int(d[["row0", "col0"]].drop_duplicates().shape[0]),
                                            "chip_pairs": int(len(d))}
                if col != "split" or sname == "buffer":
                    continue
                cls = tot.setdefault(sname, np.zeros(4, np.int64)); chg = tot.setdefault(sname + "_chg", np.zeros(3, np.int64))
                for r in d.itertuples():
                    sl = (slice(r.row0, r.row0 + size), slice(r.col0, r.col0 + size))
                    l1, l2 = lab[yi[r.year_t1]][sl], lab[yi[r.year_t2]][sl]
                    for l in (l1, l2):
                        cls += np.bincount(l[l < 4], minlength=4)
                    ok = (l1 != 255) & (l2 != 255)
                    chg += [int((ok & ~((l1 == 0) ^ (l2 == 0))).sum()), int((ok & (l1 == 0) & (l2 != 0)).sum()),
                            int((ok & (l1 != 0) & (l2 == 0)).sum())]
    for k, v in tot.items():
        if k.endswith("_chg"):
            rep["change_balance"][k[:-4]] = dict(zip(CHANGE, map(int, v)))
        else:
            rep["class_balance"][k] = dict(zip(CLASSES, map(int, v)))
    rep["total_store_mb"] = round(sum(rep["store_mb"].values()), 1)
    return rep
