"""Map labels at the validation points.

Every method yields, per point and date, a class name (Water / Forest / Bare / Other), "Non-water" for
water-only maps (Baseline A), or "" where the map has no label (no data, consensus Unlabeled). A point
is a 30 m pixel center, so the 15 m buffer rule (EDD 3.7) is the pixel that contains it.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

LAND = ["Forest", "Bare", "Other"]
CLASSES = ["Water", *LAND]


def from_key(points: pd.DataFrame, column: str) -> pd.DataFrame:
    """Automatic labels stored in the key (`rule_label`, `consensus_label`)."""
    out = pd.DataFrame(index=points.index)
    for t in ("t1", "t2"):
        out[t] = points[f"{column}_{t}"].where(points[f"{column}_{t}"].isin(CLASSES), "")
    return out


def pixel_index(points: pd.DataFrame, x0: float, y0: float, res: float) -> tuple[np.ndarray, np.ndarray]:
    col = np.floor((points.x.to_numpy() - x0) / res).astype(int)
    row = np.floor((y0 - points.y.to_numpy()) / res).astype(int)
    return row, col


def from_water_zarr(points: pd.DataFrame, pred_dir: Path, chips_dir: Path) -> pd.DataFrame:
    """Baseline A style store `<Site>.zarr` with `water` (Y,H,W: 0 non-water, 1 water, 255 no data)
    on the grid of `chips_dir/<Site>.zarr`."""
    out = pd.DataFrame("", index=points.index, columns=["t1", "t2"])
    for site, g in points.groupby("site"):
        grid = zarr.open_group(str(chips_dir / f"{site}.zarr"), mode="r").attrs
        store = zarr.open_group(str(pred_dir / f"{site}.zarr"), mode="r")
        water = store["water"][:]
        yi = {y: i for i, y in enumerate(store.attrs["years"])}
        row, col = pixel_index(g, grid["x0"], grid["y0"], grid["res"])
        if ((row < 0) | (row >= water.shape[1]) | (col < 0) | (col >= water.shape[2])).any():
            raise ValueError(f"{site}: validation point outside the prediction grid")
        for t in ("t1", "t2"):
            v = water[g[f"year_{t}"].map(yi).to_numpy(), row, col]
            out.loc[g.index, t] = np.select([v == 1, v == 0], ["Water", "Non-water"], "")
    return out
