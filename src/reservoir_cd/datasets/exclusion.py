"""Validation-point exclusion, applied at LOAD time.

Reads every ``data/ground_truth/**/*_key.csv`` (columns x, y in EPSG:32643) and, if present,
``validation_points.gpkg`` and returns a per-site boolean raster that is True within ``radius_m``
of any point. The dataset sets labels to 255 there, for ALL year pairs, so drawing the full
validation sample later needs no re-chipping.
"""
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


def load_points_xy(root: str | Path, key_glob: str, gpkg: str | None = None) -> np.ndarray:
    """All validation points as an (N, 2) array of EPSG:32643 x, y (deduplicated)."""
    root = Path(root)
    pts = []
    for f in sorted(glob(str(root / key_glob), recursive=True)):
        df = pd.read_csv(f, usecols=["x", "y"])
        pts.append(df[["x", "y"]].to_numpy(float))
    if gpkg and (root / gpkg).exists():
        import geopandas as gpd
        gdf = gpd.read_file(root / gpkg)
        gdf = gdf.to_crs("EPSG:32643")
        pts.append(np.stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()], axis=1))
    if not pts:
        return np.zeros((0, 2))
    return np.unique(np.round(np.concatenate(pts), 3), axis=0)


def exclusion_mask(points_xy: np.ndarray, x0: float, y0: float, res: float, shape: tuple[int, int],
                   radius_m: float) -> np.ndarray:
    """True where a pixel centre lies within ``radius_m`` of any point. (x0, y0) = top-left corner."""
    H, W = shape
    out = np.zeros(shape, bool)
    if radius_m is None or radius_m <= 0 or len(points_xy) == 0:
        return out
    k = int(np.ceil(radius_m / res)) + 1
    for x, y in points_xy:
        c = (x - x0) / res - 0.5
        r = (y0 - y) / res - 0.5
        r0, r1 = max(int(np.floor(r)) - k, 0), min(int(np.floor(r)) + k + 1, H)
        c0, c1 = max(int(np.floor(c)) - k, 0), min(int(np.floor(c)) + k + 1, W)
        if r0 >= r1 or c0 >= c1:
            continue
        rr, cc = np.mgrid[r0:r1, c0:c1]
        out[r0:r1, c0:c1] |= np.hypot(rr - r, cc - c) * res <= radius_m
    return out
