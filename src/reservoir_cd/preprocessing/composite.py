"""Seasonal medoid compositing."""
import warnings

import numpy as np


def season_indices(times: np.ndarray, year: int, start_month: int = 1, end_month: int = 4) -> np.ndarray:
    """Indices of ``times`` (datetime64) within [start_month, end_month] of ``year``."""
    t = times.astype("datetime64[M]")
    yr = t.astype("datetime64[Y]").astype(int) + 1970
    mo = t.astype(int) % 12 + 1
    return np.where((yr == year) & (mo >= start_month) & (mo <= end_month))[0]


def medoid_composite(refl: np.ndarray, valid: np.ndarray, times: np.ndarray) -> dict[str, np.ndarray]:
    """Per-pixel medoid over time.

    refl: (T, B, H, W) reflectance; valid: (T, H, W) bool; times: (T,) datetime64.
    For each pixel the nanmedian vector over valid observations is computed and the
    valid observation closest to it (Euclidean over bands) is chosen, so the output
    is always a real observation. Returns bands (B,H,W), source_date (H,W) datetime64[ns]
    (NaT where none), n_clear (H,W) int16, src_index (H,W) int (-1 where none).
    """
    T, B, H, W = refl.shape
    ok = valid & np.isfinite(refl).all(axis=1)
    x = np.where(ok[:, None], refl, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(x, axis=0)  # (B,H,W)
    dist = np.sqrt(((x - med[None]) ** 2).sum(axis=1))  # (T,H,W), NaN where invalid
    dist = np.where(ok, dist, np.inf)
    idx = dist.argmin(axis=0)
    n_clear = ok.sum(axis=0).astype(np.int16)
    bands = np.take_along_axis(x, idx[None, None], axis=0)[0]
    bands = np.where(n_clear[None] > 0, bands, np.nan)
    src = np.where(n_clear > 0, idx, -1)
    dates = times.astype("datetime64[ns]")[idx]
    dates = np.where(n_clear > 0, dates, np.datetime64("NaT", "ns"))
    return {"bands": bands, "source_date": dates, "n_clear": n_clear, "src_index": src}
