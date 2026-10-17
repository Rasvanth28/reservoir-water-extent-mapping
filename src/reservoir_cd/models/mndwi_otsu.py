"""Baseline A: MNDWI + Otsu on each yearly Jan-Apr composite (no training).

Predicts Water vs non-water ONLY. Forest / Bare / Other are not distinguished (N/A for this
baseline): in the 4-class evaluation Baseline A is scored on Water and on the water-change classes
(loss / gain) only. Threshold = per-composite Otsu inside the study area, restricted to clear
pixels, with the clamp and bimodality fallback of ``labels.robust_scene_threshold``; the fallback
value is the clamped composite Otsu used for the rule labels.
"""
import numpy as np

from reservoir_cd.datasets import normalize as nz
from reservoir_cd.preprocessing import indices, labels

NONWATER, WATER, INVALID = 0, 1, 255


def mndwi_from_int16(refl_i16: np.ndarray) -> np.ndarray:
    """refl_i16: (6, H, W) int16 HLS units (store format) -> MNDWI float32, NaN at no data."""
    f = refl_i16.astype(np.float32) / nz.HLS_SCALE
    f[:, ~nz.int16_to_valid(refl_i16)] = np.nan
    return indices.mndwi(f[1], f[4]).astype(np.float32)


def predict_year(refl_i16: np.ndarray, study: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    """Water map (uint8: 0 non-water, 1 water, 255 no data) and threshold info for one composite."""
    m = mndwi_from_int16(refl_i16)
    ok = np.isfinite(m)
    region = study.astype(bool) & ok
    clamp = tuple(cfg["water"]["otsu_clamp"])
    fb = labels.otsu_mndwi(m, region, clamp)
    t, fell_back, eta = labels.robust_scene_threshold(m, region, fb, cfg)
    out = np.where(m > t, WATER, NONWATER).astype(np.uint8)
    out[~ok] = INVALID
    return out, {"threshold": float(t), "fallback": bool(fell_back), "eta": float(eta), "clamped_otsu": float(fb)}


def water_change(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """0 no change, 1 water loss, 2 water gain, 255 if either date has no data."""
    ch = np.zeros(w1.shape, np.uint8)
    ch[(w1 == WATER) & (w2 == NONWATER)] = 1
    ch[(w1 == NONWATER) & (w2 == WATER)] = 2
    ch[(w1 == INVALID) | (w2 == INVALID)] = 255
    return ch
