"""Reservoir extent derived from the data: water frequency -> permanent body -> maximum extent.

Replaces the KML-based basin mask (the project KMLs are only the download bounding boxes).
"""
import numpy as np
from scipy import ndimage

from reservoir_cd.preprocessing import indices, qa
from reservoir_cd.preprocessing.labels import robust_scene_threshold

EIGHT = np.ones((3, 3), dtype=bool)


def load_scene_indices(ds, cfg: dict) -> dict:
    """Per-scene MNDWI and NDVI (float32, NaN where invalid) plus the valid mask.

    Reads one time step at a time (the cube is chunked 1 scene per chunk). Reflectance is
    already scaled in the Zarr; it is never rescaled here.
    """
    T = ds.sizes["time"]
    H, W = ds.sizes["y"], ds.sizes["x"]
    mndwi = np.full((T, H, W), np.nan, np.float32)
    ndvi = np.full((T, H, W), np.nan, np.float32)
    valid = np.zeros((T, H, W), bool)
    for i in range(T):
        a = ds.landsat_data.isel(time=i).values  # (7,H,W)
        v = qa.valid_mask(a[6], cfg["qa"]["open_size"], cfg["qa"]["buffer_px"])
        m = indices.mndwi(a[1], a[4])
        n = indices.ndvi(a[3], a[2])
        v &= np.isfinite(m) & np.isfinite(n)
        valid[i] = v
        mndwi[i] = np.where(v, m, np.nan)
        ndvi[i] = np.where(v, n, np.nan)
    return {"mndwi": mndwi, "ndvi": ndvi, "valid": valid}


def scene_thresholds(mndwi: np.ndarray, region: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust per-scene threshold; scenes failing the bimodality / clamp test get the median of
    the scenes that passed. Returns (thresholds, fallback_flags, eta)."""
    T = mndwi.shape[0]
    t = np.full(T, np.nan)
    eta = np.full(T, np.nan)
    for i in range(T):
        t[i], _, eta[i] = robust_scene_threshold(mndwi[i], region, None, cfg)
    ok = np.isfinite(t)
    fill = float(np.median(t[ok])) if ok.any() else 0.0
    return np.where(ok, t, fill), ~ok, eta


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask, structure=EIGHT)
    if n == 0:
        return np.zeros_like(mask)
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def extent_from_frequency(freq: np.ndarray, perm_min: float, max_min: float, closing_px: int) -> tuple[np.ndarray, np.ndarray]:
    """(permanent, max_extent) from a water-frequency map.

    permanent = largest 8-connected component of freq >= perm_min. max_extent = pixels with
    freq >= max_min 8-connected to that component (so detached tanks / paddy are excluded),
    then a binary closing of ``closing_px`` iterations.
    """
    f = np.nan_to_num(freq, nan=0.0)
    permanent = largest_component(f >= perm_min)
    cand = f >= max_min
    lab, n = ndimage.label(cand, structure=EIGHT)
    keep = np.unique(lab[permanent & (lab > 0)])
    ext = np.isin(lab, keep[keep > 0]) if n else np.zeros_like(cand)
    if closing_px > 0:
        ext = ndimage.binary_closing(np.pad(ext, closing_px), structure=EIGHT, iterations=closing_px)[
            closing_px:-closing_px, closing_px:-closing_px]
        # closing may touch pixels that are not water-frequency candidates; that is intended (hole filling)
    return permanent, ext


def buffer_mask(mask: np.ndarray, buffer_m: float, res: float) -> np.ndarray:
    """Pixels within ``buffer_m`` of ``mask`` (Euclidean), clipped to the grid."""
    if not mask.any():
        return mask.copy()
    d = ndimage.distance_transform_edt(~mask)
    return d * res <= buffer_m


def water_frequency(sc: dict, thr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(frequency, n_clear): fraction of clear scenes with MNDWI > per-scene threshold."""
    T = sc["mndwi"].shape[0]
    n = sc["valid"].sum(axis=0)
    w = np.zeros(n.shape, np.int32)
    for i in range(T):
        w += sc["valid"][i] & (sc["mndwi"][i] > thr[i])
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = np.where(n > 0, w / np.maximum(n, 1), np.nan)
    return freq.astype(np.float32), n.astype(np.int16)


def derive_reservoir_extent(sc: dict, cfg: dict, res: float) -> dict:
    """Two-pass derivation: (1) rough extent with scene thresholds computed on the whole grid;
    (2) final extent with thresholds computed inside the rough study area."""
    rc = cfg["reservoir"]
    out = {}
    whole = np.ones(sc["valid"].shape[1:], bool)
    for stage, region in (("rough", whole), ("final", None)):
        if stage == "final":
            region = out["rough_study"]
        thr, fb, eta = scene_thresholds(sc["mndwi"], region, cfg)
        freq, n = water_frequency(sc, thr)
        perm, ext = extent_from_frequency(freq, rc["permanent_min_freq"], rc["max_extent_min_freq"], rc["closing_px"])
        study = buffer_mask(ext, rc["study_buffer_m"], res)
        if stage == "rough":
            out["rough_study"] = study
            out["rough_fallbacks"] = int(fb.sum())
    out.update(freq=freq, n_clear=n, permanent=perm, max_extent=ext, study=study,
               scene_thr=thr, scene_fallback=fb, scene_eta=eta)
    return out
