"""Reflectance units and Prithvi normalization: the ONLY place these conventions live.

The Landsat cubes / interim zarr hold surface reflectance in roughly [-0.2, 1.0] (already scaled;
never apply DN*0.0000275-0.2 again). Prithvi-EO-2.0 was trained on HLS data in reflectance*10000
units, and its normalization statistics are in those units (verified against
HF ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL config.json ``pretrained_cfg.mean/std``, bands
B02,B03,B04,B05(NIR narrow),B06(SWIR1),B07(SWIR2) = our blue,green,red,nir08,swir16,swir22).

Pipeline:  reflectance --(x HLS_SCALE, round, int16)--> chip store --(float, (x-mean)/std)--> model.
"""
import numpy as np

BAND_NAMES = ["blue", "green", "red", "nir08", "swir16", "swir22"]
HLS_SCALE = 10000.0
INT16_NODATA = -32768
PRITHVI_MEAN = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32)
PRITHVI_STD = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32)


def reflectance_to_int16(refl: np.ndarray) -> np.ndarray:
    """Float reflectance (NaN = no data) -> int16 in HLS units (reflectance*10000)."""
    refl = np.asarray(refl, dtype=np.float32)
    scaled = np.rint(np.nan_to_num(refl, nan=0.0, posinf=0.0, neginf=0.0) * HLS_SCALE)
    out = np.clip(scaled, -9000, 32000).astype(np.int16)
    out[~np.isfinite(refl)] = INT16_NODATA
    return out


def int16_to_valid(arr: np.ndarray) -> np.ndarray:
    """Per-pixel mask (all bands present). ``arr`` has bands on axis -3."""
    return (arr != INT16_NODATA).all(axis=-3)


def normalize_hls(arr: np.ndarray) -> np.ndarray:
    """int16 HLS-unit array (..., 6, H, W) -> float32 normalized; no-data pixels become 0 (= the mean)."""
    a = arr.astype(np.float32)
    shape = [1] * (a.ndim - 3) + [6, 1, 1]
    out = (a - PRITHVI_MEAN.reshape(shape)) / PRITHVI_STD.reshape(shape)
    out[np.broadcast_to(~int16_to_valid(arr)[..., None, :, :], out.shape)] = 0.0
    return out
