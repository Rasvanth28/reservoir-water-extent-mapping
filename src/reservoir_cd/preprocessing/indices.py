"""Spectral indices. Inputs are surface reflectance (already scaled)."""
import numpy as np


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), NaN where the denominator is ~0."""
    num, den = a - b, a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(np.abs(den) < 1e-6, np.nan, out)


def mndwi(green: np.ndarray, swir16: np.ndarray) -> np.ndarray:
    return _nd(green, swir16)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return _nd(nir, red)


def bsi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray, swir16: np.ndarray) -> np.ndarray:
    """((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))."""
    return _nd(swir16 + red, nir + blue)


def all_indices(refl: np.ndarray) -> dict[str, np.ndarray]:
    """``refl`` has bands on axis -3 in order blue, green, red, nir08, swir16, swir22."""
    b, g, r, n, s1 = (refl[..., i, :, :] for i in range(5))
    return {"mndwi": mndwi(g, s1), "ndvi": ndvi(n, r), "bsi": bsi(b, r, n, s1)}
