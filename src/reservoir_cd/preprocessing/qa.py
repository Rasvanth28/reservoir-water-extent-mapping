"""Decode Landsat Collection 2 QA_PIXEL and build a valid-pixel mask.

Bit positions (USGS Landsat 8-9 C2 L2 Science Product Guide; cross-checked with the
``classification:bitfields`` metadata of the Planetary Computer ``qa_pixel`` asset):
0 fill, 1 dilated cloud, 2 cirrus (high conf.), 3 cloud (high conf.), 4 cloud shadow
(high conf.), 5 snow/ice, 6 clear, 7 water.
"""
import numpy as np
from scipy import ndimage
from skimage.morphology import disk

BITS = {
    "fill": 0, "dilated_cloud": 1, "cirrus": 2, "cloud": 3,
    "cloud_shadow": 4, "snow": 5, "clear": 6, "water": 7,
}
INVALID_BITS = ("fill", "dilated_cloud", "cirrus", "cloud", "cloud_shadow")


def to_uint16(qa: np.ndarray) -> np.ndarray:
    """Cast float-stored QA_PIXEL to uint16 (NaN -> 1, i.e. fill)."""
    qa = np.asarray(qa)
    if qa.dtype.kind == "f":
        qa = np.where(np.isfinite(qa), qa, 1.0)
    return qa.astype(np.uint16)


def decode_bit(qa: np.ndarray, name: str) -> np.ndarray:
    """Boolean array: is bit ``name`` set?"""
    return ((to_uint16(qa) >> BITS[name]) & 1).astype(bool)


def invalid_raw(qa: np.ndarray) -> np.ndarray:
    """Fill | dilated cloud | cirrus | cloud | cloud shadow (no morphology)."""
    q = to_uint16(qa)
    mask = np.zeros(q.shape, dtype=bool)
    for name in INVALID_BITS:
        mask |= ((q >> BITS[name]) & 1).astype(bool)
    return mask


def _morph(mask: np.ndarray, fn, selem: np.ndarray) -> np.ndarray:
    """Apply a 2-D binary morphology op over the last two axes of a (..., H, W) array."""
    if mask.ndim == 2:
        return fn(mask, structure=selem)
    return np.stack([_morph(m, fn, selem) for m in mask])


def valid_mask(qa: np.ndarray, open_size: int = 3, buffer_px: int = 2) -> np.ndarray:
    """Valid (clear) mask: invalid -> 3x3 opening -> dilate by ``buffer_px``; return ~invalid.

    Accepts (H, W) or (T, H, W). Opening removes isolated invalid specks; the
    dilation then buffers the surviving cloud/shadow/fill regions.
    """
    inv = invalid_raw(qa)
    if open_size and open_size > 1:
        sq = np.ones((open_size, open_size), dtype=bool)
        inv = _morph(inv, ndimage.binary_opening, sq)
    if buffer_px and buffer_px > 0:
        inv = _morph(inv, ndimage.binary_dilation, disk(buffer_px).astype(bool))
    return ~inv
