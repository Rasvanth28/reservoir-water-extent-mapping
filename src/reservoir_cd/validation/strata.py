"""Stratification of (pixel, year-pair) population from rule labels.

S1 stable Water interior, S2 stable Forest, S3 stable Bare, S4 stable Other,
S5 stable (any class) within ``width`` px of a water/non-water edge in either year,
S6 water->non-water, S7 non-water->water, S8 Bare<->Other change with a real NDVI change (Forest is static, so
Forest<->Bare/Other class changes are excluded; Forest can only change via water/bare).
0 = not eligible.
"""
import numpy as np
from scipy import ndimage

STRATA = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]


def shoreline(label: np.ndarray, width: int = 1) -> np.ndarray:
    """Pixels whose (2w+1)^2 neighbourhood holds both water and (valid) non-water pixels."""
    water = label == 0
    land = (label != 255) & ~water
    size = 2 * width + 1
    return ndimage.maximum_filter(water, size=size) & ndimage.maximum_filter(land, size=size)


def assign_strata(lab1: np.ndarray, lab2: np.ndarray, eligible: np.ndarray, width: int = 1,
                  change_ok: np.ndarray | None = None, static_forest: bool = True) -> np.ndarray:
    """uint8 stratum code (1..8) per pixel, 0 where not eligible. Mutually exclusive.

    ``change_ok`` (optional bool) marks pixels whose index change is large enough to count as real
    land change; S8 requires it (class change alone can be threshold flicker). With ``static_forest``
    (default) S8 is Bare<->Other only; ``static_forest=False`` is the legacy pilot rule (any
    Forest/Bare/Other change).
    """
    ok = eligible & (lab1 != 255) & (lab2 != 255)
    w1, w2 = lab1 == 0, lab2 == 0
    stable = ok & (lab1 == lab2)
    edge = shoreline(lab1, width) | shoreline(lab2, width)
    s = np.zeros(lab1.shape, dtype=np.uint8)
    s[stable & w1 & ~edge] = 1
    s[stable & (lab1 == 1) & ~edge] = 2
    s[stable & (lab1 == 2) & ~edge] = 3
    s[stable & (lab1 == 3) & ~edge] = 4
    s[stable & edge] = 5
    s[ok & w1 & ~w2] = 6
    s[ok & ~w1 & w2] = 7
    s8 = ok & ~w1 & ~w2 & (lab1 != lab2)
    if static_forest:
        s8 &= ((lab1 == 2) & (lab2 == 3)) | ((lab1 == 3) & (lab2 == 2))
    if change_ok is not None:
        s8 &= change_ok
    s[s8] = 8
    return s


STRATA_V2 = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]


def assign_strata_v2(lab1: np.ndarray, lab2: np.ndarray, eligible: np.ndarray, width: int = 1) -> np.ndarray:
    """v2 partition (validation_protocol.md v2): codes 1..7, 0 = not eligible. S8 is absent.

    S5 shoreline = same class at both dates but within ``width`` px of a water edge on either date
    (priority over S1-S4); S1/S2/S3 stable Water interior / Forest / Bare; S6 Water->non-water;
    S7 non-water->Water; S4 = every remaining non-water-at-both-dates pixel (stable Other and any
    class change among Forest/Bare/Other). Uses RULE labels (full coverage, static forest).
    """
    ok = eligible & (lab1 != 255) & (lab2 != 255)
    w1, w2 = lab1 == 0, lab2 == 0
    stable = ok & (lab1 == lab2)
    edge = shoreline(lab1, width) | shoreline(lab2, width)
    s = np.zeros(lab1.shape, dtype=np.uint8)
    s[ok & ~w1 & ~w2] = 4
    s[stable & w1 & ~edge] = 1
    s[stable & (lab1 == 1) & ~edge] = 2
    s[stable & (lab1 == 2) & ~edge] = 3
    s[stable & edge] = 5
    s[ok & w1 & ~w2] = 6
    s[ok & ~w1 & w2] = 7
    return s
