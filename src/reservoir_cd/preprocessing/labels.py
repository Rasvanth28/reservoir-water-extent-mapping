"""Rule-based labels, temporal agreement and consensus labels.

Classes: 0 Water, 1 Forest, 2 Bare, 3 Other, 255 unlabeled.
Precedence: Water > Bare > Forest > Other (see EDD 3.2).
"""
import numpy as np
from skimage.filters import threshold_otsu

WATER, FOREST, BARE, OTHER, UNLABELED = 0, 1, 2, 3, 255
CLASS_NAMES = {0: "Water", 1: "Forest", 2: "Bare", 3: "Other", 255: "Unlabeled"}


def otsu_mndwi(mndwi: np.ndarray, region: np.ndarray, clamp: tuple[float, float]) -> float:
    """Otsu threshold on MNDWI within ``region``, clamped to ``clamp`` (used for the composite)."""
    v = mndwi[region & np.isfinite(mndwi)]
    if v.size < 50 or np.ptp(v) < 1e-6:
        return float(np.clip(0.0, *clamp))
    return float(np.clip(threshold_otsu(v), *clamp))


def otsu_with_eta(v: np.ndarray) -> tuple[float, float]:
    """(Otsu threshold, eta) where eta = between-class variance / total variance (0..1).

    eta is ~0.6+ for a clearly bimodal water/land histogram and drops towards 0.4 or lower
    for a unimodal one.
    """
    t = float(threshold_otsu(v))
    lo = v <= t
    w0 = lo.mean()
    if w0 in (0.0, 1.0):
        return t, 0.0
    between = w0 * (1 - w0) * (v[lo].mean() - v[~lo].mean()) ** 2
    tot = float(v.var())
    return t, float(between / tot) if tot > 0 else 0.0


def robust_scene_threshold(mndwi: np.ndarray, region: np.ndarray, fallback_t: float | None,
                           cfg: dict) -> tuple[float, bool, float]:
    """Per-scene Otsu with a bimodality test.

    Falls back to ``fallback_t`` (NaN if None) when there are too few valid pixels, eta <
    ``water.bimodality_min`` or the raw Otsu value lies on/outside the clamp.
    Returns (threshold, fell_back, eta).
    """
    v = mndwi[region & np.isfinite(mndwi)]
    fb = np.nan if fallback_t is None else float(fallback_t)
    if v.size < 50 or np.ptp(v) < 1e-6:
        return fb, True, 0.0
    t, eta = otsu_with_eta(v)
    lo, hi = cfg["water"]["otsu_clamp"]
    if eta < cfg["water"]["bimodality_min"] or t <= lo or t >= hi:
        return fb, True, eta
    return t, False, eta


def brightness(refl: np.ndarray) -> np.ndarray:
    """Mean of red and swir16 (bands on axis -3: blue, green, red, nir08, swir16, swir22)."""
    return 0.5 * (refl[..., 2, :, :] + refl[..., 4, :, :])


def rule_label(mndwi: np.ndarray, ndvi: np.ndarray, bright: np.ndarray, t: float, ext: np.ndarray,
               p90: np.ndarray, cfg: dict) -> np.ndarray:
    """Water > Bare > Forest > Other; 255 where MNDWI or NDVI is NaN.

    Water: MNDWI > t.  Bare: inside max extent, not water, NDVI < bare.ndvi_max (exposed reservoir
    bed); outside the extent only if bare.outside_enabled and NDVI < outside_ndvi_max and
    brightness > outside_bright_min.  Forest: annual NDVI percentile ``p90`` >= forest.ndvi_min and
    NDVI (dry-season composite or scene) >= forest.dry_ndvi_min.
    """
    ok = np.isfinite(mndwi) & np.isfinite(ndvi)
    fc, bc = cfg["forest"], cfg["bare"]
    lab = np.full(mndwi.shape, OTHER, dtype=np.uint8)
    lab[(p90 >= fc["ndvi_min"]) & (ndvi >= fc["dry_ndvi_min"])] = FOREST
    bare = ext & (ndvi < bc["ndvi_max"])
    if bc.get("outside_enabled", False):
        bare = bare | (~ext & (ndvi < bc["outside_ndvi_max"]) & (bright > bc["outside_bright_min"]))
    lab[bare] = BARE
    lab[mndwi > t] = WATER
    lab[~ok] = UNLABELED
    return lab


def temporal_agreement(scene_mndwi: np.ndarray, scene_ndvi: np.ndarray, scene_bright: np.ndarray,
                       scene_valid: np.ndarray, comp_label: np.ndarray, region: np.ndarray,
                       ext: np.ndarray, p90: np.ndarray, comp_t: float,
                       cfg: dict) -> tuple[np.ndarray, list[float], list[bool]]:
    """Fraction of clear scenes whose own rule label equals the composite rule label.

    Each scene gets a robust Otsu threshold inside ``region`` (study area), falling back to the
    composite threshold ``comp_t``. Also returns the per-scene thresholds and fallback flags.
    """
    S = scene_valid.shape[0]
    agree = np.zeros(comp_label.shape, np.float32)
    count = np.zeros(comp_label.shape, np.float32)
    thr, fbs = [], []
    for s in range(S):
        v = scene_valid[s] & np.isfinite(scene_mndwi[s]) & np.isfinite(scene_ndvi[s])
        m = np.where(v, scene_mndwi[s], np.nan)
        t, fb, _ = robust_scene_threshold(m, region & v, comp_t, cfg)
        thr.append(t)
        fbs.append(fb)
        lab = rule_label(m, scene_ndvi[s], scene_bright[s], t, ext, p90, cfg)
        count += v
        agree += v & (lab == comp_label)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(count > 0, agree / count, np.nan)
    return frac.astype(np.float32), thr, fbs


def consensus_label(rule: np.ndarray, mndwi: np.ndarray, t: float, n_clear: np.ndarray,
                    study: np.ndarray, agreement: np.ndarray, esri: np.ndarray | None,
                    ext: np.ndarray, p90: np.ndarray, cfg: dict) -> np.ndarray:
    """Confidence-filtered training label (see EDD 3.2).

    - MNDWI margin only for classes that depend on the water decision (Water; Bare in max extent).
    - Forest margin on the annual NDVI percentile.
    - Esri years: rule == Esri, except Bare inside max extent, which needs temporal agreement.
    - Non-Esri years: temporal agreement >= consensus.min_agreement.
    """
    out = rule.copy()
    cc = cfg["consensus"]
    drop = ~study | (n_clear < cc["min_clear"])
    water_dep = (rule == WATER) | ((rule == BARE) & ext)
    drop |= water_dep & ~(np.abs(mndwi - t) >= cfg["water"]["confidence_margin"])  # NaN also dropped
    drop |= (rule == FOREST) & ~(np.abs(p90 - cfg["forest"]["ndvi_min"]) >= cfg["forest"]["margin"])
    temporal_ok = agreement >= cc["min_agreement"]
    if esri is not None:
        esri_ok = (esri != UNLABELED) & (esri == rule)
        bare_bed = (rule == BARE) & ext
        drop |= np.where(bare_bed, ~temporal_ok, ~esri_ok)
    else:
        drop |= ~temporal_ok
    out[drop] = UNLABELED
    return out


# ----------------------------------------------------------------------------------------------
# Static-forest labelling (build_labels.py): Forest is one map for all years, see forest.py.
# Per year: Water (MNDWI > t) > Bare (inside max extent, not water, NDVI < bare.ndvi_max)
#           > Forest (static rule forest) > Other.
# ----------------------------------------------------------------------------------------------
def rule_label_static(mndwi: np.ndarray, ndvi: np.ndarray, t: float, ext: np.ndarray,
                      forest: np.ndarray, cfg: dict) -> np.ndarray:
    ok = np.isfinite(mndwi) & np.isfinite(ndvi)
    lab = np.full(mndwi.shape, OTHER, dtype=np.uint8)
    lab[forest] = FOREST
    lab[ext & (ndvi < cfg["bare"]["ndvi_max"])] = BARE
    lab[mndwi > t] = WATER
    lab[~ok] = UNLABELED
    return lab


def temporal_agreement_static(scene_mndwi: np.ndarray, scene_ndvi: np.ndarray, scene_valid: np.ndarray,
                              comp_label: np.ndarray, region: np.ndarray, ext: np.ndarray,
                              forest: np.ndarray, comp_t: float, cfg: dict) -> np.ndarray:
    """Fraction of clear scenes whose own static-forest rule label equals the composite rule label."""
    agree = np.zeros(comp_label.shape, np.float32)
    count = np.zeros(comp_label.shape, np.float32)
    for s in range(scene_valid.shape[0]):
        v = scene_valid[s] & np.isfinite(scene_mndwi[s]) & np.isfinite(scene_ndvi[s])
        m = np.where(v, scene_mndwi[s], np.nan)
        t, _, _ = robust_scene_threshold(m, region & v, comp_t, cfg)
        lab = rule_label_static(m, scene_ndvi[s], t, ext, forest, cfg)
        count += v
        agree += v & (lab == comp_label)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(count > 0, agree / count, np.nan)
    return frac.astype(np.float32)


def consensus_label_static(rule: np.ndarray, mndwi: np.ndarray, t: float, n_clear: np.ndarray,
                           study: np.ndarray, agreement: np.ndarray, esri: np.ndarray | None,
                           ext: np.ndarray, forest_cons: np.ndarray, cfg: dict) -> np.ndarray:
    """Confidence-filtered label with static forest.

    - Forest: kept iff the static consensus says forest (1); Other: kept iff it says non-forest (0).
    - Water, and Bare inside the max extent: |MNDWI - t| >= margin.
    - Esri years: Water needs Esri Water; Other needs Esri in {Crops, Rangeland, Built} (mapped to 3);
      in-extent Bare needs temporal agreement (Esri mislabels exposed bed).
    - Non-Esri years: Water, Bare, Other need temporal agreement >= consensus.min_agreement.
    """
    out = rule.copy()
    cc = cfg["consensus"]
    drop = ~study | (n_clear < cc["min_clear"])
    water_dep = (rule == WATER) | ((rule == BARE) & ext)
    drop |= water_dep & ~(np.abs(mndwi - t) >= cfg["water"]["confidence_margin"])
    drop |= (rule == FOREST) & (forest_cons != 1)
    drop |= (rule == OTHER) & (forest_cons != 0)
    temporal_ok = agreement >= cc["min_agreement"]
    needs = (rule == WATER) | (rule == BARE) | (rule == OTHER)
    if esri is not None:
        bare_bed = (rule == BARE) & ext
        esri_ok = (esri != UNLABELED) & (esri == rule)
        drop |= needs & np.where(bare_bed, ~temporal_ok, ~esri_ok)
    else:
        drop |= needs & ~temporal_ok
    out[drop] = UNLABELED
    return out
