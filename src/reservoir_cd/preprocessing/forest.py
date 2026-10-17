"""Static Forest map (one map for 2015-2026).

Forest is a static class: consensus of (A) Esri multi-year Trees and (C1) a Random Forest on pooled
multi-year Landsat phenology, trained on weak labels derived from Esri outside the reservoir extent.

Phenology features (per pixel, from ALL clear scenes 2015-2026, same-day scenes from adjacent
rows mosaicked first by averaging their valid NDVI so a date is counted once):
  c0, amp, phase_cos, phase_sin  - 1st-harmonic NDVI regression, NDVI(d) = c0 + a1 cos(2 pi d/365.25)
                                   + b1 sin(...); accumulated as normal equations (streaming in time)
  p10, p50, p90, std             - NDVI percentiles / std over all clear dates
  ia_std                         - std over years of the yearly Jan-Apr median NDVI
  mndwi_med                      - median MNDWI over clear scenes
"""
import warnings

import numpy as np

FEATURES = ["c0", "amp", "phase_cos", "phase_sin", "p10", "p50", "p90", "std", "ia_std", "mndwi_med"]
FOREST, NONFOREST, UNLABELED = 1, 0, 255
ESRI_TREES = 2
ESRI_NONFOREST = (5, 7, 11)  # Crops, Built area, Rangeland


# ---------------------------------------------------------------- harmonic regression
def design_row(doy: float) -> np.ndarray:
    w = 2 * np.pi * doy / 365.25
    return np.array([1.0, np.cos(w), np.sin(w)])


class HarmonicAccumulator:
    """Per-pixel normal equations (X'X, X'y) for NDVI = c0 + a1 cos + b1 sin, updated one date at a time."""

    def __init__(self, shape: tuple[int, int], n_terms: int = 3):
        self.xtx = np.zeros((n_terms, n_terms) + shape, np.float64)
        self.xty = np.zeros((n_terms,) + shape, np.float64)
        self.n = np.zeros(shape, np.int32)

    def add(self, y: np.ndarray, doy: float) -> None:
        m = np.isfinite(y)
        x = design_row(doy)
        yy = np.where(m, y, 0.0)
        for i in range(len(x)):
            self.xty[i] += x[i] * yy
            for j in range(i, len(x)):
                self.xtx[i, j] += x[i] * x[j] * m
        self.n += m

    def solve(self, min_obs: int = 8, ridge: float = 1e-9) -> np.ndarray:
        """(3, H, W) coefficients [c0, a1, b1]; NaN where fewer than ``min_obs`` observations."""
        k = self.xtx.shape[0]
        H, W = self.n.shape
        a = np.zeros((H * W, k, k))
        for i in range(k):
            for j in range(i, k):
                a[:, i, j] = a[:, j, i] = self.xtx[i, j].ravel()
        a += ridge * np.eye(k)[None]
        b = self.xty.reshape(k, -1).T[..., None]
        ok = self.n.ravel() >= min_obs
        out = np.full((H * W, k), np.nan)
        out[ok] = np.linalg.solve(a[ok], b[ok])[..., 0]
        return out.T.reshape(k, H, W)


# ---------------------------------------------------------------- phenology features
def group_same_day(times: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """(unique_dates[D] datetime64[D], list of scene-index arrays per date)."""
    d = times.astype("datetime64[D]")
    uniq, inv = np.unique(d, return_inverse=True)
    return uniq, [np.where(inv == k)[0] for k in range(len(uniq))]


def daily_ndvi(ndvi: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """(D,H,W) float32: same-day scenes averaged over their valid (non-NaN) NDVI."""
    out = np.full((len(groups),) + ndvi.shape[1:], np.nan, np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k, g in enumerate(groups):
            out[k] = ndvi[g[0]] if len(g) == 1 else np.nanmean(ndvi[g], axis=0)
    return out


def phenology_features(sc: dict, times: np.ndarray, fcfg: dict, season: tuple[int, int] = (1, 4),
                       chunk_rows: int = 64) -> dict[str, np.ndarray]:
    """Compute the FEATURES maps (float32, NaN where < ``min_obs`` clear dates) from scene indices ``sc``."""
    uniq, groups = group_same_day(times)
    dn = daily_ndvi(sc["ndvi"], groups)
    D, H, W = dn.shape
    dd = uniq.astype("datetime64[D]")
    doy = (dd - dd.astype("datetime64[Y]")).astype(int) + 1.0
    years = dd.astype("datetime64[Y]").astype(int) + 1970
    months = dd.astype("datetime64[M]").astype(int) % 12 + 1

    acc = HarmonicAccumulator((H, W))
    for k in range(D):
        acc.add(dn[k], doy[k])
    c0, a1, b1 = acc.solve(fcfg["min_obs"])
    amp = np.hypot(a1, b1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pc, ps = a1 / amp, b1 / amp

    feats = {k: np.full((H, W), np.nan, np.float32) for k in ("p10", "p50", "p90", "std", "ia_std", "mndwi_med")}
    yrs = np.unique(years)
    in_season = (months >= season[0]) & (months <= season[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for r0 in range(0, H, chunk_rows):
            sl = slice(r0, min(H, r0 + chunk_rows))
            blk = dn[:, sl]
            q = np.nanpercentile(blk, [10, 50, 90], axis=0)
            feats["p10"][sl], feats["p50"][sl], feats["p90"][sl] = q
            feats["std"][sl] = np.nanstd(blk, axis=0)
            ym = np.stack([np.nanmedian(blk[(years == y) & in_season], axis=0)
                           if ((years == y) & in_season).any() else np.full(blk.shape[1:], np.nan, np.float32)
                           for y in yrs])
            ok = np.isfinite(ym).sum(axis=0) >= 3
            feats["ia_std"][sl] = np.where(ok, np.nanstd(ym, axis=0), np.nan)
            feats["mndwi_med"][sl] = np.nanmedian(sc["mndwi"][:, sl], axis=0)
    low = acc.n < fcfg["min_obs"]
    for k in feats:
        feats[k][low] = np.nan
    out = {"c0": c0.astype(np.float32), "amp": amp.astype(np.float32), "phase_cos": pc.astype(np.float32),
           "phase_sin": ps.astype(np.float32), **feats}
    out["phase"] = np.arctan2(b1, a1).astype(np.float32)
    out["n_obs"] = acc.n.astype(np.int16)
    return out


# ---------------------------------------------------------------- Esri multi-year summaries
def esri_year_counts(raws: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(# years with Esri Trees, # years with Crops/Built/Rangeland) from raw Esri code rasters."""
    trees = sum((r == ESRI_TREES).astype(np.int8) for r in raws)
    non = sum(np.isin(r, ESRI_NONFOREST).astype(np.int8) for r in raws)
    return trees.astype(np.uint8), non.astype(np.uint8)


def block_ids(shape: tuple[int, int], res: float, x0: float, y0: float, block_m: float) -> np.ndarray:
    """Integer id of the block_m x block_m square containing each pixel (unique per block)."""
    r, c = np.indices(shape)
    bx = ((c * res) // block_m).astype(np.int64)
    by = ((r * res) // block_m).astype(np.int64)
    return bx * 100000 + by


def training_set(feat: np.ndarray, trees_years: np.ndarray, nonforest_years: np.ndarray, exclude: np.ndarray,
                 blocks: np.ndarray, fcfg: dict, rng: np.random.Generator):
    """Balanced weak-label sample. feat (F,H,W); returns X (n,F), y (n,), block (n,)."""
    ok = np.isfinite(feat).all(axis=0) & ~exclude
    pos = np.flatnonzero((ok & (trees_years >= fcfg["esri_pos_min_years"])).ravel())
    neg = np.flatnonzero((ok & (nonforest_years >= fcfg["esri_neg_min_years"])).ravel())
    cap = fcfg["cap_per_class_per_site"]
    pos = rng.choice(pos, min(cap, pos.size), replace=False) if pos.size else pos
    neg = rng.choice(neg, min(cap, neg.size), replace=False) if neg.size else neg
    idx = np.concatenate([pos, neg])
    f2 = feat.reshape(feat.shape[0], -1)
    return f2[:, idx].T, np.concatenate([np.ones(pos.size, int), np.zeros(neg.size, int)]), blocks.ravel()[idx]


def make_rf(fcfg: dict, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**fcfg["rf"], n_jobs=-1, random_state=seed)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"n": int(y.size), "accuracy": float((y == p).mean()), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12)}


def evaluate_rf(sets: dict, fcfg: dict, seed: int) -> dict:
    """Spatial-block CV (1 km blocks, GroupKFold) on the pooled sample + leave-one-site-out.

    ``sets``: {site: (X, y, block)}. Block ids are made unique per site.
    """
    from sklearn.model_selection import GroupKFold
    names = list(sets)
    X = np.concatenate([sets[s][0] for s in names])
    y = np.concatenate([sets[s][1] for s in names])
    site = np.concatenate([np.full(len(sets[s][1]), i) for i, s in enumerate(names)])
    grp = np.unique(np.stack([site, np.concatenate([sets[s][2] for s in names])], 1), axis=0, return_inverse=True)[1].ravel()
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=fcfg["cv_folds"]).split(X, y, grp):
        pred[te] = make_rf(fcfg, seed).fit(X[tr], y[tr]).predict(X[te])
    out = {"block_cv": _metrics(y, pred), "block_cv_by_site": {s: _metrics(y[site == i], pred[site == i])
                                                             for i, s in enumerate(names)},
           "leave_one_site_out": {}}
    for i, s in enumerate(names if len(names) > 1 else []):
        m = make_rf(fcfg, seed).fit(X[site != i], y[site != i])
        out["leave_one_site_out"][s] = _metrics(y[site == i], m.predict(X[site == i]))
    full = make_rf(fcfg, seed).fit(X, y)
    out["feature_importance"] = {f: float(v) for f, v in zip(FEATURES, full.feature_importances_)}
    out["n_train"] = {s: {"forest": int((sets[s][1] == 1).sum()), "nonforest": int((sets[s][1] == 0).sum())}
                      for s in names}
    return out, full


def predict_prob(model, feat: np.ndarray, batch: int = 200000) -> np.ndarray:
    """P(forest) per pixel, NaN where any feature is NaN."""
    F, H, W = feat.shape
    f2 = feat.reshape(F, -1).T
    ok = np.isfinite(f2).all(axis=1)
    out = np.full(f2.shape[0], np.nan, np.float32)
    idx = np.flatnonzero(ok)
    for i in range(0, idx.size, batch):
        j = idx[i:i + batch]
        out[j] = model.predict_proba(f2[j])[:, 1]
    return out.reshape(H, W)


# ---------------------------------------------------------------- rule / consensus forest
def forest_maps(prob: np.ndarray, trees_years: np.ndarray, fcfg: dict) -> dict[str, np.ndarray]:
    """forest_c1 (prob >= thr), forest_esri (trees_years >= k), rule forest (= c1) and consensus map
    (1 forest, 0 confidently non-forest, 255 disagreement / probability margin / no features)."""
    thr, margin = fcfg["prob_threshold"], fcfg["prob_margin"]
    has = np.isfinite(prob)
    c1 = has & (prob >= thr)
    esri = trees_years >= fcfg["esri_trees_min_years"]
    cons = np.full(prob.shape, UNLABELED, np.uint8)
    cons[c1 & esri] = FOREST
    cons[has & ~c1 & ~esri] = NONFOREST
    cons[~has | (np.abs(prob - thr) < margin)] = UNLABELED
    return {"forest_c1": c1, "forest_esri": esri, "forest_rule": c1, "forest_consensus": cons}
