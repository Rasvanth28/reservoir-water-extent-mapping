"""Design-based accuracy estimates for the stratified validation sample.

The strata (S1-S7, from the rule labels) are not the classes of the map being assessed, so every
estimate is a ratio R = Y/X of population totals, estimated with the stratified combined ratio
estimator and its linearised variance (Stehman 2014, RSE 35, eqs. 2-3; Olofsson et al. 2014 is the
special case strata = map classes):

    R = sum_h N_h ybar_h / sum_h N_h xbar_h
    V(R) = 1/X^2 sum_h N_h^2 (1 - n_h/N_h) s2_zh / n_h,   z_u = y_u - R x_u

A sample unit is a (pixel, year pair) point. Per-date metrics put both dates into one unit
(y and x count over the dates that are valid), so they estimate accuracy over (pixel, pair, date).
"""
import numpy as np
import pandas as pd

from reservoir_cd.evaluation.sampling import CLASSES

CHANGE = ["stable", "loss", "gain"]


def ratio(y: np.ndarray, x: np.ndarray, h: np.ndarray, N: np.ndarray) -> tuple[float, float]:
    """Stratified ratio estimate and standard error. `h` = stratum code per unit (0..H-1), `N` = population
    per stratum. Strata without sample units contribute nothing."""
    H = len(N)
    n = np.bincount(h, minlength=H).astype(float)
    s = n > 0
    ybar = np.divide(np.bincount(h, y, H), n, out=np.zeros(H), where=s)
    xbar = np.divide(np.bincount(h, x, H), n, out=np.zeros(H), where=s)
    X = (N * xbar).sum()
    if X <= 0:
        return np.nan, np.nan
    R = (N * ybar).sum() / X
    z = y - R * x
    zbar = np.divide(np.bincount(h, z, H), n, out=np.zeros(H), where=s)
    ss = np.bincount(h, (z - zbar[h]) ** 2, H)
    s2 = np.divide(ss, n - 1, out=np.zeros(H), where=n > 1)
    fpc = np.clip(1 - np.divide(n, N, out=np.ones(H), where=N > 0), 0, 1)
    var = (N ** 2 * fpc * np.divide(s2, n, out=np.zeros(H), where=s)).sum() / X ** 2
    return float(R), float(np.sqrt(var))


def f1(ua: float, pa: float) -> float:
    return 2 * ua * pa / (ua + pa) if ua + pa > 0 else np.nan


def water_change(a: pd.Series, b: pd.Series) -> pd.Series:
    wa, wb = a.eq("Water"), b.eq("Water")
    return pd.Series("stable", index=a.index).mask(wa & ~wb, "loss").mask(~wa & wb, "gain")


def unit_terms(ref: pd.DataFrame, pred: pd.DataFrame, pure_only: bool, land_classes: bool) -> dict:
    """Per-unit (y, x) numerator/denominator pairs for every metric.

    `ref` holds label_t1/t2 and mixed_t1/t2 (interpreter); `pred` holds t1/t2 (map). A date is valid
    when the interpreter gave a class (not NA), the map has a label and, with `pure_only`, the square
    is not mixed. Change metrics need both dates valid.
    """
    valid, refv, rw, mw, rc, mc = {}, {}, {}, {}, {}, {}
    for t in ("t1", "t2"):
        r, m = ref[f"label_{t}"], pred[t]
        v = r.isin(CLASSES)
        if pure_only:
            v &= ~ref[f"mixed_{t}"].astype(bool)
        refv[t] = v.to_numpy()
        valid[t], rw[t], mw[t], rc[t], mc[t] = (v & m.ne("")).to_numpy(), r.eq("Water").to_numpy(), \
            m.eq("Water").to_numpy(), r, m

    terms = {}

    def add(name, per_date):
        y = sum(per_date(t)[0] for t in ("t1", "t2")).astype(float)
        x = sum(per_date(t)[1] for t in ("t1", "t2")).astype(float)
        terms[name] = (y, x)

    add("coverage", lambda t: (valid[t], refv[t]))  # share of the population the map labels at all
    add("water_OA", lambda t: (valid[t] & (rw[t] == mw[t]), valid[t]))
    add("water_UA", lambda t: (valid[t] & mw[t] & rw[t], valid[t] & mw[t]))
    add("water_PA", lambda t: (valid[t] & mw[t] & rw[t], valid[t] & rw[t]))
    if land_classes:
        add("class_OA", lambda t: (valid[t] & (rc[t] == mc[t]).to_numpy(), valid[t]))
        for c in CLASSES[1:]:
            add(f"{c}_UA", lambda t, c=c: (valid[t] & (mc[t] == c).to_numpy() & (rc[t] == c).to_numpy(),
                                           valid[t] & (mc[t] == c).to_numpy()))
            add(f"{c}_PA", lambda t, c=c: (valid[t] & (mc[t] == c).to_numpy() & (rc[t] == c).to_numpy(),
                                           valid[t] & (rc[t] == c).to_numpy()))

    v = valid["t1"] & valid["t2"]
    rch = water_change(ref.label_t1, ref.label_t2).to_numpy()
    mch = water_change(pred.t1, pred.t2).to_numpy()
    terms["change_OA"] = ((v & (rch == mch)).astype(float), v.astype(float))
    for c in ("loss", "gain"):
        hit = (v & (rch == c) & (mch == c)).astype(float)
        terms[f"{c}_UA"] = (hit, (v & (mch == c)).astype(float))
        terms[f"{c}_PA"] = (hit, (v & (rch == c)).astype(float))
    hit = (v & (rch != "stable") & (mch != "stable")).astype(float)
    terms["anychange_UA"] = (hit, (v & (mch != "stable")).astype(float))
    terms["anychange_PA"] = (hit, (v & (rch != "stable")).astype(float))
    return terms


F1_OF = {"water_F1": ("water_UA", "water_PA"), "loss_F1": ("loss_UA", "loss_PA"),
         "gain_F1": ("gain_UA", "gain_PA"), "anychange_F1": ("anychange_UA", "anychange_PA"),
         **{f"{c}_F1": (f"{c}_UA", f"{c}_PA") for c in CLASSES[1:]}}


def estimate(terms: dict, h: np.ndarray, N: np.ndarray, idx: np.ndarray | None = None) -> dict:
    """{metric: (estimate, se)}; F1 has no analytic SE (use the bootstrap). `idx` selects (bootstrap) units."""
    hh = h if idx is None else h[idx]
    out = {}
    for name, (y, x) in terms.items():
        out[name] = ratio(y if idx is None else y[idx], x if idx is None else x[idx], hh, N)
    for name, (u, p) in F1_OF.items():
        if u in out:
            out[name] = (f1(out[u][0], out[p][0]), np.nan)
    return out


def confusion(ref: pd.Series, pred: pd.Series, valid: pd.Series, labels: list[str]) -> pd.DataFrame:
    """Sample-count confusion matrix (rows map, columns reference) over valid units."""
    r, m = ref[valid], pred[valid]
    return pd.crosstab(m, r).reindex(index=labels, columns=labels, fill_value=0).rename_axis(
        index="map", columns="reference")
