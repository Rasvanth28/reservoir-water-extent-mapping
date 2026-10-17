"""Spatial-block splits within each site, and leave-one-site-out folds (E1).

Within a site the unit of assignment is an *eval window* (256x256 on the non-overlapping eval grid,
the last row/column shifted back to the scene edge). A seeded draw picks val and test windows
(preferring windows that contain reservoir extent so val/test see water), which must not overlap
each other. Every other window position is ``train`` unless it lies closer than ``gap_px``
(Chebyshev distance between footprints) to a val/test window, in which case it is ``buffer`` and
is never used. So no train chip shares a pixel with a val/test chip, and there is a gap of
``gap_px`` pixels between them. Sites too small to host disjoint val + test + train windows
(Pillur: 372x367 px < 2 chips) get no within-site val/test: all their windows are ``train``.
Val/test windows are chosen greedily to lose the fewest train windows, among windows with reservoir
extent and true water change (so val/test contain water and loss/gain pixels).

Output columns (Parquet): site, row0, col0, split (within-site: train|val|test|buffer) and one
column ``loso_<Site>`` per site for the E1 fold that holds <Site> out: held-out site -> ``test`` for
every window; other sites -> their within-site ``train`` / ``val`` windows, everything else ``none``.
"""
import numpy as np
import pandas as pd


def _overlap(a, b, size: int, gap: int = 0) -> bool:
    """Footprints (r, c, size) are closer than ``gap`` (gap=0: share at least one pixel)."""
    return abs(a[0] - b[0]) < size + gap and abs(a[1] - b[1]) < size + gap


def split_site(pos: pd.DataFrame, pairs: pd.DataFrame, cfg: dict) -> pd.Series:
    """Greedy, loss-minimising choice of val/test windows (see module docstring)."""
    sc, size, gap = cfg["splits"], cfg["chip_size"], cfg["splits"]["gap_px"]
    rng = np.random.default_rng(sc["seed"])
    g = pairs.groupby(["row0", "col0"])
    lf = g.labeled_frac.mean()
    chg = (g.loss_px.sum() + g.gain_px.sum())
    p = pos.copy()
    p["lf"] = [lf.get((r, c), 0.0) for r, c in zip(p.row0, p.col0)]
    p["chg"] = [chg.get((r, c), 0) for r, c in zip(p.row0, p.col0)]
    cand = p[p.in_eval_grid & (p.lf >= sc["min_labeled_frac"])]
    good = cand[(cand.max_extent_px >= sc["min_water_px"]) & (cand.chg >= sc["min_change_px"])]
    pool = good if len(good) >= 3 else cand
    n_val = max(1, round(sc["val_frac"] * len(cand)))
    n_test = max(1, round(sc["test_frac"] * len(cand)))
    wins = list(zip(pos.row0, pos.col0))

    def alive(chosen):  # train windows not overlapping / within gap of any chosen window
        return sum(all(not _overlap(w, c, size, gap) for c in chosen) for w in wins if w not in chosen)

    chosen: list[tuple[int, int]] = []
    names: list[str] = []
    for name in ["test"] * n_test + ["val"] * n_val:
        opts = [(int(r), int(c)) for r, c in pool[["row0", "col0"]].to_numpy()
                if all(not _overlap((r, c), x, size) for x in chosen)]
        if not opts:
            break
        score = [(alive(chosen + [o]), int(pool[(pool.row0 == o[0]) & (pool.col0 == o[1])].chg.iloc[0]),
                  rng.random()) for o in opts]
        best = opts[max(range(len(opts)), key=lambda i: score[i])]
        chosen.append(best); names.append(name)
    split = pd.Series("train", index=pos.index)
    if len(names) < n_test + n_val or alive(chosen) == 0:
        return split  # scene too small for leak-free val + test + train: everything stays train
    for i, w in enumerate(wins):
        if w in chosen:
            split.iloc[i] = names[chosen.index(w)]
        elif any(_overlap(w, c, size, gap) for c in chosen):
            split.iloc[i] = "buffer"
    return split


def make_splits(positions: pd.DataFrame, pairs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = []
    for site, pos in positions.groupby("site", sort=False):
        pos = pos.reset_index(drop=True)
        s = split_site(pos, pairs[pairs.site == site], cfg)
        out.append(pd.DataFrame({"site": site, "row0": pos.row0, "col0": pos.col0, "split": s}))
    sp = pd.concat(out, ignore_index=True)
    for held in sp.site.unique():
        sp[f"loso_{held}"] = np.where(sp.site == held, "test",
                                      np.where(sp.split.isin(["train", "val"]), sp.split, "none"))
    return sp


def assert_no_leakage(sp: pd.DataFrame, size: int) -> None:
    """Raise if any train window shares a pixel with a val/test window (within-site split)."""
    for site, d in sp.groupby("site"):
        tr = d[d.split == "train"][["row0", "col0"]].to_numpy()
        ev = d[d.split.isin(["val", "test"])][["row0", "col0"]].to_numpy()
        for r, c in ev:
            if len(tr) and ((np.abs(tr[:, 0] - r) < size) & (np.abs(tr[:, 1] - c) < size)).any():
                raise AssertionError(f"leak in {site} at {(r, c)}")
