"""Paired tests and the spatial block bootstrap (EDD 3.7)."""
import numpy as np
from scipy.stats import binomtest


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on paired correct/incorrect outcomes (two-sided binomial on discordant pairs)."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"n": int(len(correct_a)), "a_only": b, "b_only": c,
            "acc_a": float(correct_a.mean()), "acc_b": float(correct_b.mean()), "p": float(p)}


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.minimum(1, (len(p) - np.arange(len(p))) * p[order])
    adj = np.maximum.accumulate(adj)
    out = np.empty_like(adj)
    out[order] = adj
    return out


class BlockBootstrap:
    """Stratified spatial block bootstrap: within each stratum, resample its 1 km blocks with replacement
    and keep all of their points, so spatially clustered points move together."""

    def __init__(self, h: np.ndarray, block: np.ndarray, seed: int):
        self.rng = np.random.default_rng(seed)
        self.groups = []
        for s in np.unique(h):
            units = np.flatnonzero(h == s)
            blocks = block[units]
            self.groups.append([units[blocks == b] for b in np.unique(blocks)])

    def sample(self) -> np.ndarray:
        parts = []
        for g in self.groups:
            pick = self.rng.integers(0, len(g), len(g))
            parts.extend(g[i] for i in pick)
        return np.concatenate(parts)
