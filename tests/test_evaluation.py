import numpy as np
import pandas as pd

from reservoir_cd.evaluation import metrics as M
from reservoir_cd.evaluation.stats import BlockBootstrap, holm, mcnemar


def test_ratio_single_stratum_is_sample_proportion():
    y = np.array([1, 0, 1, 1], float)
    R, se = M.ratio(y, np.ones(4), np.zeros(4, int), np.array([1e9]))
    assert np.isclose(R, 0.75)
    assert np.isclose(se, np.sqrt(0.25 / 4))  # s2 = 0.25 (ddof 1), fpc ~ 1


def test_ratio_weights_strata_by_population():
    # stratum 0: all correct, N=900; stratum 1: all wrong, N=100 -> OA = 0.9 whatever the sample sizes
    y = np.array([1, 1, 0, 0, 0, 0], float)
    h = np.array([0, 0, 1, 1, 1, 1])
    R, se = M.ratio(y, np.ones(6), h, np.array([900.0, 100.0]))
    assert np.isclose(R, 0.9) and np.isclose(se, 0)


def test_ratio_full_census_has_zero_variance():
    y = np.array([1, 0, 1], float)
    _, se = M.ratio(y, np.ones(3), np.zeros(3, int), np.array([3.0]))
    assert se == 0


def test_users_accuracy_matches_stehman_definition():
    # map water at units 0,1,2; reference water at 0,3. UA = 1/3 within one stratum
    ref = pd.DataFrame({"label_t1": ["Water", "Bare", "Other", "Water"], "label_t2": ["NA"] * 4,
                        "mixed_t1": [False] * 4, "mixed_t2": [False] * 4})
    pred = pd.DataFrame({"t1": ["Water", "Water", "Water", "Non-water"], "t2": ["Water"] * 4})
    terms = M.unit_terms(ref, pred, pure_only=False, land_classes=False)
    est = M.estimate(terms, np.zeros(4, int), np.array([1e9]))
    assert np.isclose(est["water_UA"][0], 1 / 3) and np.isclose(est["water_PA"][0], 0.5)
    assert np.isclose(est["water_OA"][0], 0.25)  # NA dates at t2 are left out


def test_mcnemar_and_holm():
    a = np.array([True] * 10 + [False] * 0)
    b = np.array([False] * 10)
    r = mcnemar(a, b)
    assert r["a_only"] == 10 and r["b_only"] == 0 and r["p"] < 0.01
    assert np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])


def test_block_bootstrap_keeps_strata_and_blocks():
    h = np.array([0, 0, 0, 1, 1])
    blk = np.array(["a", "a", "b", "c", "d"])
    idx = BlockBootstrap(h, blk, seed=0).sample()
    assert set(h[idx]) == {0, 1}
    s0 = idx[h[idx] == 0]
    assert sorted(set(s0)) in ([0, 1], [2], [0, 1, 2])  # block "a" moves as a whole
