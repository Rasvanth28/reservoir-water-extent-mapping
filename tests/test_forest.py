import numpy as np

from reservoir_cd.preprocessing import forest, labels


def test_harmonic_normal_equations_match_batch_lstsq():
    rng = np.random.default_rng(0)
    H, W, D = 3, 4, 60
    doy = np.sort(rng.uniform(1, 366, D))
    c0, a1, b1 = rng.uniform(0.2, 0.5, (H, W)), rng.uniform(-.2, .2, (H, W)), rng.uniform(-.2, .2, (H, W))
    w = 2 * np.pi * doy / 365.25
    y = c0 + a1 * np.cos(w)[:, None, None] + b1 * np.sin(w)[:, None, None] + rng.normal(0, 0.02, (D, H, W))
    y = y.astype(np.float32)
    y[rng.random((D, H, W)) < 0.3] = np.nan          # gaps (clouds)
    acc = forest.HarmonicAccumulator((H, W))
    for k in range(D):                                 # streamed one date at a time
        acc.add(y[k], doy[k])
    got = acc.solve(min_obs=8)
    for i in range(H):
        for j in range(W):
            m = np.isfinite(y[:, i, j])
            X = np.stack([np.ones(m.sum()), np.cos(w[m]), np.sin(w[m])], 1)
            ref = np.linalg.lstsq(X, y[m, i, j].astype(np.float64), rcond=None)[0]
            np.testing.assert_allclose(got[:, i, j], ref, atol=1e-6)
    # too few observations -> NaN
    acc2 = forest.HarmonicAccumulator((1, 1))
    for k in range(5):
        acc2.add(np.array([[0.4]], np.float32), doy[k])
    assert np.isnan(acc2.solve(min_obs=8)).all()


def test_same_day_scenes_counted_once():
    t = np.array(["2020-01-01T05:00", "2020-01-01T05:01", "2020-01-17T05:00"], dtype="datetime64[ns]")
    uniq, groups = forest.group_same_day(t)
    assert len(uniq) == 2 and [len(g) for g in groups] == [2, 1]
    nd = np.array([[[0.2]], [[np.nan]], [[0.6]]], np.float32)
    d = forest.daily_ndvi(nd, groups)
    assert d.shape == (2, 1, 1) and np.isclose(d[0, 0, 0], 0.2) and np.isclose(d[1, 0, 0], 0.6)
    nd2 = np.array([[[0.2]], [[0.4]], [[0.6]]], np.float32)
    assert np.isclose(forest.daily_ndvi(nd2, groups)[0, 0, 0], 0.3)


FCFG = {"prob_threshold": 0.5, "prob_margin": 0.15, "esri_trees_min_years": 4}


def test_static_forest_consensus_logic():
    prob = np.array([0.9, 0.9, 0.2, 0.2, 0.55, 0.45, np.nan, 0.8])
    trees = np.array([7, 1, 0, 6, 7, 0, 7, 4], np.uint8)
    m = forest.forest_maps(prob, trees, FCFG)
    assert m["forest_c1"].tolist() == [1, 1, 0, 0, 1, 0, 0, 1]
    assert m["forest_esri"].tolist() == [1, 0, 0, 1, 1, 0, 1, 1]
    # both forest -> 1; both non-forest -> 0; disagreement -> 255; within margin -> 255; NaN -> 255
    assert m["forest_consensus"].tolist() == [1, 255, 0, 255, 255, 255, 255, 1]
    assert (m["forest_rule"] == m["forest_c1"]).all()


def test_consensus_label_static_uses_forest_map():
    cfg = {"consensus": {"min_clear": 2, "min_agreement": 0.75},
           "water": {"confidence_margin": 0.1}, "bare": {"ndvi_max": 0.35}}
    rule = np.array([[labels.FOREST, labels.FOREST, labels.OTHER, labels.OTHER, labels.OTHER, labels.WATER]], np.uint8)
    fcons = np.array([[1, 255, 0, 255, 0, 0]], np.uint8)
    mndwi = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.5]])
    n = np.full((1, 6), 5)
    study = np.ones((1, 6), bool)
    agree = np.array([[0.1, 0.1, 0.9, 0.9, 0.1, 0.9]], np.float32)
    ext = np.zeros((1, 6), bool)
    # non-Esri year: Other needs static non-forest AND temporal agreement; Forest only static
    out = labels.consensus_label_static(rule, mndwi, 0.0, n, study, agree, None, ext, fcons, cfg)
    assert out.tolist() == [[1, 255, 0 + 3, 255, 255, 0]]
    # Esri year: Other additionally requires Esri Crops/Rangeland/Built (= 3)
    esri = np.array([[1, 1, 3, 3, 2, 0]], np.uint8)
    out = labels.consensus_label_static(rule, mndwi, 0.0, n, study, np.ones((1, 6), np.float32), esri, ext, fcons, cfg)
    assert out.tolist() == [[1, 255, 3, 255, 255, 0]]
    esri2 = np.array([[1, 1, 2, 3, 3, 0]], np.uint8)   # Esri says Trees for the Other pixel -> dropped
    out = labels.consensus_label_static(rule, mndwi, 0.0, n, study, np.ones((1, 6), np.float32), esri2, ext, fcons, cfg)
    assert out[0, 2] == 255


def test_rule_label_static_precedence():
    cfg = {"bare": {"ndvi_max": 0.35}}
    mndwi = np.array([[0.5, -0.5, -0.5, -0.5, -0.5]])
    ndvi = np.array([[0.1, 0.1, 0.6, 0.6, np.nan]])
    ext = np.array([[1, 1, 0, 0, 0]], bool)
    fo = np.array([[1, 1, 1, 0, 0]], bool)
    out = labels.rule_label_static(mndwi, ndvi, 0.0, ext, fo, cfg)
    assert out.tolist() == [[0, 2, 1, 3, 255]]
