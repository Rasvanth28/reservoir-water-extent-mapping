import numpy as np

from reservoir_cd.preprocessing import composite, indices, qa
from reservoir_cd.preprocessing.labels import consensus_label, robust_scene_threshold, rule_label
from reservoir_cd.preprocessing.reservoir_extent import buffer_mask, extent_from_frequency


def test_qa_bits_decode():
    vals = np.array([1, 2, 4, 8, 16, 32, 64, 128, 21824.0, 21762.0])  # float-stored
    for name, bit in qa.BITS.items():
        got = qa.decode_bit(vals, name)
        assert got[bit] and got[:8].sum() == 1, name
    inv = qa.invalid_raw(vals)
    assert inv[:5].all() and not inv[5:8].any()  # snow/clear/water are valid
    assert not inv[8]  # 21824 = 0b101010101000000: clear bit only (+ confidences)
    assert inv[9]      # 21762 has the dilated-cloud bit (bit 1)


def test_qa_opening_and_buffer():
    q = np.full((20, 20), 64.0)  # clear
    q[10, 10] = 8                # isolated cloud speck -> removed by opening
    assert qa.valid_mask(q, 3, 2).all()
    q[2:8, 2:8] = 8              # 6x6 cloud survives, dilated by 2 px
    v = qa.valid_mask(q, 3, 2)
    assert not v[2:8, 2:8].any()
    assert not v[9, 5] and v[11, 5] and v[5, 12]


def test_indices_known_values():
    a, b = np.array([0.3, 0.0]), np.array([0.1, 0.0])
    out = indices.mndwi(a, b)
    assert np.isclose(out[0], 0.5) and np.isnan(out[1])  # zero denominator guarded
    assert np.isclose(indices.ndvi(np.array([0.6]), np.array([0.2]))[0], 0.5)
    # bsi: ((swir+red)-(nir+blue))/((swir+red)+(nir+blue)) = (0.5-0.3)/0.8
    assert np.isclose(indices.bsi(np.array([0.1]), np.array([0.2]), np.array([0.2]), np.array([0.3]))[0], 0.25)


def test_medoid_picks_real_observation():
    rng = np.random.default_rng(0)
    T, B, H, W = 5, 6, 4, 4
    refl = rng.random((T, B, H, W))
    valid = np.ones((T, H, W), bool)
    valid[0, 0, 0] = False
    valid[:, 1, 1] = False  # no clear obs
    times = np.array([f"2020-01-{d:02d}" for d in range(1, 6)], dtype="datetime64[ns]")
    out = composite.medoid_composite(refl, valid, times)
    for r in range(H):
        for c in range(W):
            if r == 1 and c == 1:
                assert out["n_clear"][r, c] == 0 and np.isnat(out["source_date"][r, c])
                assert np.isnan(out["bands"][:, r, c]).all()
                continue
            t = out["src_index"][r, c]
            assert valid[t, r, c]
            assert np.array_equal(out["bands"][:, r, c], refl[t, :, r, c])
            assert out["source_date"][r, c] == times[t]
    assert out["n_clear"][0, 0] == 4


def test_medoid_closest_to_median():
    # 3 obs, one band-vector outlier; medoid must be the middle one
    refl = np.zeros((3, 6, 1, 1))
    refl[0, :, 0, 0], refl[1, :, 0, 0], refl[2, :, 0, 0] = 0.1, 0.2, 0.9
    out = composite.medoid_composite(refl, np.ones((3, 1, 1), bool), np.arange(3).astype("datetime64[D]"))
    assert out["src_index"][0, 0] == 1


CFG = {"bare": {"ndvi_max": 0.25, "outside_enabled": False, "outside_ndvi_max": 0.15, "outside_bright_min": 0.25},
       "forest": {"ndvi_min": 0.6, "dry_ndvi_min": 0.2, "margin": 0.05},
       "water": {"confidence_margin": 0.1, "otsu_clamp": [-0.1, 0.3], "bimodality_min": 0.5},
       "consensus": {"min_clear": 2, "min_agreement": 0.75}}


def test_rule_label_precedence():
    # water, forest, bare-in-extent, other (not in extent), unlabeled (NaN), bare outside extent (off)
    m = np.array([0.5, -0.5, -0.5, -0.5, np.nan, -0.5])
    n = np.array([0.7, 0.7, 0.1, 0.3, 0.3, 0.1])
    p90 = np.array([0.8, 0.8, 0.3, 0.3, 0.3, 0.3])
    ext = np.array([True, False, True, False, False, False])
    br = np.full(6, 0.3)
    lab = rule_label(m, n, br, 0.0, ext, p90, CFG)
    assert lab.tolist() == [0, 1, 2, 3, 255, 3]


def test_bare_inside_extent_rule():
    ext = np.array([True, True, True, False, False])
    m = np.array([-0.5, -0.5, 0.5, -0.5, -0.5])
    n = np.array([0.20, 0.30, 0.05, 0.20, 0.10])  # in-ext low NDVI -> Bare; 0.30 -> not Bare; water wins
    p90 = np.full(5, 0.3)
    br = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
    assert rule_label(m, n, br, 0.0, ext, p90, CFG).tolist() == [2, 3, 0, 3, 3]
    cfg2 = {**CFG, "bare": {**CFG["bare"], "outside_enabled": True}}
    assert rule_label(m, n, br, 0.0, ext, p90, cfg2).tolist() == [2, 3, 0, 3, 2]  # outside: NDVI<0.15 & bright
    # annual forest needs both p90 and dry NDVI; Bare in extent beats Forest
    p90f = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
    nf = np.array([0.20, 0.10, 0.5, 0.5, 0.15])
    out = rule_label(np.full(5, -0.5), nf, br, 0.0, np.array([False, False, False, False, False]), p90f, CFG)
    assert out.tolist() == [1, 3, 1, 1, 3]  # dry NDVI < 0.2 -> not Forest


def test_bimodality_fallback():
    rng = np.random.default_rng(0)
    region = np.ones((100, 100), bool)
    bi = np.where(rng.random((100, 100)) < 0.3, rng.normal(0.3, 0.05, (100, 100)), rng.normal(-0.2, 0.05, (100, 100)))
    t, fb, eta = robust_scene_threshold(bi, region, 0.02, CFG)
    assert not fb and eta > 0.9 and -0.1 < t < 0.3
    uni = rng.normal(0.0, 0.1, (100, 100))  # unimodal -> low eta -> fallback to given value
    # a Gaussian has Otsu eta = 2/pi ~ 0.64, so it passes the default 0.5 but fails a stricter 0.7
    strict = {**CFG, "water": {**CFG["water"], "bimodality_min": 0.7}}
    t, fb, eta = robust_scene_threshold(uni, region, 0.02, strict)
    assert fb and 0.5 < eta < 0.7 and t == 0.02
    # clean bimodal but raw Otsu outside clamp (e.g. land at -0.9, water at -0.5) -> fallback
    low = np.where(rng.random((100, 100)) < 0.3, rng.normal(-0.5, 0.02, (100, 100)), rng.normal(-0.9, 0.02, (100, 100)))
    t, fb, eta = robust_scene_threshold(low, region, 0.02, CFG)
    assert fb and t == 0.02
    t, fb, _ = robust_scene_threshold(bi, region, None, CFG)
    assert not fb
    t, fb, _ = robust_scene_threshold(uni, region, None, strict)
    assert fb and np.isnan(t)


def test_consensus_filters():
    # classes: Water, Water(near t), Forest(far from p90 margin), Forest(inside margin), Other
    rule = np.array([0, 0, 1, 1, 3], dtype=np.uint8)
    mndwi = np.array([0.5, 0.05, -0.5, -0.5, -0.5])
    p90 = np.array([0.9, 0.9, 0.9, 0.62, 0.3])
    ext = np.zeros(5, bool)
    n_clear = np.full(5, 5)
    study = np.ones(5, bool)
    agree = np.array([1.0, 1.0, 1.0, 1.0, 0.5])
    out = consensus_label(rule, mndwi, 0.0, n_clear, study, agree, None, ext, p90, CFG)
    assert out.tolist() == [0, 255, 1, 255, 255]
    # MNDWI margin does NOT apply to Forest/Other far from water (here near threshold but not water-dependent)
    out_n = consensus_label(np.array([3], np.uint8), np.array([0.01]), 0.0, np.array([5]), np.ones(1, bool),
                            np.array([1.0], np.float32), None, np.zeros(1, bool), np.array([0.3]), CFG)
    assert out_n.tolist() == [3]
    # Esri years: rule must equal Esri, except Bare inside extent which uses temporal agreement only
    rule2 = np.array([0, 2, 2, 2, 1], dtype=np.uint8)
    esri = np.array([0, 3, 3, 2, 3], dtype=np.uint8)
    ext2 = np.array([False, True, True, False, False])
    agree2 = np.array([1.0, 0.9, 0.5, 1.0, 1.0])
    mndwi2 = np.array([0.5, -0.5, -0.5, -0.5, -0.5])
    out2 = consensus_label(rule2, mndwi2, 0.0, np.full(5, 5), np.ones(5, bool), agree2, esri, ext2,
                           np.full(5, 0.9), CFG)
    assert out2.tolist() == [0, 2, 255, 2, 255]
    # Bare in extent but within the MNDWI margin is dropped (water-dependent class)
    out3 = consensus_label(np.array([2], np.uint8), np.array([0.05]), 0.0, np.array([5]), np.ones(1, bool),
                           np.array([1.0], np.float32), np.array([3], np.uint8), np.ones(1, bool), np.array([0.3]), CFG)
    assert out3.tolist() == [255]


def test_reservoir_extent_connected_component():
    freq = np.zeros((40, 40), np.float32)
    freq[10:30, 10:20] = 0.9          # permanent body (largest)
    freq[10:30, 20:24] = 0.2          # seasonal margin touching it -> kept
    freq[5:8, 35:38] = 1.0            # smaller permanent tank, detached
    freq[33:36, 5:30] = 0.3           # detached seasonal paddy strip -> dropped
    freq[2, 2] = 0.06                 # isolated speck -> dropped
    freq[12, 14] = 0.0                # hole inside body -> filled by closing
    perm, ext = extent_from_frequency(freq, 0.75, 0.05, 1)
    assert perm[15, 15] and not perm[6, 36] and not perm[15, 21]
    assert ext[15, 22] and ext[12, 14]
    assert not ext[6, 36] and not ext[34, 10] and not ext[2, 2]
    # diagonal-only (8-connected) contact keeps the component
    f2 = np.zeros((10, 10), np.float32)
    f2[2:5, 2:5] = 0.9
    f2[5, 5] = 0.1
    _, e2 = extent_from_frequency(f2, 0.75, 0.05, 0)
    assert e2[5, 5]
    # study buffer: 2 px at 30 m = 60 m
    b = buffer_mask(np.pad(np.ones((1, 1), bool), 10), 60.0, 30.0)
    assert b.sum() == 13  # discs of radius 2 px: 13 pixels
