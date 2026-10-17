import numpy as np

from reservoir_cd.grid import Grid
from reservoir_cd.validation import sampling, strata


def test_strata_tiny_example():
    # 1x? strip of 8 columns far enough apart to isolate effects (width=1)
    W = 20
    l1 = np.full((3, W), 1, dtype=np.uint8)  # all Forest
    l2 = l1.copy()
    l1[1, 0:3] = 0; l2[1, 0:3] = 0           # stable water block (edge with forest at col 3)
    l1[1, 6] = 0; l2[1, 6] = 3               # water loss
    l1[1, 9] = 3; l2[1, 9] = 0               # water gain
    l1[1, 12] = 1; l2[1, 12] = 2             # forest -> bare
    l1[1, 15] = 3; l2[1, 15] = 3             # stable other
    l1[1, 17] = 2; l2[1, 17] = 2             # stable bare
    elig = np.ones((3, W), bool)
    s = strata.assign_strata(l1, l2, elig, width=1, static_forest=False)  # legacy semantics
    assert s[1, 6] == 6 and s[1, 9] == 7 and s[1, 12] == 8
    assert s[1, 15] == 4 and s[1, 17] == 3
    assert s[1, 1] == 5 and s[1, 2] == 5     # stable water at edge: S5 priority over S1
    assert s[1, 19] == 2                     # far forest
    assert s[1, 6] != 5 and (s[0, 6] == 5)   # forest next to water-loss pixel is S5
    # exclusivity is by construction (one code per pixel); ineligible -> 0
    s2 = strata.assign_strata(l1, l2, np.zeros((3, W), bool))
    assert (s2 == 0).all()


def test_s8_requires_real_index_change():
    l1 = np.full((3, 6), 1, dtype=np.uint8)
    l2 = l1.copy()
    l1[1, 1], l2[1, 1] = 1, 3   # Forest -> Other with big index change
    l1[1, 4], l2[1, 4] = 1, 3   # Forest -> Other, threshold flicker (small index change)
    ok = np.zeros((3, 6), bool)
    ok[1, 1] = True
    s = strata.assign_strata(l1, l2, np.ones((3, 6), bool), 1, ok, static_forest=False)
    assert s[1, 1] == 8 and s[1, 4] == 0
    assert strata.assign_strata(l1, l2, np.ones((3, 6), bool), 1, static_forest=False)[1, 4] == 8  # no filter -> old behaviour


def test_interior_water_is_s1():
    l = np.full((9, 9), 0, dtype=np.uint8)
    s = strata.assign_strata(l, l.copy(), np.ones((9, 9), bool))
    assert (s == 1).all()


def test_sampling_spacing_and_shortfall():
    grid = Grid(0.0, 300.0, 30.0, 10, 10)
    sm = np.zeros((10, 10), np.uint8)
    sm[:, :] = 4
    sm[0, 0], sm[9, 9] = 6, 6   # only 2 S6 candidates
    alloc = {s: 0 for s in strata.STRATA}
    alloc.update(S6=5, S4=3)
    pts, rep = sampling.draw_sample([sm], alloc, grid, 90.0, seed=1)
    assert rep["sampled"]["S6"] == 2 and rep["shortfall_redistributed"] == 3
    xy = np.array([(p["x"], p["y"]) for p in pts])
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    assert (d[np.triu_indices(len(xy), 1)] >= 90).all()


def test_s8_static_forest_bare_other_with_delta_ndvi():
    # Bare<->Other needs a real NDVI change; Forest<->Bare/Other is never S8 (forest is static)
    l1 = np.full((3, 12), 3, dtype=np.uint8)
    l2 = l1.copy()
    l1[1, 1], l2[1, 1] = 2, 3   # Bare -> Other, big dNDVI
    l1[1, 4], l2[1, 4] = 3, 2   # Other -> Bare, small dNDVI (flicker)
    l1[1, 7], l2[1, 7] = 1, 2   # Forest -> Bare: not S8
    l1[1, 10], l2[1, 10] = 1, 3  # Forest -> Other: not S8
    ndvi1 = np.full((3, 12), 0.30); ndvi2 = ndvi1.copy()
    ndvi2[1, 1] = 0.60; ndvi2[1, 4] = 0.35; ndvi2[1, 7] = 0.0; ndvi2[1, 10] = 0.0
    change_ok = np.abs(ndvi1 - ndvi2) >= 0.15
    s = strata.assign_strata(l1, l2, np.ones((3, 12), bool), 1, change_ok)
    assert s[1, 1] == 8
    assert s[1, 4] == 0
    assert s[1, 7] == 0 and s[1, 10] == 0
    # without the NDVI requirement the flicker would be S8
    assert strata.assign_strata(l1, l2, np.ones((3, 12), bool), 1)[1, 4] == 8


def test_v2_partition_exclusive_no_s8():
    rng = np.random.default_rng(0)
    l1 = rng.integers(0, 4, (40, 40)).astype(np.uint8)
    l2 = rng.integers(0, 4, (40, 40)).astype(np.uint8)
    l1[:5, :5] = 255
    elig = np.ones((40, 40), bool)
    elig[:, -3:] = False
    s = strata.assign_strata_v2(l1, l2, elig)
    ok = elig & (l1 != 255) & (l2 != 255)
    assert ((s > 0) == ok).all()          # every eligible pixel in exactly one stratum (one code each)
    assert s.max() <= 7 and not (s == 8).any()
    assert (s[ok & (l1 == 0) & (l2 != 0)] == 6).all() and (s[ok & (l1 != 0) & (l2 == 0)] == 7).all()
    assert (s[ok & (l1 != 0) & (l2 != 0)] >= 2).all() and (s[ok & (l1 != 0) & (l2 != 0)] <= 5).all()


def test_v2_classes():
    W = 24
    l1 = np.full((3, W), 3, np.uint8)
    l2 = l1.copy()
    l1[1, 0:3] = l2[1, 0:3] = 0        # stable water, edge with Other at col 3
    l1[1, 8] = 1; l2[1, 8] = 1         # stable forest
    l1[1, 11] = 2; l2[1, 11] = 2       # stable bare
    l1[1, 14] = 2; l2[1, 14] = 3       # Bare -> Other  => S4
    l1[1, 17] = 1; l2[1, 17] = 3       # Forest -> Other => S4
    l1[1, 20] = 3; l2[1, 20] = 3       # stable Other => S4
    s = strata.assign_strata_v2(l1, l2, np.ones((3, W), bool))
    assert s[1, 1] == 5 and s[1, 8] == 2 and s[1, 11] == 3
    assert s[1, 14] == 4 and s[1, 17] == 4 and s[1, 20] == 4
    l = np.zeros((9, 9), np.uint8)
    assert (strata.assign_strata_v2(l, l.copy(), np.ones((9, 9), bool)) == 1).all()


def test_v2_shortfall_to_s5():
    grid = Grid(0.0, 3000.0, 30.0, 100, 100)
    from reservoir_cd.validation import full_sample
    sm = np.full((100, 100), 4, np.uint8)
    sm[::10, ::10] = 5                    # 100 S5 spaced 300 m apart
    sm[1, 1] = 3; sm[1, 2] = 3            # only one S3 survives spacing
    alloc = {s: 0 for s in strata.STRATA_V2}
    alloc.update(S3=3, S5=2)
    pts, rep = full_sample.draw_sample_v2([sm], alloc, grid, 90.0, seed=0)
    assert rep["drawn"]["S3"] == 1 and rep["moved_to_S5"] == 2 and rep["drawn"]["S5"] == 4
