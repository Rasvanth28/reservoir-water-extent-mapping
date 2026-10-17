import numpy as np
import pandas as pd

from reservoir_cd.datasets import normalize as nz
from reservoir_cd.datasets import splits
from reservoir_cd.datasets.chips import window_starts
from reservoir_cd.datasets.exclusion import exclusion_mask
from reservoir_cd.datasets.pairs import consecutive_pairs


def test_pairs():
    p = consecutive_pairs(list(range(2015, 2027)))
    assert len(p) == 11 and p[0] == (2015, 2016) and p[-1] == (2025, 2026)
    assert consecutive_pairs([2015, 2017]) == []


def test_window_starts_cover_scene():
    s = window_starts(609, 256, 128)
    assert s[0] == 0 and s[-1] == 609 - 256 and max(b - a for a, b in zip(s, s[1:])) <= 128


def test_normalize_roundtrip_and_nodata():
    refl = np.full((6, 4, 4), np.nan, np.float32)
    refl[:, :2] = 0.1  # 0.1 reflectance -> 1000 HLS units
    i16 = nz.reflectance_to_int16(refl)
    assert i16[0, 0, 0] == 1000 and i16[0, 3, 3] == nz.INT16_NODATA
    n = nz.normalize_hls(i16)
    assert np.isclose(n[0, 0, 0], (1000 - 1087) / 2248, atol=1e-5)
    assert (n[:, 3, 3] == 0).all()


def test_exclusion_radius():
    pts = np.array([[15.0 + 30 * 10, 3000.0 - 15.0 - 30 * 10]])  # centre of pixel (10, 10) of a 30 m grid at (0, 3000)
    m = exclusion_mask(pts, 0.0, 3000.0, 30.0, (30, 30), 150.0)
    assert m[10, 10] and m[10, 15] and not m[10, 16] and m.sum() == 81  # disc of radius 5 px
    assert exclusion_mask(pts, 0.0, 3000.0, 30.0, (30, 30), 0).sum() == 0


def test_change_target():
    from reservoir_cd.datasets.torch_dataset import change_target
    l1 = np.array([0, 0, 1, 255, 3])
    l2 = np.array([0, 1, 0, 0, 3])
    assert change_target(l1, l2).tolist() == [0, 1, 2, 255, 0]


def test_splits_no_leakage():
    cfg = {"chip_size": 256, "splits": {"seed": 1, "val_frac": 0.15, "test_frac": 0.15, "gap_px": 0,
                                        "min_labeled_frac": 0.0, "min_water_px": 1, "min_change_px": 0}}
    rs, cs = window_starts(900, 256, 128), window_starts(1000, 256, 128)
    pos = pd.DataFrame([dict(site="S", row0=r, col0=c, in_train_grid=True, in_eval_grid=(r % 256 == 0 or r == 644) and (c % 256 == 0 or c == 744),
                             max_extent_px=100) for r in rs for c in cs])
    pairs = pd.DataFrame([dict(site="S", row0=r, col0=c, labeled_frac=0.5, loss_px=10, gain_px=10) for r in rs for c in cs])
    sp = splits.make_splits(pos, pairs, cfg)
    assert {"val", "test"} <= set(sp.split)
    splits.assert_no_leakage(sp, 256)
    assert (sp["loso_S"] == "test").all()
