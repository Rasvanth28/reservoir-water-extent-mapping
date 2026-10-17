"""PrithviReservoirDataset: (T1, T2) chip pairs for the Prithvi / FC-Siam-diff models.

Sample dict (all tensors):
  image            float32 [2, 6, 256, 256]  normalized (T, C, H, W); no-data pixels = 0 (the mean)
  mask             int64   [2, 256, 256]     per-date labels 0 Water 1 Forest 2 Bare 3 Other, 255 ignore
  change           int64   [256, 256]        0 no change, 1 water loss, 2 water gain, 255 if either date unlabeled
  valid            bool    [256, 256]        clear observation at both dates
  temporal_coords  float32 [2, 2]            (year, day-of-year, 0-based) of the per-chip median medoid date
  location_coords  float32 [2]               (lat, lon) of the chip centre (Prithvi TL order)
  site_idx, year_t1, year_t2, row0, col0     ints (for bookkeeping)

Validation-point exclusion is applied at load time: labels (and so ``change``) are set to 255
within ``exclusion_radius_m`` of every point found under ``ground_truth_dir`` (all year pairs).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from reservoir_cd.datasets import normalize as nz
from reservoir_cd.datasets.chips import open_store
from reservoir_cd.datasets.exclusion import exclusion_mask, load_points_xy

LABEL_SOURCES = {"consensus": "label_cons", "rule": "label_rule", "esri": "label_esri"}


def change_target(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """0 no change, 1 water loss, 2 water gain; 255 where either label is 255 (class codes: 0 = Water)."""
    ch = np.zeros(l1.shape, np.int64)
    ch[(l1 == 0) & (l2 != 0)] = 1
    ch[(l1 != 0) & (l2 == 0)] = 2
    ch[(l1 == 255) | (l2 == 255)] = 255
    return ch


class PrithviReservoirDataset(Dataset):
    def __init__(self, chips_dir: str | Path = "data/processed/chips", split: str | list[str] = "train",
                 split_column: str = "split", sites: list[str] | None = None, label_source: str = "consensus",
                 exclusion_radius_m: float | None = 150.0, ground_truth_dir: str | Path = ".",
                 key_glob: str = "data/ground_truth/**/*_key.csv", gpkg: str | None = "data/ground_truth/validation_points.gpkg",
                 chip_size: int = 256, augment: bool = False, normalize: bool = True,
                 years: list[tuple[int, int]] | None = None):
        self.dir, self.size, self.augment, self.normalize = Path(chips_dir), chip_size, augment, normalize
        self.lkey = LABEL_SOURCES[label_source]
        splits = pd.read_parquet(self.dir / "splits.parquet")
        want = [split] if isinstance(split, str) else list(split)
        splits = splits[splits[split_column].isin(want)][["site", "row0", "col0"]]
        names = sites or sorted(splits.site.unique())
        parts = []
        for n in names:
            idx = pd.read_parquet(self.dir / f"{n}_index.parquet")
            parts.append(idx.merge(splits, on=["site", "row0", "col0"]))
        self.index = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if years is not None and len(self.index):
            self.index = self.index[[(a, b) in set(years) for a, b in zip(self.index.year_t1, self.index.year_t2)]]
        self.index = self.index.reset_index(drop=True)
        self.sites = names
        self._stores: dict = {}
        self._meta, self._excl = {}, {}
        pts = load_points_xy(ground_truth_dir, key_glob, gpkg) if exclusion_radius_m else np.zeros((0, 2))
        self.n_exclusion_points = len(pts)
        for n in names:
            a = dict(open_store(self.dir / f"{n}.zarr").attrs)
            self._meta[n] = dict(a, years_idx={y: i for i, y in enumerate(a["years"])})
            self._excl[n] = exclusion_mask(pts, a["x0"], a["y0"], a["res"], (a["height"], a["width"]),
                                           exclusion_radius_m or 0)

    def __len__(self) -> int:
        return len(self.index)

    def _store(self, site: str):
        if site not in self._stores:  # opened lazily per worker process
            self._stores[site] = open_store(self.dir / f"{site}.zarr")
        return self._stores[site]

    def __getitem__(self, i: int) -> dict:
        r = self.index.iloc[i]
        site, s = r.site, self.size
        g, m = self._store(r.site), self._meta[r.site]
        y1, y2 = m["years_idx"][int(r.year_t1)], m["years_idx"][int(r.year_t2)]
        r0, c0 = int(r.row0), int(r.col0)
        sl = (slice(r0, r0 + s), slice(c0, c0 + s))
        refl = np.stack([g["refl"][y, :, sl[0], sl[1]] for y in (y1, y2)])  # (2,6,H,W) int16
        valid = nz.int16_to_valid(refl).all(axis=0)
        lab = np.stack([g[self.lkey][y, sl[0], sl[1]] for y in (y1, y2)]).astype(np.int64)
        lab[:, ~valid] = 255
        lab[:, self._excl[site][sl]] = 255
        change = change_target(lab[0], lab[1])
        img = nz.normalize_hls(refl) if self.normalize else refl.astype(np.float32)
        if self.augment:
            k, flip = np.random.randint(4), np.random.rand() < 0.5
            f = lambda a: np.ascontiguousarray(np.flip(np.rot90(a, k, axes=(-2, -1)), -1) if flip else np.rot90(a, k, axes=(-2, -1)))
            img, lab, change, valid = f(img), f(lab), f(change), f(valid)
        return {
            "image": torch.from_numpy(img.astype(np.float32)),
            "mask": torch.from_numpy(lab),
            "change": torch.from_numpy(change),
            "valid": torch.from_numpy(valid),
            "temporal_coords": torch.tensor([[r.obs_year_t1, r.obs_doy_t1], [r.obs_year_t2, r.obs_doy_t2]], dtype=torch.float32),
            "location_coords": torch.tensor([r.lat, r.lon], dtype=torch.float32),
            "site_idx": self.sites.index(site), "year_t1": int(r.year_t1), "year_t2": int(r.year_t2),
            "row0": r0, "col0": c0,
        }
