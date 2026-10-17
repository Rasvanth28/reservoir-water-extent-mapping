"""Scene store + chip index (EDD 3.3).

Design: rather than materialising every 256x256 chip (stride 128 duplicates pixels ~4x and each
chip appears in up to 2 pairs), ``build_scene_store`` writes ONE compressed Zarr per site with the
yearly full-scene rasters; the Parquet index lists chip windows (row0, col0) per year pair and the
dataset slices windows at load time. The same store serves full-scene inference.

Store layout (data/processed/chips/<Site>.zarr), Y = number of years, N = 6 bands:
  refl         int16  (Y, 6, H, W)   reflectance * 10000 (HLS units), -32768 = no data
  label_cons   uint8  (Y, H, W)      consensus labels (0 Water 1 Forest 2 Bare 3 Other 255 unlabeled)
  label_rule   uint8  (Y, H, W)      rule labels (all pixels in study area; E2/E3 ablations)
  label_esri   uint8  (Y, H, W)      Esri labels mapped to our classes (255 outside Esri years)
  n_clear      uint8  (Y, H, W)
  date_days    int32  (Y, H, W)      medoid source date, days since 1970-01-01 (-1 = none)
  study_area, max_extent  uint8 (H, W)
  attrs: years, x0, y0 (top-left corner), res, crs, band names
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from pyproj import Transformer

from reservoir_cd.datasets import normalize as nz
from reservoir_cd.datasets.pairs import consecutive_pairs

CODEC = zarr.codecs.ZstdCodec(level=5)


def window_starts(n: int, size: int, stride: int) -> list[int]:
    """Start offsets covering [0, n) with the last window shifted back to end exactly at n."""
    if n < size:
        raise ValueError(f"scene dimension {n} < chip size {size}")
    s = list(range(0, n - size + 1, stride))
    if s[-1] != n - size:
        s.append(n - size)
    return s


def build_scene_store(labels_zarr: str | Path, out_path: str | Path, years: list[int], chip: int = 256) -> dict:
    """Convert data/interim/<Site>_labels.zarr into the compressed int16 scene store."""
    static = xr.open_zarr(labels_zarr, group="static")
    present = []
    for y in years:
        try:
            xr.open_zarr(labels_zarr, group=f"year_{y}")
            present.append(y)
        except Exception:
            pass
    H, W = static.sizes["y"], static.sizes["x"]
    xs, ys = static.x.values, static.y.values
    res = float(xs[1] - xs[0])
    g = zarr.open_group(str(out_path), mode="w")
    kw = dict(compressors=CODEC)
    refl = g.create_array("refl", shape=(len(present), 6, H, W), chunks=(1, 6, chip, chip), dtype="i2",
                          fill_value=nz.INT16_NODATA, **kw)
    arrs = {k: g.create_array(k, shape=(len(present), H, W), chunks=(1, chip, chip), dtype=dt, **kw)
            for k, dt in [("label_cons", "u1"), ("label_rule", "u1"), ("label_esri", "u1"),
                          ("n_clear", "u1"), ("date_days", "i4")]}
    for i, y in enumerate(present):
        d = xr.open_zarr(labels_zarr, group=f"year_{y}")
        assert list(d.band.values) == ["blue", "green", "red", "nir08", "swir16", "swir22"]
        refl[i] = nz.reflectance_to_int16(d.composite.values)
        arrs["label_cons"][i] = d.consensus_label.values
        arrs["label_rule"][i] = d.rule_label.values
        arrs["label_esri"][i] = d.esri_label.values
        arrs["n_clear"][i] = np.clip(d.n_clear.values, 0, 255).astype("u1")
        days = d.source_date.values.astype("datetime64[D]").astype("int64")
        days[np.isnat(d.source_date.values)] = -1
        arrs["date_days"][i] = days.astype("i4")
    for k in ("study_area", "max_extent"):
        a = g.create_array(k, shape=(H, W), chunks=(chip, chip), dtype="u1", **kw)
        a[:] = static[k].values.astype("u1")
    g.attrs.update(years=present, x0=float(xs[0] - res / 2), y0=float(ys[0] + res / 2), res=res,
                   crs="EPSG:32643", bands=nz.BAND_NAMES, height=H, width=W)
    return dict(dict(g.attrs))


def open_store(path: str | Path):
    return zarr.open_group(str(path), mode="r")


def _doy_year(days: float) -> tuple[int, int]:
    d = np.datetime64(int(days), "D")
    y = d.astype("datetime64[Y]")
    return int(y.astype(int) + 1970), int((d - y.astype("datetime64[D]")).astype(int))  # doy 0-based


def build_index(store_path: str | Path, site: str, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (positions, pair_index).

    positions: one row per window position on the union of the train / eval grids.
    pair_index: one row per (position, year pair) that passes the valid / labeled thresholds.
    """
    g = open_store(store_path)
    a = dict(g.attrs)
    H, W, res, x0, y0 = a["height"], a["width"], a["res"], a["x0"], a["y0"]
    years = a["years"]
    size = cfg["chip_size"]
    pairs = consecutive_pairs(years, cfg["pair_step_years"])
    rows_t = window_starts(H, size, cfg["stride_train"])
    cols_t = window_starts(W, size, cfg["stride_train"])
    rows_e = window_starts(H, size, cfg["stride_eval"])
    cols_e = window_starts(W, size, cfg["stride_eval"])
    pos = {(r, c): [True, False] for r in rows_t for c in cols_t}
    for r in rows_e:
        for c in cols_e:
            pos.setdefault((r, c), [False, False])[1] = True
    tr = Transformer.from_crs(a["crs"], "EPSG:4326", always_xy=True)
    refl, lab, ncl, dd = g["refl"][:], g["label_cons"][:], g["n_clear"][:], g["date_days"][:]
    ext, study = g["max_extent"][:], g["study_area"][:]
    valid = (refl != nz.INT16_NODATA).all(axis=1) & (ncl > 0)  # (Y,H,W)
    yi = {y: i for i, y in enumerate(years)}
    prow, qrow = [], []
    for (r, c), (in_t, in_e) in sorted(pos.items()):
        cx, cy = x0 + res * (c + size / 2), y0 - res * (r + size / 2)
        lon, lat = tr.transform(cx, cy)
        sl = (slice(r, r + size), slice(c, c + size))
        prow.append(dict(site=site, row0=r, col0=c, in_train_grid=in_t, in_eval_grid=in_e, centre_x=cx,
                         centre_y=cy, lat=lat, lon=lon, max_extent_px=int(ext[sl].sum()),
                         study_px=int(study[sl].sum())))
        for t1, t2 in pairs:
            i1, i2 = yi[t1], yi[t2]
            v = valid[i1][sl] & valid[i2][sl]
            l1, l2 = lab[i1][sl], lab[i2][sl]
            lab_both = (l1 != 255) & (l2 != 255) & v
            vf, lf = float(v.mean()), float(lab_both.mean())
            if vf < cfg["min_valid_frac"] or lf < cfg["min_labeled_frac"]:
                continue
            md = []
            for ii in (i1, i2):
                dv = dd[ii][sl][valid[ii][sl] & (dd[ii][sl] > 0)]
                md.append(float(np.median(dv)) if dv.size else -1.0)
            (y_a, d_a), (y_b, d_b) = _doy_year(md[0]), _doy_year(md[1])
            qrow.append(dict(site=site, year_t1=t1, year_t2=t2, row0=r, col0=c, valid_frac=vf,
                             labeled_frac=lf, water_t1=int((l1[lab_both] == 0).sum()),
                             water_t2=int((l2[lab_both] == 0).sum()),
                             loss_px=int(((l1 == 0) & (l2 != 0) & lab_both).sum()),
                             gain_px=int(((l1 != 0) & (l2 == 0) & lab_both).sum()),
                             med_date_t1=md[0], med_date_t2=md[1], obs_year_t1=y_a, obs_doy_t1=d_a,
                             obs_year_t2=y_b, obs_doy_t2=d_b, lat=lat, lon=lon))
    return pd.DataFrame(prow), pd.DataFrame(qrow)
