"""End-to-end pilot for Bhavanisagar: composites, labels, strata, 50 validation points.

Usage: .venv/bin/python scripts/run_pilot.py [configs/pilot.yaml]
Reads the existing Zarr cube (read-only); never touches data_collection.py outputs.
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pyproj import Transformer

from reservoir_cd.config import load_yaml
from reservoir_cd.grid import Grid
from reservoir_cd.preprocessing import composite, esri, indices, labels, reservoir_extent
from reservoir_cd.validation import sampling, strata

CLASS_COLORS = {0: "#1f78b4", 1: "#1b5e20", 2: "#d2b48c", 3: "#ff7f00", 255: "#9e9e9e"}
STRATUM_COLORS = ["#1f78b4", "#1b5e20", "#d2b48c", "#ff7f00", "#e31a1c", "#6a3d9a", "#00bcd4", "#ffd700"]


def lab_cmap() -> ListedColormap:
    return ListedColormap([CLASS_COLORS[k] for k in (0, 1, 2, 3, 255)])


def lab_img(a: np.ndarray) -> np.ndarray:
    """Map class codes to 0..4 for plotting with lab_cmap."""
    return np.where(a == 255, 4, a)


def fmt_date(d: np.datetime64) -> str:
    return "" if np.isnat(d) else str(d.astype("datetime64[D]"))


def frac_by_class(lab: np.ndarray, region: np.ndarray) -> dict[str, float]:
    tot = (region & (lab != 255)).sum()
    return {labels.CLASS_NAMES[c]: float(((lab == c) & region).sum() / max(tot, 1)) for c in range(4)}


def annual_ndvi_pct(sc: dict, yr: np.ndarray, year: int, pct: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel NDVI percentile over all clear scenes of calendar ``year`` (NaN where none)."""
    m = yr == year
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p = np.nanpercentile(sc["ndvi"][m], pct, axis=0)
    return p.astype(np.float32), sc["valid"][m].sum(axis=0).astype(np.int16)


def jrc_comparison(grid: Grid, bbox: list[float], cache: Path, ext: np.ndarray, study: np.ndarray) -> dict | None:
    """Compare our max extent with JRC GSW 'extent' (ever water, to 2020). None if unavailable."""
    try:
        path = cache / "Bhavanisagar_jrc_extent_30m.tif"
        if path.exists():
            with rasterio.open(path) as src:
                j = src.read(1)
        else:
            import planetary_computer as pc
            import pystac_client
            from rasterio.enums import Resampling
            from rasterio.vrt import WarpedVRT
            cat = pystac_client.Client.open(esri.STAC_URL, modifier=pc.sign_inplace)
            items = list(cat.search(collections=["jrc-gsw"], bbox=bbox).items())
            j = np.zeros(grid.shape, np.uint8)
            for it in items:
                with rasterio.open(it.assets["extent"].href) as src:
                    with WarpedVRT(src, crs=grid.crs, transform=grid.transform, width=grid.width,
                                   height=grid.height, resampling=Resampling.nearest) as vrt:
                        j = np.maximum(j, vrt.read(1).astype(np.uint8))
            cache.mkdir(parents=True, exist_ok=True)
            with rasterio.open(path, "w", driver="GTiff", height=grid.height, width=grid.width, count=1,
                               dtype="uint8", crs=grid.crs, transform=grid.transform, compress="deflate") as dst:
                dst.write(j, 1)
        jw = (j == 1) & study
        inter = (jw & ext).sum()
        return {"jrc_extent_km2_in_study": float(jw.sum() * grid.res ** 2 / 1e6),
                "frac_of_our_extent_in_jrc": float(inter / max(ext.sum(), 1)),
                "frac_of_jrc_in_our_extent": float(inter / max(jw.sum(), 1)),
                "iou": float(inter / max((jw | ext).sum(), 1))}
    except Exception as e:  # network / schema problems must not break the pilot
        print("JRC comparison skipped:", repr(e))
        return None


def year_composite(year: int, ds: xr.Dataset, sc: dict, cfg: dict) -> dict:
    times = ds.time.values
    idx = composite.season_indices(times, year, cfg["season"]["start_month"], cfg["season"]["end_month"])
    cube = ds.landsat_data.isel(time=idx).values  # (T,7,H,W) float64, reflectance already scaled
    refl = cube[:, :6]
    comp = composite.medoid_composite(refl, sc["valid"][idx], times[idx])
    return dict(idx=idx, refl=refl, comp=comp)


def process_year(year: int, ds: xr.Dataset, grid: Grid, sc: dict, yr: np.ndarray, ext: dict, cfg: dict,
                 pcfg: dict) -> dict:
    study, mext = ext["study"], ext["max_extent"]
    times = ds.time.values
    yc = year_composite(year, ds, sc, cfg)
    idx, refl, comp = yc["idx"], yc["refl"], yc["comp"]
    ind = {"mndwi": indices.mndwi(comp["bands"][1], comp["bands"][4]),
           "ndvi": indices.ndvi(comp["bands"][3], comp["bands"][2])}
    bright = labels.brightness(comp["bands"])
    p90, n_year = annual_ndvi_pct(sc, yr, year, cfg["forest"]["ndvi_pct"])
    clamp = tuple(cfg["water"]["otsu_clamp"])
    has = comp["n_clear"] > 0
    t = labels.otsu_mndwi(ind["mndwi"], study & has, clamp)
    rule = labels.rule_label(ind["mndwi"], ind["ndvi"], bright, t, mext, p90, cfg)
    rule[~study | ~has] = labels.UNLABELED
    agree, scene_t, scene_fb = labels.temporal_agreement(
        sc["mndwi"][idx], sc["ndvi"][idx], labels.brightness(refl), sc["valid"][idx], rule, study,
        mext, p90, t, cfg)
    esri_lab = None
    if year in pcfg["esri_years"]:
        raw = esri.fetch_esri_year(year, grid, pcfg["bbox_lonlat"], pcfg["esri_cache"], pcfg["site"])
        esri_lab = esri.map_esri(raw, cfg["esri_map"])
    cons = labels.consensus_label(rule, ind["mndwi"], t, comp["n_clear"], study, agree, esri_lab, mext, p90, cfg)
    # diagnostic: what the secondary "Bare outside extent" rule would catch
    cfg_out = {**cfg, "bare": {**cfg["bare"], "outside_enabled": True}}
    rule_out = labels.rule_label(ind["mndwi"], ind["ndvi"], bright, t, mext, p90, cfg_out)
    outside_bare = (rule_out == labels.BARE) & (rule != labels.BARE) & study & has & (rule != labels.WATER)
    print(f"{year}: {len(idx)} scenes, Otsu t={t:.3f}, scene thr={np.round(scene_t, 2).tolist()} fb={sum(scene_fb)}")
    return dict(year=year, bands=comp["bands"], source_date=comp["source_date"], n_clear=comp["n_clear"],
                scene_dates=[fmt_date(d) for d in times[idx]], t=t, scene_t=scene_t, scene_fb=scene_fb,
                rule=rule, agree=agree, esri=esri_lab, cons=cons, p90=p90, n_year=n_year,
                outside_bare=outside_bare, **ind)


def water_only_map(year: int, ds, sc: dict, study: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(Water/Other/255 map, n_clear) from the Jan-Apr composite; used for full-pair S6/S7 counts."""
    yc = year_composite(year, ds, sc, cfg)
    comp = yc["comp"]
    m = indices.mndwi(comp["bands"][1], comp["bands"][4])
    n = indices.ndvi(comp["bands"][3], comp["bands"][2])
    has = comp["n_clear"] > 0
    t = labels.otsu_mndwi(m, study & has, tuple(cfg["water"]["otsu_clamp"]))
    lab = np.where(m > t, labels.WATER, labels.OTHER).astype(np.uint8)
    lab[~(np.isfinite(m) & np.isfinite(n)) | ~study | ~has] = labels.UNLABELED
    return lab, comp["n_clear"]


def save_interim(path: Path, grid: Grid, res: dict, ext: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    band_names = ["blue", "green", "red", "nir08", "swir16", "swir22"]
    static = xr.Dataset(
        {"water_frequency": (("y", "x"), ext["freq"]), "n_clear_all": (("y", "x"), ext["n_clear"]),
         "permanent": (("y", "x"), ext["permanent"].astype("uint8")),
         "max_extent": (("y", "x"), ext["max_extent"].astype("uint8")),
         "study_area": (("y", "x"), ext["study"].astype("uint8"))},
        coords={"y": grid.yc, "x": grid.xc}, attrs={"crs": grid.crs})
    static.to_zarr(path, group="static", mode="w", consolidated=False)
    first = False
    for y, r in res.items():
        ds = xr.Dataset(
            {
                "composite": (("band", "y", "x"), r["bands"].astype("float32")),
                "source_date": (("y", "x"), r["source_date"]),
                "n_clear": (("y", "x"), r["n_clear"]),
                "mndwi": (("y", "x"), r["mndwi"].astype("float32")),
                "ndvi": (("y", "x"), r["ndvi"].astype("float32")),
                "ndvi_p90": (("y", "x"), r["p90"]),
                "rule_label": (("y", "x"), r["rule"]),
                "agreement": (("y", "x"), r["agree"]),
                "esri_label": (("y", "x"), r["esri"] if r["esri"] is not None else np.full(grid.shape, 255, np.uint8)),
                "consensus_label": (("y", "x"), r["cons"]),
            },
            coords={"band": band_names, "y": grid.yc, "x": grid.xc},
            attrs={"otsu_threshold": r["t"], "has_esri": r["esri"] is not None,
                   "scene_dates": r["scene_dates"], "crs": grid.crs},
        )
        ds.to_zarr(path, group=f"year_{y}", mode="w" if first else "a", consolidated=False)
        first = False


def true_color(bands: np.ndarray) -> np.ndarray:
    rgb = np.stack([bands[2], bands[1], bands[0]], axis=-1)
    out = np.zeros_like(rgb)
    for i in range(3):
        v = rgb[..., i]
        ok = np.isfinite(v)
        lo, hi = np.percentile(v[ok], [2, 98])
        out[..., i] = np.clip((v - lo) / (hi - lo), 0, 1)
    return np.nan_to_num(out, nan=1.0)


def extent_figure(fig_dir: Path, ext: dict, bands: np.ndarray, grid: Grid) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    e = [grid.x0, grid.x0 + grid.res * grid.width, grid.y0 - grid.res * grid.height, grid.y0]
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].imshow(true_color(bands), extent=e)
    axs[0].set_title("True colour (2017 Jan-Apr medoid)")
    im = axs[1].imshow(true_color(bands), extent=e)
    fr = np.ma.masked_where(~(ext["freq"] > 0), ext["freq"])
    h = axs[1].imshow(fr, cmap="viridis", vmin=0, vmax=1, extent=e, alpha=0.85, interpolation="nearest")
    xc, yc = grid.xc, grid.yc
    axs[1].contour(xc, yc, ext["max_extent"].astype(float), levels=[0.5], colors="red", linewidths=0.8)
    axs[1].contour(xc, yc, ext["study"].astype(float), levels=[0.5], colors="white", linewidths=0.8, linestyles="--")
    axs[1].contour(xc, yc, ext["permanent"].astype(float), levels=[0.5], colors="cyan", linewidths=0.6)
    plt.colorbar(h, ax=axs[1], shrink=0.7, label="water frequency")
    axs[1].set_title("Water frequency; red = max extent, cyan = permanent, white dashed = study area")
    fig.savefig(fig_dir / "reservoir_extent.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_figures(fig_dir: Path, res: dict, study: np.ndarray, extm: np.ndarray, grid: Grid, pairs, smaps, points) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    legend = [Patch(color=CLASS_COLORS[k], label=n) for k, n in
              zip((0, 1, 2, 3, 255), ["Water", "Forest", "Bare", "Other", "Unlabeled"])]
    ext = [grid.x0, grid.x0 + grid.res * grid.width, grid.y0 - grid.res * grid.height, grid.y0]
    for y, r in res.items():
        plt.imsave(fig_dir / f"{y}_truecolor.png", true_color(r["bands"]))
        for key, name in (("rule", "rule_label"), ("cons", "consensus_label")):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(lab_img(r[key]), cmap=lab_cmap(), vmin=0, vmax=4, interpolation="nearest", extent=ext)
            ax.set_title(f"Bhavanisagar {y} {name}")
            ax.legend(handles=legend, loc="lower right", fontsize=7)
            fig.savefig(fig_dir / f"{y}_{name}.png", dpi=130, bbox_inches="tight")
            plt.close(fig)
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.5))
        m = r["mndwi"][study & np.isfinite(r["mndwi"])]
        axs[0].hist(m, bins=120, color="#1f78b4")
        axs[0].axvline(r["t"], color="r", label=f"Otsu t={r['t']:.3f}")
        axs[0].set_title(f"{y} MNDWI (study area)")
        axs[0].legend()
        v = r["ndvi"][study & np.isfinite(r["ndvi"])]
        axs[1].hist(v, bins=120, color="#1b5e20")
        axs[1].axvline(0.35, color="k", ls=":", label="Bare max (in extent)")
        axs[1].axvline(0.2, color="r", ls="--", label="Forest dry min")
        axs[1].set_title(f"{y} Jan-Apr composite NDVI (study area)")
        axs[1].legend()
        pv = r["p90"][study & ~extm & np.isfinite(r["p90"])]
        axs[2].hist(pv, bins=120, color="#2e7d32")
        axs[2].axvline(0.6, color="r", ls="--", label="Forest p90 min 0.6")
        axs[2].set_title(f"{y} annual NDVI p90 (study, outside extent)")
        axs[2].legend()
        fig.savefig(fig_dir / f"{y}_histograms.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    cm = ListedColormap(["#ffffff"] + STRATUM_COLORS)
    for p, (a, b) in enumerate(pairs):
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(smaps[p], cmap=cm, vmin=0, vmax=8, interpolation="nearest", extent=ext)
        for pt in points:
            if pt["pair"] == p:
                ax.plot(pt["x"], pt["y"], "k+", ms=8, mew=1.5)
        ax.set_title(f"Strata {a}->{b} (black + = sampled points)")
        ax.legend(handles=[Patch(color=c, label=s) for s, c in zip(strata.STRATA, STRATUM_COLORS)],
                  loc="lower right", fontsize=7, ncol=2)
        fig.savefig(fig_dir / f"strata_{a}_{b}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    pcfg = load_yaml(sys.argv[1] if len(sys.argv) > 1 else "configs/pilot.yaml")
    cfg = load_yaml(pcfg["preprocessing_config"])
    ds = xr.open_zarr(pcfg["zarr"])
    grid = Grid.from_cube_coords(ds.x.values, ds.y.values)
    yr = ds.time.values.astype("datetime64[Y]").astype(int) + 1970

    sc = reservoir_extent.load_scene_indices(ds, cfg)
    ext = reservoir_extent.derive_reservoir_extent(sc, cfg, grid.res)
    study, mext = ext["study"], ext["max_extent"]
    px_km2 = grid.res ** 2 / 1e6
    print(f"max extent {mext.sum() * px_km2:.1f} km2, permanent {ext['permanent'].sum() * px_km2:.1f} km2, "
          f"study area {study.sum()} px of {study.size}; scene fallbacks {int(ext['scene_fallback'].sum())}/{len(yr)}")
    jrc = jrc_comparison(grid, pcfg["bbox_lonlat"], Path(pcfg["esri_cache"]).parent / "jrc", mext, study | mext)

    res = {y: process_year(y, ds, grid, sc, yr, ext, cfg, pcfg) for y in pcfg["years"]}
    save_interim(Path(pcfg["interim_zarr"]), grid, res, ext)

    # --- strata ---
    pairs = [tuple(p) for p in pcfg["pairs"]]
    dmin = cfg["strata"]["min_delta_ndvi"]
    smaps = []
    for a, b in pairs:
        ra, rb = res[a], res[b]
        elig = study & (ra["n_clear"] >= 1) & (rb["n_clear"] >= 1)
        forest = (ra["rule"] == labels.FOREST) | (rb["rule"] == labels.FOREST)
        d_p90 = np.abs(ra["p90"] - rb["p90"])
        d_ndvi = np.abs(ra["ndvi"] - rb["ndvi"])
        change_ok = np.where(forest, d_p90 >= dmin, d_ndvi >= dmin)
        smaps.append(strata.assign_strata(ra["rule"], rb["rule"], elig, cfg["shoreline_width_px"], change_ok,
                                         static_forest=False))  # legacy pilot semantics
    pts, rep = sampling.draw_sample(smaps, pcfg["allocation"], grid, pcfg["min_spacing_m"], pcfg["seed"])

    # --- point attributes; random-order IDs ---
    rng = np.random.default_rng(pcfg["seed"] + 1)
    order = rng.permutation(len(pts))
    to_ll = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    code, bs = pcfg["site_code"], pcfg["block_size_m"]
    final = []
    for n, i in enumerate(order, start=1):
        p = dict(pts[i])
        a, b = pairs[p["pair"]]
        r, c = p["row"], p["col"]
        p.update(point_id=f"{code}-P-{n:04d}", site=pcfg["site"], year_t1=a, year_t2=b,
                 date_t1=fmt_date(res[a]["source_date"][r, c]), date_t2=fmt_date(res[b]["source_date"][r, c]),
                 block_id=f"{int(p['x'] // bs)}_{int(p['y'] // bs)}",
                 rule_label_t1=labels.CLASS_NAMES[int(res[a]["rule"][r, c])],
                 rule_label_t2=labels.CLASS_NAMES[int(res[b]["rule"][r, c])],
                 consensus_label_t1=labels.CLASS_NAMES[int(res[a]["cons"][r, c])],
                 consensus_label_t2=labels.CLASS_NAMES[int(res[b]["cons"][r, c])])
        p["lon"], p["lat"] = to_ll.transform(p["x"], p["y"])
        final.append(p)
    out = Path(pcfg["points_dir"])
    out.mkdir(parents=True, exist_ok=True)
    sampling.write_kml(sorted(final, key=lambda q: q["point_id"]), out / f"{code}_points.kml", grid)
    sampling.write_csvs(final, pcfg["site"], out, code)

    fig_dir = Path(pcfg["figures_dir"])
    make_figures(fig_dir, res, study, mext, grid, pairs, smaps, final)
    extent_figure(fig_dir, ext, res[2017]["bands"] if 2017 in res else next(iter(res.values()))["bands"], grid)

    # --- all consecutive Jan-Apr pairs: S6 / S7 candidate counts (water maps only) ---
    wmaps = {}
    for y in range(2015, 2027):
        wmaps[y] = water_only_map(y, ds, sc, study, cfg)
    full = {}
    for y in range(2015, 2026):
        (l1, n1), (l2, n2) = wmaps[y], wmaps[y + 1]
        s = strata.assign_strata(l1, l2, study & (n1 >= 1) & (n2 >= 1), cfg["shoreline_width_px"])
        full[f"{y}->{y + 1}"] = {"S6": int((s == 6).sum()), "S7": int((s == 7).sum())}

    # --- summary ---
    summary = {"study_area_px": int(study.sum()), "study_area_km2": float(study.sum() * px_km2),
               "reservoir": {"max_extent_px": int(mext.sum()), "max_extent_km2": float(mext.sum() * px_km2),
                             "permanent_px": int(ext["permanent"].sum()),
                             "permanent_km2": float(ext["permanent"].sum() * px_km2),
                             "n_scenes_all": int(len(yr)),
                             "scene_threshold_fallbacks_all": int(ext["scene_fallback"].sum()),
                             "scene_threshold_fallbacks_rough_pass": int(ext["rough_fallbacks"]),
                             "scene_eta_min_median_max": [float(np.min(ext["scene_eta"])),
                                                          float(np.median(ext["scene_eta"])),
                                                          float(np.max(ext["scene_eta"]))],
                             "jrc_comparison": jrc},
               "years": {}, "pairs": {}, "sampling": rep, "full_pair_change_counts": full}
    for y, r in res.items():
        nc = r["n_clear"][study]
        inx, outx = study & mext, study & ~mext
        ent = {
            "otsu_threshold": r["t"], "n_scenes": len(r["scene_dates"]), "scene_threshold": r["scene_t"],
            "scene_fallbacks": int(sum(r["scene_fb"])),
            "n_clear": {"min": int(nc.min()), "median": float(np.median(nc)), "mean": float(nc.mean()),
                        "max": int(nc.max()), "frac_zero": float((nc == 0).mean()),
                        "frac_lt2": float((nc < 2).mean())},
            "n_clear_scenes_in_year": int(r["n_year"].max()),
            "rule_fractions": frac_by_class(r["rule"], study),
            "rule_fractions_inside_max_extent": frac_by_class(r["rule"], inx),
            "rule_fractions_outside_max_extent": frac_by_class(r["rule"], outx),
            "bare_px_inside_extent": int(((r["rule"] == labels.BARE) & inx).sum()),
            "bare_px_outside_extent": int(((r["rule"] == labels.BARE) & outx).sum()),
            "outside_bare_rule_would_add_px": int(r["outside_bare"].sum()),
            "consensus_kept_by_rule_class": {}, "esri_agreement_by_rule_class": None,
        }
        if r["esri"] is not None:
            ent["outside_bare_rule_esri_bare_frac"] = float((r["esri"][r["outside_bare"]] == labels.BARE).mean()) \
                if r["outside_bare"].any() else None
        for c in range(4):
            m = study & (r["rule"] == c)
            ent["consensus_kept_by_rule_class"][labels.CLASS_NAMES[c]] = float((r["cons"][m] != 255).mean()) if m.any() else None
        ent["consensus_kept_overall"] = float((r["cons"][study & (r["rule"] != 255)] != 255).mean())
        ent["consensus_class_counts"] = {labels.CLASS_NAMES[c]: int(((r["cons"] == c) & study).sum()) for c in range(4)}
        if r["esri"] is not None:
            ag = {}
            for c in range(4):
                m = study & (r["rule"] == c) & (r["esri"] != 255)
                ag[labels.CLASS_NAMES[c]] = float((r["esri"][m] == c).mean()) if m.any() else None
            both = study & (r["rule"] != 255) & (r["esri"] != 255)
            ag["overall"] = float((r["esri"][both] == r["rule"][both]).mean())
            ent["esri_agreement_by_rule_class"] = ag
            ent["esri_unlabeled_frac"] = float((r["esri"][study] == 255).mean())
            # Esri Trees recall / precision outside the extent (Forest rule)
            o = outx & (r["esri"] != 255)
            f, t_ = (r["rule"] == 1) & o, (r["esri"] == 1) & o
            ent["forest_vs_esri_trees_outside_extent"] = {
                "precision": float((f & t_).sum() / max(f.sum(), 1)), "recall": float((f & t_).sum() / max(t_.sum(), 1))}
        summary["years"][y] = ent
    summary["forest_fraction_by_year"] = {y: r["rule_fractions"]["Forest"] for y, r in
                                          ((y, summary["years"][y]) for y in res)}
    for p, (a, b) in enumerate(pairs):
        summary["pairs"][f"{a}->{b}"] = {s: int((smaps[p] == i).sum()) for i, s in enumerate(strata.STRATA, 1)}
    summary["sampled_by_pair"] = {f"{a}->{b}": sum(1 for q in final if q["pair"] == p) for p, (a, b) in enumerate(pairs)}
    Path(pcfg["summary_json"]).write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
