"""Build static-forest + yearly rule/consensus labels for all sites and all years.

Usage: .venv/bin/python scripts/build_labels.py [--sites BHA MET PIL] [--config configs/build.yaml]
Reads the Zarr cubes (read-only; reflectance is already scaled, never rescaled), caches Esri/JRC
rasters under data/external/, writes data/interim/<Site>_labels.zarr, data/interim/labels_summary.json
and QA figures under visualizations/labels/<Site>/. Does not draw any validation sample.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from reservoir_cd import viz
from reservoir_cd.config import load_yaml
from reservoir_cd.grid import Grid, grid_bbox_lonlat
from reservoir_cd.preprocessing import composite, esri, forest, indices, jrc, labels, reservoir_extent
from reservoir_cd.validation import sampling, strata

BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]


def fmt_date(d) -> str:
    return "" if np.isnat(d) else str(d.astype("datetime64[D]"))


def class_fracs(lab: np.ndarray, region: np.ndarray) -> dict:
    tot = (region & (lab != 255)).sum()
    return {labels.CLASS_NAMES[c]: float(((lab == c) & region).sum() / max(tot, 1)) for c in range(4)}


# ------------------------------------------------------------------ phase 1: extent + phenology
def phase1(site: dict, cfg: dict, bcfg: dict) -> dict:
    t0 = time.time()
    ds = xr.open_zarr(site["zarr"])
    g = Grid.from_cube_coords(ds.x.values, ds.y.values)
    times = ds.time.values
    sc = reservoir_extent.load_scene_indices(ds, cfg)
    print(f"[{site['code']}] loaded scene indices: T={len(times)} ({time.time() - t0:.0f}s)", flush=True)
    ext = reservoir_extent.derive_reservoir_extent(sc, cfg, g.res)
    feats = forest.phenology_features(sc, times, cfg["forest"], (cfg["season"]["start_month"], cfg["season"]["end_month"]))
    print(f"[{site['code']}] extent + phenology done ({time.time() - t0:.0f}s)", flush=True)
    bbox = grid_bbox_lonlat(g)
    raws = [esri.fetch_esri_year(y, g, bbox, bcfg["esri_cache"], site["name"]) for y in bcfg["esri_years"]]
    trees_years, nonforest_years = forest.esri_year_counts(raws)
    j = jrc.fetch_jrc_extent(g, bbox, bcfg["jrc_cache"], site["name"])
    jc = jrc.compare_extent(j, ext["max_extent"], ext["study"] | ext["max_extent"], g.res)
    return dict(site=site, grid=g, ext=ext, feats=feats, trees_years=trees_years, nonforest_years=nonforest_years,
                jrc=jc, n_scenes=len(times), n_dates=len(np.unique(times.astype("datetime64[D]"))),
                seconds=time.time() - t0)


# ------------------------------------------------------------------ phase 2: yearly labels
def year_composite(year, ds, sc, cfg):
    times = ds.time.values
    idx = composite.season_indices(times, year, cfg["season"]["start_month"], cfg["season"]["end_month"])
    if len(idx) == 0:
        return idx, None
    refl = ds.landsat_data.isel(time=idx).values[:, :6]  # reflectance already scaled
    return idx, composite.medoid_composite(refl, sc["valid"][idx], times[idx])


def spaced_candidates(smap: np.ndarray, quota: dict, bcfg: dict, res: float, rng) -> dict:
    out = {}
    for si, name in enumerate(strata.STRATA, start=1):
        cap = int(quota[name] * bcfg["feasibility_factor"] * 4)
        out[name] = sampling.greedy_spaced_count(smap == si, res, bcfg["min_spacing_m"], rng, cap)
    return out


def phase2(p1: dict, fmaps: dict, cfg: dict, bcfg: dict, fig_dir: Path) -> dict:
    t0 = time.time()
    site, g, ext = p1["site"], p1["grid"], p1["ext"]
    name, code = site["name"], site["code"]
    ds = xr.open_zarr(site["zarr"])
    sc = reservoir_extent.load_scene_indices(ds, cfg)
    times = ds.time.values
    study, mext = ext["study"], ext["max_extent"]
    clamp = tuple(cfg["water"]["otsu_clamp"])
    zpath = Path(bcfg["interim_dir"]) / f"{name}_labels.zarr"
    import shutil
    if zpath.exists():
        shutil.rmtree(zpath)

    static = {"water_frequency": ext["freq"], "n_clear_all": ext["n_clear"],
              "permanent": ext["permanent"].astype("uint8"), "max_extent": mext.astype("uint8"),
              "study_area": study.astype("uint8"),
              **{k: p1["feats"][k] for k in forest.FEATURES + ["phase", "n_obs"]},
              "forest_prob_c1": fmaps["prob"], "forest_c1": fmaps["forest_c1"].astype("uint8"),
              "esri_trees_years": p1["trees_years"], "forest_esri": fmaps["forest_esri"].astype("uint8"),
              "forest_consensus": fmaps["forest_consensus"]}
    xr.Dataset({k: (("y", "x"), v) for k, v in static.items()}, coords={"y": g.yc, "x": g.xc},
               attrs={"crs": g.crs}).to_zarr(zpath, group="static", mode="w", consolidated=True)

    frule = fmaps["forest_rule"]
    fcons = fmaps["forest_consensus"]
    res, yearly, fig_bands = {}, {}, None
    for year in bcfg["years"]:
        idx, comp = year_composite(year, ds, sc, cfg)
        if comp is None:
            print(f"[{code}] {year}: no Jan-Apr scenes", flush=True)
            continue
        b = comp["bands"]
        mndwi, ndvi = indices.mndwi(b[1], b[4]), indices.ndvi(b[3], b[2])
        has = comp["n_clear"] > 0
        t = labels.otsu_mndwi(mndwi, study & has, clamp)
        rule = labels.rule_label_static(mndwi, ndvi, t, mext, frule, cfg)
        rule[~study | ~has] = labels.UNLABELED
        agree = labels.temporal_agreement_static(sc["mndwi"][idx], sc["ndvi"][idx], sc["valid"][idx], rule, study,
                                                 mext, frule, t, cfg)
        esri_lab = None
        if year in bcfg["esri_years"]:
            raw = esri.fetch_esri_year(year, g, grid_bbox_lonlat(g), bcfg["esri_cache"], name)
            esri_lab = esri.map_esri(raw, cfg["esri_map"])
        cons = labels.consensus_label_static(rule, mndwi, t, comp["n_clear"], study, agree, esri_lab, mext, fcons, cfg)
        res[year] = dict(rule=rule, cons=cons, ndvi=ndvi.astype(np.float32), n_clear=comp["n_clear"])
        nc = comp["n_clear"][study]
        ent = {"otsu_threshold": t, "n_scenes_jan_apr": int(len(idx)),
               "n_clear": {"median": float(np.median(nc)), "frac_zero": float((nc == 0).mean()),
                           "frac_lt2": float((nc < 2).mean())},
               "rule_fractions": class_fracs(rule, study),
               "rule_fractions_inside_max_extent": class_fracs(rule, study & mext),
               "consensus_kept_by_rule_class": {}, "consensus_fractions_of_study": {}}
        for c in range(4):
            m = study & (rule == c)
            ent["consensus_kept_by_rule_class"][labels.CLASS_NAMES[c]] = float((cons[m] != 255).mean()) if m.any() else None
            ent["consensus_fractions_of_study"][labels.CLASS_NAMES[c]] = float(((cons == c) & study).sum() / max(study.sum(), 1))
        both = study & (rule != 255)
        ent["consensus_kept_overall"] = float((cons[both] != 255).mean())
        if esri_lab is not None:
            m = both & (esri_lab != 255)
            ent["esri_agreement_overall"] = float((esri_lab[m] == rule[m]).mean())
        yearly[year] = ent
        ds_y = xr.Dataset(
            {"composite": (("band", "y", "x"), b.astype("float32")),
             "source_date": (("y", "x"), comp["source_date"]), "n_clear": (("y", "x"), comp["n_clear"]),
             "mndwi": (("y", "x"), mndwi.astype("float32")), "ndvi": (("y", "x"), ndvi.astype("float32")),
             "rule_label": (("y", "x"), rule), "agreement": (("y", "x"), agree),
             "esri_label": (("y", "x"), esri_lab if esri_lab is not None else np.full(g.shape, 255, np.uint8)),
             "consensus_label": (("y", "x"), cons)},
            coords={"band": BANDS, "y": g.yc, "x": g.xc},
            attrs={"otsu_threshold": float(t), "has_esri": esri_lab is not None, "crs": g.crs,
                   "scene_dates": [fmt_date(d) for d in times[idx]]})
        ds_y.to_zarr(zpath, group=f"year_{year}", mode="a", consolidated=True)
        viz.year_overview(fig_dir / f"{year}_overview.png", name, year, b, rule, cons, g, len(idx))
        if year == 2018 or (fig_bands is None):
            fig_bands = (year, b)
        print(f"[{code}] {year}: {len(idx)} scenes t={t:.3f} rule={ {k: round(v, 3) for k, v in ent['rule_fractions'].items()} } "
              f"kept={ent['consensus_kept_overall']:.2f}", flush=True)
        del comp, b

    viz.extent_figure(fig_dir / "reservoir_extent.png", name, ext, fig_bands[1], g, fig_bands[0])
    viz.static_forest_figure(fig_dir / "static_forest.png", name, fig_bands[1], fmaps["prob"], p1["trees_years"],
                             fmaps, g, mext)
    viz.phenology_figure(fig_dir / "phenology_features.png", name, p1["feats"])

    # ---- strata candidates for all consecutive pairs
    rng = np.random.default_rng(bcfg["seed"])
    dmin = cfg["strata"]["min_delta_ndvi"]
    pairs = {}
    ys = sorted(res)
    for a, b_ in zip(ys[:-1], ys[1:]):
        if b_ != a + 1:
            continue
        ra, rb = res[a], res[b_]
        elig = study & (ra["n_clear"] >= 1) & (rb["n_clear"] >= 1)
        with np.errstate(invalid="ignore"):
            change_ok = np.abs(ra["ndvi"] - rb["ndvi"]) >= dmin
        s = strata.assign_strata(ra["rule"], rb["rule"], elig, cfg["shoreline_width_px"], change_ok)
        pairs[f"{a}->{b_}"] = {"raw": {st: int((s == i).sum()) for i, st in enumerate(strata.STRATA, 1)},
                               "after_spacing": spaced_candidates(s, bcfg["quota"], bcfg, g.res, rng)}
    pooled_raw = {st: sum(p["raw"][st] for p in pairs.values()) for st in strata.STRATA}
    pooled_sp = {st: sum(p["after_spacing"][st] for p in pairs.values()) for st in strata.STRATA}
    flags = {st: pooled_sp[st] < bcfg["feasibility_factor"] * bcfg["quota"][st] for st in strata.STRATA}
    print(f"[{code}] labels + strata done ({time.time() - t0:.0f}s)", flush=True)
    return {"years": yearly, "pairs": pairs, "pooled_raw": pooled_raw, "pooled_after_spacing": pooled_sp,
            "shortfall_flags": flags, "seconds_phase2": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/build.yaml")
    ap.add_argument("--sites", nargs="*", help="site codes (default: all)")
    args = ap.parse_args()
    bcfg = load_yaml(args.config)
    cfg = load_yaml(bcfg["preprocessing_config"])
    sites = [s for s in load_yaml(bcfg["sites_config"])["sites"] if not args.sites or s["code"] in args.sites]
    fcfg, seed = cfg["forest"], bcfg["seed"]
    rng = np.random.default_rng(seed)
    Path(bcfg["interim_dir"]).mkdir(parents=True, exist_ok=True)

    p1s = {s["name"]: phase1(s, cfg, bcfg) for s in sites}

    # ---- static forest: pooled RF, evaluation, per-site prediction
    sets = {}
    for n, p in p1s.items():
        g = p["grid"]
        feat = np.stack([p["feats"][k] for k in forest.FEATURES])
        blocks = forest.block_ids(g.shape, g.res, g.x0, g.y0, fcfg["block_m"])
        sets[n] = forest.training_set(feat, p["trees_years"], p["nonforest_years"], p["ext"]["max_extent"],
                                      blocks, fcfg, rng)
        p["feat_stack"] = feat
    rf_metrics, model = forest.evaluate_rf(sets, fcfg, seed)
    print("RF metrics:", json.dumps({k: rf_metrics[k] for k in ("block_cv", "leave_one_site_out")}, indent=1), flush=True)

    summary = {"forest_model": rf_metrics, "sites": {}}
    for n, p in p1s.items():
        site, g, ext = p["site"], p["grid"], p["ext"]
        prob = forest.predict_prob(model, p["feat_stack"])
        fm = forest.forest_maps(prob, p["trees_years"], fcfg)
        fm["prob"] = prob
        px_km2 = g.res ** 2 / 1e6
        study, mext = ext["study"], ext["max_extent"]
        outx = study & ~mext & np.isfinite(prob)
        c1, es, cons = fm["forest_c1"], fm["forest_esri"], fm["forest_consensus"]
        tp = (c1 & es & outx).sum()
        fdir = Path(bcfg["figures_dir"]) / n
        fdir.mkdir(parents=True, exist_ok=True)
        p2 = phase2(p, fm, cfg, bcfg, fdir)
        if site["code"] == sites[0]["code"]:
            viz.importance_figure(Path(bcfg["figures_dir"]) / "rf_feature_importance.png", rf_metrics["feature_importance"])
        summary["sites"][n] = {
            "code": site["code"], "n_scenes": p["n_scenes"], "n_unique_dates": p["n_dates"],
            "seconds_phase1": p["seconds"], "seconds_phase2": p2["seconds_phase2"],
            "study_area_km2": float(study.sum() * px_km2), "max_extent_km2": float(mext.sum() * px_km2),
            "permanent_km2": float(ext["permanent"].sum() * px_km2), "jrc_comparison": p["jrc"],
            "scene_threshold_fallbacks": int(ext["scene_fallback"].sum()),
            "static_forest": {
                "frac_c1_outside_extent": float(c1[outx].mean()), "frac_esri_outside_extent": float(es[outx].mean()),
                "frac_consensus_forest_outside_extent": float((cons[outx] == 1).mean()),
                "frac_consensus_nonforest_outside_extent": float((cons[outx] == 0).mean()),
                "frac_unlabeled_outside_extent": float((cons[outx] == 255).mean()),
                "c1_vs_esri_agreement": float((c1[outx] == es[outx]).mean()),
                "c1_precision_vs_esri": float(tp / max((c1 & outx).sum(), 1)),
                "c1_recall_vs_esri": float(tp / max((es & outx).sum(), 1)),
                "frac_pixels_without_features": float((~np.isfinite(prob))[study].mean())},
            **p2}
    Path(bcfg["summary_json"]).write_text(json.dumps(summary, indent=2, default=float))
    print_report(summary)


def print_report(summary: dict) -> None:
    m = summary["forest_model"]
    print("\n=== Forest RF ===")
    print("block-CV:", {k: round(v, 3) for k, v in m["block_cv"].items()})
    for s, v in m["leave_one_site_out"].items():
        print(f"  LOSO {s}:", {k: round(x, 3) for k, x in v.items()})
    print("  importance:", {k: round(v, 3) for k, v in sorted(m["feature_importance"].items(), key=lambda kv: -kv[1])})
    for n, s in summary["sites"].items():
        print(f"\n=== {n} ({s['code']}) ===")
        print(f"max extent {s['max_extent_km2']:.1f} km2, permanent {s['permanent_km2']:.1f} km2, "
              f"JRC IoU {s['jrc_comparison']['iou']:.3f}, static forest {s['static_forest']}")
        print("year  Water Forest Bare Other | kept W F B O")
        for y, e in s["years"].items():
            f, k = e["rule_fractions"], e["consensus_kept_by_rule_class"]
            kk = " ".join("  -" if k[c] is None else f"{100 * k[c]:3.0f}" for c in ("Water", "Forest", "Bare", "Other"))
            print(f"{y}  {f['Water']:.3f} {f['Forest']:.3f} {f['Bare']:.3f} {f['Other']:.3f} | {kk}")
        print("pair        " + " ".join(f"{k:>7}" for k in strata.STRATA))
        for pr, v in s["pairs"].items():
            print(f"{pr}  " + " ".join(f"{v['raw'][k]:7d}" for k in strata.STRATA))
        print("pooled raw  " + " ".join(f"{s['pooled_raw'][k]:7d}" for k in strata.STRATA))
        print("pooled sp90 " + " ".join(f"{s['pooled_after_spacing'][k]:7d}" for k in strata.STRATA))
        print("shortfall:", [k for k, v in s["shortfall_flags"].items() if v])


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")
    main()
