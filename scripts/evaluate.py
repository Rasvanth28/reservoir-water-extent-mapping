"""Score every method of `configs/evaluation.yaml` against a frozen validation version.

Usage: .venv/bin/python scripts/evaluate.py [--config configs/evaluation.yaml] [--n-boot 1000]
Writes to outputs/metrics/<version>/:
  metrics.csv     method, scope (site or All), subset (all / pure = non-mixed squares), metric, estimate,
                  se (stratified ratio estimator, Stehman 2014), ci_lo/ci_hi (95% block bootstrap), n_units
  mcnemar.csv     exact McNemar per method pair (water per point-date, water on true-change points,
                  change class per point), Holm-adjusted within each test family
  confusion.csv   sample-count confusion matrices (water, change, 4 classes)
  run.json        config, frozen manifest sha256, git commit
Strata = site x S1-S7 with populations from strata_weights.csv (pair "ALL", after pilot exclusion).
"""
import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_cd.config import load_yaml
from reservoir_cd.evaluation import metrics as M
from reservoir_cd.evaluation import sampling as S
from reservoir_cd.evaluation.stats import BlockBootstrap, holm, mcnemar

ROOT = Path(__file__).resolve().parent.parent


def load_points(frozen: Path):
    pts = pd.read_csv(frozen / "validation_points.csv", dtype=str, keep_default_na=False)
    for c in ("x", "y"):
        pts[c] = pts[c].astype(float)
    for c in ("year_t1", "year_t2"):
        pts[c] = pts[c].astype(int)
    for c in ("mixed_t1", "mixed_t2"):
        pts[c] = pts[c].str.upper() == "TRUE"
    w = pd.read_csv(frozen / "strata_weights.csv")
    w = w[w.pair == "ALL"]
    pop = {(r.site, r.stratum): float(r.n_pixels_after_pilot_exclusion) for r in w.itertuples()}
    return pts, pop


def predictions(pts, cfg):
    out = {}
    for name, m in cfg["methods"].items():
        if m["type"] == "key":
            out[name] = (S.from_key(pts, m["column"]), True)
        elif m["type"] == "water_zarr":
            out[name] = (S.from_water_zarr(pts, ROOT / m["path"], ROOT / cfg["chips_dir"]), False)
        else:
            raise ValueError(f"unknown method type {m['type']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/evaluation.yaml")
    ap.add_argument("--n-boot", type=int)
    args = ap.parse_args()
    cfg = load_yaml(ROOT / args.config)
    n_boot = args.n_boot if args.n_boot is not None else cfg["n_boot"]
    version = cfg["validation_version"]
    frozen = ROOT / cfg["frozen_dir"] / version
    out = ROOT / cfg["out_dir"] / version
    out.mkdir(parents=True, exist_ok=True)

    pts, pop = load_points(frozen)
    preds = predictions(pts, cfg)
    ref = pts[["label_t1", "label_t2", "mixed_t1", "mixed_t2"]]

    rows, conf = [], []
    scopes = [("All", pts.index)] + [(s, g.index) for s, g in pts.groupby("site")]
    for scope, ix in scopes:
        sub = pts.loc[ix]
        keys = sorted({(s, h) for s, h in zip(sub.site, sub.stratum)})
        code = {k: i for i, k in enumerate(keys)}
        h = np.array([code[k] for k in zip(sub.site, sub.stratum)])
        N = np.array([pop[k] for k in keys])
        boot = BlockBootstrap(h, (sub.site + "|" + sub.block_id).to_numpy(), cfg["seed"])
        reps = [boot.sample() for _ in range(n_boot)]
        for name, (pred, land) in preds.items():
            for subset in ("all", "pure"):
                terms = M.unit_terms(ref.loc[ix], pred.loc[ix], subset == "pure", land)
                est = M.estimate(terms, h, N)
                bs = [M.estimate(terms, h, N, r) for r in reps]
                for metric, (e, se) in est.items():
                    b = np.array([d[metric][0] for d in bs], float)
                    b = b[np.isfinite(b)]
                    lo, hi = np.percentile(b, [2.5, 97.5]) if len(b) else (np.nan, np.nan)
                    base = metric.rsplit("_", 1)[0]
                    x = terms.get(metric, terms.get(f"{base}_PA", terms.get(f"{base}_UA")))[1]
                    rows.append(dict(method=name, scope=scope, subset=subset, metric=metric, estimate=e,
                                     se=se, ci_lo=lo, ci_hi=hi, n_units=int((x > 0).sum())))

            r2 =pd.concat([ref.loc[ix, f"label_{t}"] for t in ("t1", "t2")], ignore_index=True)
            p2 = pd.concat([pred.loc[ix, t] for t in ("t1", "t2")], ignore_index=True)
            v = r2.isin(S.CLASSES) & p2.ne("")
            wr = r2.where(r2.eq("Water"), "Non-water")
            wp = p2.where(p2.eq("Water"), "Non-water")
            for kind, (a, b, labs, vv) in {
                "water": (wr, wp, ["Water", "Non-water"], v),
                "class": (r2, p2, S.CLASSES, v if land else v & False),
            }.items():
                if vv.any():
                    cm = M.confusion(a, b, vv, labs).stack().rename("n").reset_index()
                    conf.append(cm.assign(method=name, scope=scope, kind=kind))
            vc = (ref.loc[ix, "label_t1"].isin(S.CLASSES) & ref.loc[ix, "label_t2"].isin(S.CLASSES)
                  & pred.loc[ix, "t1"].ne("") & pred.loc[ix, "t2"].ne(""))
            rc = M.water_change(ref.loc[ix, "label_t1"], ref.loc[ix, "label_t2"])
            mc = M.water_change(pred.loc[ix, "t1"], pred.loc[ix, "t2"])
            cm = M.confusion(rc, mc, vc, M.CHANGE).stack().rename("n").reset_index()
            conf.append(cm.assign(method=name, scope=scope, kind="change"))

    met = pd.DataFrame(rows)
    met.to_csv(out / "metrics.csv", index=False)
    pd.concat(conf, ignore_index=True)[["method", "scope", "kind", "map", "reference", "n"]].to_csv(
        out / "confusion.csv", index=False)

    tests = []
    rc = M.water_change(ref.label_t1, ref.label_t2)
    true_change = np.concatenate([rc.ne("stable").to_numpy()] * 2)
    r2 = np.concatenate([ref.label_t1.to_numpy(), ref.label_t2.to_numpy()])
    for a, b in itertools.combinations(preds, 2):
        pa, pb = preds[a][0], preds[b][0]
        p2a = np.concatenate([pa.t1.to_numpy(), pa.t2.to_numpy()])
        p2b = np.concatenate([pb.t1.to_numpy(), pb.t2.to_numpy()])
        v = np.isin(r2, S.CLASSES) & (p2a != "") & (p2b != "")
        ca = (p2a == "Water") == (r2 == "Water")
        cb = (p2b == "Water") == (r2 == "Water")
        tests.append(dict(family="water, point-dates", a=a, b=b, **mcnemar(ca[v], cb[v])))
        vt = v & true_change
        tests.append(dict(family="water, true-change points", a=a, b=b, **mcnemar(ca[vt], cb[vt])))
        vc = (ref.label_t1.isin(S.CLASSES) & ref.label_t2.isin(S.CLASSES)
              & pa.t1.ne("") & pa.t2.ne("") & pb.t1.ne("") & pb.t2.ne("")).to_numpy()
        ma, mb = M.water_change(pa.t1, pa.t2).to_numpy(), M.water_change(pb.t1, pb.t2).to_numpy()
        tests.append(dict(family="change class, points", a=a, b=b,
                          **mcnemar((ma == rc.to_numpy())[vc], (mb == rc.to_numpy())[vc])))
    tests = pd.DataFrame(tests)
    tests["p_holm"] = tests.groupby("family").p.transform(lambda p: holm(p.to_numpy()))
    tests.to_csv(out / "mcnemar.csv", index=False)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest = frozen / "manifest.json"
    (out / "run.json").write_text(json.dumps({
        "config": cfg, "n_boot": n_boot, "git_commit": commit,
        "frozen_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}, indent=2) + "\n")

    show = met[(met.scope == "All")
               & met.metric.isin(["coverage", "water_OA", "water_UA", "water_PA", "water_F1", "change_OA",
                                  "loss_F1", "gain_F1", "class_OA"])]
    show = show.assign(v=show.apply(lambda r: f"{r.estimate:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]", axis=1))
    print(show.pivot_table(index=["subset", "metric"], columns="method", values="v", aggfunc="first").to_string())
    print()
    print(tests[["family", "a", "b", "n", "acc_a", "acc_b", "a_only", "b_only", "p", "p_holm"]].round(4).to_string())
    print(f"\n→ {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
