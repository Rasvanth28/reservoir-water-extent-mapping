"""Freeze the labeled validation sample (validation_protocol.md §6 step 5).

Runs the labeling-sheet QC on each site's `<SITE>_labels_filled.csv`, merges it with `<SITE>_key.csv`
(strata, x/y, block, automatic labels) and writes an immutable, hashed snapshot:

  data/ground_truth/frozen/<version>/validation_points.gpkg   EPSG:32643 points, layer "points"
  data/ground_truth/frozen/<version>/validation_points.csv
  data/ground_truth/frozen/<version>/strata_weights.csv       copy, for the Olofsson estimator
  data/ground_truth/frozen/<version>/manifest.json            sha256 of every input and output
  validation_freeze/<version>.json                           tracked copy of the manifest (data/ is gitignored)

Refuses to overwrite an existing version. Freeze before any model is scored; corrections after the
second-interpreter adjudication go into a new version (e.g. v1.1), never into an existing one.

Usage: .venv/bin/python scripts/freeze_validation.py --version v1
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_qc import clean, qc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data/ground_truth/full"
CRS = "EPSG:32643"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_site(site: str) -> tuple[pd.DataFrame, list[Path]]:
    d = FULL / site
    sheet, key, xlsx = d / f"{site}_labels_filled.csv", d / f"{site}_key.csv", d / f"{site}_labels.xlsx"
    if xlsx.exists() and xlsx.stat().st_mtime > sheet.stat().st_mtime + 1:
        sys.exit(f"{xlsx.name} is newer than {sheet.name}; run scripts/label_workbook.py export first")

    df = clean(pd.read_csv(sheet, dtype=str, keep_default_na=False))
    issues = qc(df)
    blank = df[(df.label_t1 == "") | (df.label_t2 == "")]
    if issues or len(blank):
        for i in issues:
            print("  ", *i)
        sys.exit(f"{site}: {len(issues)} QC issue(s), {len(blank)} unlabeled point(s); fix before freezing")

    k = pd.read_csv(key, dtype=str, keep_default_na=False)
    m = df.merge(k, on="point_id", how="inner", suffixes=("", "_key"))
    if len(m) != len(df) or len(m) != len(k):
        sys.exit(f"{site}: point_id mismatch between {sheet.name} ({len(df)}) and {key.name} ({len(k)})")
    for c in ("site", "date_t1", "date_t2"):
        if (m[c] != m[f"{c}_key"]).any():
            sys.exit(f"{site}: column {c} differs between sheet and key")
    m = m.drop(columns=[c for c in m.columns if c.endswith("_key")])
    return m, [sheet, key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="e.g. v1, v1.1")
    ap.add_argument("--sites", nargs="+", default=["BHA", "MET", "PIL"])
    args = ap.parse_args()

    out = ROOT / "data/ground_truth/frozen" / args.version
    tracked = ROOT / "validation_freeze" / f"{args.version}.json"
    if out.exists() or tracked.exists():
        sys.exit(f"version {args.version} already frozen ({out}); pick a new version")

    parts, inputs = [], []
    for s in args.sites:
        m, files = load_site(s)
        parts.append(m)
        inputs += files
        print(f"{s}: {len(m)} points, QC clean")
    df = pd.concat(parts, ignore_index=True)

    for c in ("x", "y", "lon", "lat", "minutes"):
        df[c] = pd.to_numeric(df[c])
    for c in ("year_t1", "year_t2", "confidence_t1", "confidence_t2"):
        df[c] = df[c].astype(int)
    for c in ("mixed_t1", "mixed_t2", "second_interpreter"):
        df[c] = df[c].str.upper() == "TRUE"

    weights = FULL / "strata_weights.csv"
    inputs.append(weights)

    out.mkdir(parents=True)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=CRS)
    gdf.to_file(out / "validation_points.gpkg", layer="points", driver="GPKG")
    df.to_csv(out / "validation_points.csv", index=False)
    shutil.copy2(weights, out / "strata_weights.csv")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "n_points": len(df),
        "n_per_site": df.site.value_counts().to_dict(),
        "n_per_stratum": df.groupby(["site", "stratum"]).size().unstack(fill_value=0).to_dict("index"),
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        "outputs": {f.name: sha256(f) for f in sorted(out.iterdir())},
    }
    text = json.dumps(manifest, indent=2) + "\n"
    (out / "manifest.json").write_text(text)
    tracked.parent.mkdir(exist_ok=True)
    tracked.write_text(text)
    print(f"Frozen {len(df)} points → {out}\nManifest: {tracked.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
