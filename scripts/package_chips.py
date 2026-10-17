"""Zip data/processed/chips (scene stores + indices + splits) for upload to Google Drive.

Usage: .venv/bin/python scripts/package_chips.py [--src data/processed/chips] [--out data/processed/chips.zip]
The Zarr chunks are already zstd-compressed, so the archive is stored without extra compression.
Archive layout (extract into <root>/data/ on Colab): processed/chips/... and, with --with-keys,
ground_truth/**/*_key.csv (validation-point exclusion happens at load time, so the keys must be there
at training time). --code also writes reservoir_cd_code.zip (src/, configs/, scripts/, pyproject.toml).
"""
import argparse
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/chips")
    ap.add_argument("--out", default="data/processed/chips.zip")
    ap.add_argument("--with-keys", action="store_true")
    ap.add_argument("--code", action="store_true", help="also zip the package code next to --out")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    files = [f for f in sorted(src.rglob("*")) if f.is_file()]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(f, Path("processed/chips") / f.relative_to(src))
        if a.with_keys:
            for k in Path("data/ground_truth").rglob("*_key.csv"):
                z.write(k, Path("ground_truth") / k.relative_to("data/ground_truth"))
    if a.code:
        cz = out.with_name("reservoir_cd_code.zip")
        with zipfile.ZipFile(cz, "w", zipfile.ZIP_DEFLATED) as z:
            z.write("pyproject.toml")
            for d in ("src", "configs", "scripts"):
                for f in sorted(Path(d).rglob("*")):
                    if f.is_file() and "__pycache__" not in f.parts and ".egg-info" not in str(f):
                        z.write(f)
        print(f"code -> {cz} ({cz.stat().st_size / 1e6:.2f} MB)")
    print(f"{len(files)} files -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
