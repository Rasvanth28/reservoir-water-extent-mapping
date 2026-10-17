# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research pipeline (target: ISPRS/TGRS submission) for dynamic reservoir water-extent change detection using the Prithvi-EO-2.0 geospatial foundation model. It builds dense multi-year (2015–2026) Landsat 8/9 time series for three Tamil Nadu reservoirs — Pillur, Bhavanisagar, Mettur — then will fine-tune Prithvi with LoRA to detect structural/climate-driven water extent transitions, validated against 1,500 manually digitized points.

The intended end-to-end pipeline is documented in `base_architecture.md` (the original 7-step plan) and `progress.md` (what's actually been done vs. adapted vs. pending — read this first to know current state). `README.md` has the project abstract. `Questions.md` and `Technical_Defense_QA.md` capture design-decision rationale (why Zarr, why Prithvi, why MNDWI, etc.) — check these before re-litigating a design choice that was already deliberated.

**Current phase:** Step 1 (geospatial data engineering) is ~80% complete, transitioning into Step 3 (tensor patch generation). Steps 2 (ground truth/labeling), 4-7 (baselines, LoRA fine-tuning, statistical evaluation, publication packaging) are not started. Always check `progress.md` for the latest status rather than assuming the architecture doc reflects reality — the plan has already diverged from it in documented ways (dense time series instead of anniversary dates, bi-annual chunking not in the original plan, morphological masking deferred to the dataset loader).

## Environment & Commands

- Python virtualenv at `.venv/` (Python 3.13, homebrew). Activate with `source .venv/bin/activate`, or invoke directly via `.venv/bin/python`.
- No `requirements.txt`/`pyproject.toml` exists — dependencies are only reflected in the venv's installed packages (key ones: `dask`, `stackstac`, `planetary-computer`, `pystac-client`, `xarray`, `zarr`, `rasterio`, `numpy`, `matplotlib`, `tqdm`). If you add a new dependency, install it into `.venv` and consider whether a requirements file is now warranted.
- There is no test suite, linter, or build step configured. Scripts are run directly, e.g.:
  - `.venv/bin/python data_collection.py`
  - `.venv/bin/python visualize_zarr.py`
  - `.venv/bin/python salvage_zarr.py`

## Architecture

### Data acquisition (`data_collection.py`)
Queries Microsoft Planetary Computer's STAC API (`landsat-c2-l2` collection) for Landsat 8/9 Level-2 surface reflectance imagery per reservoir bounding box, stacks bands `[blue, green, red, nir08, swir16, swir22, qa_pixel]` via `stackstac` at 30m/EPSG:32643, and writes to per-reservoir Zarr stores under `data/`.

Non-obvious constraints baked into this script:
- **Platform filter is date-gated**: only `landsat-8` before 2022, both `landsat-8`/`landsat-9` from 2022 onward (STAC `query` filter in `query_and_stack_reservoir`).
- **Bi-annual temporal chunking is required, not optional.** Planetary Computer SAS tokens expire in 45–60 minutes; a full-year pull for a large area (e.g. Mettur) exceeds that and throws `RasterioIOError`. The fix is looping in 6-month (H1/H2) windows and appending each chunk to the Zarr store via `to_zarr(..., mode="a", append_dim="time")`. Do not "simplify" this back into a single multi-year query.
- Before appending, non-dimensional coords and all attrs are stripped/cleared (`coords_to_drop`, `.attrs.clear()`) — Zarr append fails on coordinate/attr mismatches between chunks otherwise.
- `RESERVOIR_BBOXES` in the script currently has Pillur and Mettur commented out — only Bhavanisagar is active. Check this dict before assuming all three reservoirs will be (re)processed by a run.
- Running `main()` deletes and rebuilds the Zarr store for any reservoir whose `zarr_path` already exists (`shutil.rmtree`) — this is destructive to existing downloaded data.

### Salvage/repair (`salvage_zarr.py`)
A one-off recovery script for Zarr stores left with disjoint `stackstac-<hash>` arrays (an artifact of an earlier bug/interruption in the append flow) instead of a unified `landsat_data` array. It merges them in file-modification-time order, sets `_ARRAY_DIMENSIONS = ["time", "band", "y", "x"]` for xarray compatibility, deletes the original hashed arrays, and consolidates metadata. Only relevant if a Zarr store is found in that broken state; a healthy store already has a single `landsat_data` array.

### Visual QA (`visualize_zarr.py`)
Reads each `data/*.zarr` store directly via the low-level `zarr` API (bypassing `xarray` to avoid dimension-metadata constraints), pulls the first and last time-index scenes, and renders percentile-stretched (2nd–98th) True Color (RGB) and False Color (NIR-R-G) composites to `visualizations/`. It applies Landsat C2 scaling (`DN * 0.0000275 - 0.2`), but **the stored data is already surface reflectance** (stackstac rescaled it on download; values ≈ -0.2…1.0, QA_PIXEL stored as float). The double scaling is harmless here only because the percentile stretch cancels a linear transform. **New code must not rescale.** Band order assumption is fixed: index 0=Blue, 1=Green, 2=Red, 3=NIR, matching `TARGET_BANDS` in `data_collection.py` — if that band list ever changes, the hardcoded indices here must change too.
- The Mettur and Pillur stores (recovered via `salvage_zarr.py`) do keep their acquisition dates in a `time` array. They were unreadable by xarray only because `dimension_names` was missing; `scripts/fix_zarr_dims.py` added it (backups in `*.bak`; `--restore` undoes it). This script still reads via raw `zarr` and may label those frames `Index_0`/`Index_N`.

### Data layout
- `data/<Reservoir>_datacube.zarr/` — per-reservoir Zarr store, dims `[time, band, y, x]`, variable `landsat_data`.
- `data/ground_truth/` — validation points: `pilot/` (50 labeled BHA points, never overwrite) and `full/` (the 1,500-point v2 sample, being labeled). ISRO Bhuvan LULC maps were removed from the pipeline (access not granted), so training labels come from consensus rules (spectral indices + Esri LULC) and the basin prior from the data-derived maximum reservoir extent. See `progress.md`.
- `*.kml` (`bhavanisagar.kml`, `mettur.kml`, `pillur.kml`) — **just the download bounding-box rectangles** (`*_AOI`, same as `RESERVOIR_BBOXES`), not reservoir outlines. The reservoir extent is derived from the data (water frequency 2015–2026).
- `visualizations/` — generated QA composite PNGs, named `<Reservoir>_<First|Last>_<TrueColor|FalseColor>.png`.
- `Archive/` — superseded datasets/scripts, kept for reference only.

### Package `src/reservoir_cd/` (pilot stage)
Installed editable (`.venv/bin/pip install -e ".[dev]"`). `preprocessing/` (QA decoding, indices, Jan–Apr medoid composite, basin mask, Esri LULC, consensus labels) and `validation/` (strata S1–S8, point sampling and KML export) follow `EDD.md`. The thresholds live in `configs/preprocessing.yaml`. Run the pilot with `.venv/bin/python scripts/run_pilot.py` and the tests with `.venv/bin/python -m pytest`. `scripts/build_labels.py` (configs `sites.yaml`, `build.yaml`) builds the static forest map plus yearly rule/consensus labels for all three sites and years 2015–2026 into `data/interim/<Site>_labels.zarr` (static group + one group per year), `data/interim/labels_summary.json` and `visualizations/labels/<Site>/`; it does not draw the validation sample. The full v2 validation sample (1,500 points, 500 per site, S1-S7 from rule labels, all 11 Jan-Apr year pairs pooled) is drawn by `scripts/draw_validation_sample.py` (config `configs/validation.yaml`, logic in `validation/full_sample.py` and `strata.assign_strata_v2`) into `data/ground_truth/full/<SITE>/` (blind `*_labels.csv`, `*_key.csv` the labeler must not open, KML, `copernicus/`, `reference_chips/`) plus `strata_weights.csv`; the pilot in `data/ground_truth/pilot/` is never overwritten. Planning docs: `PRD.md`, `EDD.md`, `validation_protocol.md`, `LABELING_GUIDE.md`.

`datasets/` (`normalize.py` is the single place for the reflectance×10000 and Prithvi mean/std conventions; `chips.py` builds one compressed int16 scene Zarr per site under `data/processed/chips/<Site>.zarr` plus Parquet chip-pair indices; `splits.py` spatial-block + leave-one-site-out splits; `exclusion.py` validation-point masking at load time; `torch_dataset.py` `PrithviReservoirDataset`) is built by `scripts/make_chips.py` (config `configs/chips.yaml`). `models/` has Baseline A (`mndwi_otsu.py`, run with `scripts/run_baseline_a.py` → `data/processed/predictions/baseline_a/`), B (`siam_diff.py`), C/D (`prithvi.py`, TerraTorch; E4 variants by `model.adaptation` in `configs/train_{B,C,D}.yaml`) and `losses.py`. `training/` (Lightning; `python -m reservoir_cd.training.train --config configs/train_D.yaml`; checkpoints every epoch, auto-resume from `last.ckpt`) and `inference/predict.py` (tiled, overlap-blended full-scene inference) are meant to run on Colab (`notebooks/train_colab.ipynb`; package the data with `scripts/package_chips.py --with-keys --code`). The training stack is installed in `.venv` (`pip install -e ".[train]"`); locally only smoke tests are run (`pretrained=false`). Never apply the DN scaling to the stored reflectance.
