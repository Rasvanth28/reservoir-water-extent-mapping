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
Reads each `data/*.zarr` store directly via the low-level `zarr` API (bypassing `xarray` to avoid dimension-metadata constraints), pulls the first and last time-index scenes, and renders percentile-stretched (2nd–98th) True Color (RGB) and False Color (NIR-R-G) composites to `visualizations/`. Applies Landsat Collection 2 reflectance scaling (`DN * 0.0000275 - 0.2`) before compositing. Band order assumption is fixed: index 0=Blue, 1=Green, 2=Red, 3=NIR, matching `TARGET_BANDS` in `data_collection.py` — if that band list ever changes, the hardcoded indices here must change too.

### Data layout
- `data/<Reservoir>_datacube.zarr/` — per-reservoir Zarr store, dims `[time, band, y, x]`, variable `landsat_data`.
- `data/ground_truth/bhuvan/` — ISRO Bhuvan LULC static basin masks (weak supervision/topographical priors, not yet ingested per `progress.md`).
- `*.kml` (`bhavanisagar.kml`, `mettur.kml`, `pillur.kml`) — reservoir boundary geometries from Bhuvan.
- `visualizations/` — generated QA composite PNGs, named `<Reservoir>_<First|Last>_<TrueColor|FalseColor>.png`.
- `Archive/` — superseded datasets/scripts, kept for reference only.

### Upcoming components referenced in the docs but not yet present
`preprocessing.py` (MNDWI + Otsu dynamic water masking, QA_PIXEL morphological opening) and `dataset.py` (`PrithviReservoirDataset` PyTorch class for 256×256 tensor chipping) are planned per `base_architecture.md`/`progress.md` but do not exist yet — do not assume they're implemented elsewhere.
