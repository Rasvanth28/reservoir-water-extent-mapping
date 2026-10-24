# Project Progress & Architecture Alignment

This document tracks our current progress against the pipeline defined in `base_architecture.md`, highlighting what has been strictly followed and where we have made strategic architectural adaptations.

## Update 2026-10-24 (accuracy assessment against v1)
*   **Evaluation module** `src/reservoir_cd/evaluation/` (`sampling.py`, `metrics.py`: Stehman 2014 stratified ratio estimator with site × S1–S7 strata; `stats.py`: exact McNemar, Holm, stratified 1 km block bootstrap) and `scripts/evaluate.py` (config `configs/evaluation.yaml`) → `outputs/metrics/v1/` (`metrics.csv`, `mcnemar.csv`, `confusion.csv`, `run.json`). Tests in `tests/test_evaluation.py`.
*   **Results (area-weighted, all sites):** Baseline A water F1 0.981 [0.976, 0.985], loss F1 0.897, gain F1 0.884. Pillur is the weak site: water F1 0.856, loss F1 0.31, gain F1 0.57. Hard strata (sample accuracy): S5 shoreline 0.87, S6 loss 0.84, S7 gain 0.88; 31% of S6 and 22% of S7 points are stable per the interpreter (false change is the main error).
*   **Finding:** rule-label water is identical to Baseline A at all 3,000 point-dates (same per-year MNDWI-Otsu threshold), and consensus water equals it where labeled (coverage 86%). Models trained on rule/consensus labels therefore learn Baseline A's water map; the gain must come from generalising beyond the labels (Pillur, shoreline, false change). Consensus accuracy is on its labeled 86% only.

## Update 2026-10-20 (validation sample labeled and frozen)
*   **All 1,500 v2 validation points labeled** (500 per site, interpreter KC, `scripts/label_app.py`), QC clean, and **frozen as v1**: `scripts/freeze_validation.py --version v1` → `data/ground_truth/frozen/v1/` (`validation_points.gpkg/.csv`, `strata_weights.csv`, `manifest.json` with sha256 of all inputs/outputs; tracked copy `validation_freeze/v1.json`). `pilot_qc.py` rules updated to the 2026-10-19 decision log (mixed/S2 may have confidence 3).
*   **Second interpreter done (2026-10-24):** VS labeled the 150-point subset (`scripts/interpreter_agreement.py`, report `validation_freeze/v1_interpreter_agreement.txt`, VS labels archived with hashes in `data/ground_truth/frozen/v1_second_interpreter/`). Water vs non-water κ 0.70 (BHA 0.81, MET 0.80, PIL 0.50); 4-class κ 0.39; water-transition κ 0.65. **Decision:** both interpretations kept as-is, no adjudication; KC (v1) remains the reference, so there is no v1.1. Known limitation to report: VS never used Other, marked mixed once in 300, and labeled 35 KC-Water point-dates as Forest (23 at Pillur, mostly shoreline squares).

## Update 2026-10-10 (labels, chips, Baseline A, training scaffolding)
*   **Labels:** static consensus forest + yearly rule/consensus labels are built for all three sites and 2015–2026 (`data/interim/<Site>_labels.zarr`, `scripts/build_labels.py`).
*   **Chips (Step 3):** done. `scripts/make_chips.py` builds one int16 scene Zarr per site + Parquet chip-pair indices + spatial/leave-one-site-out splits under `data/processed/chips/` (~142 MB total, zipped by `scripts/package_chips.py`). `PrithviReservoirDataset` implemented, with validation-point exclusion applied at load time.
*   **Baseline A:** `scripts/run_baseline_a.py` writes full-scene MNDWI+Otsu water/change maps for every pair (`data/processed/predictions/baseline_a/`).
*   **Scaffolding (no real training yet):** Baseline B (FC-Siam-diff), C/D (TerraTorch Prithvi, frozen / LoRA), Lightning training with resume, tiled inference, `notebooks/train_colab.ipynb`. Smoke-tested locally with random-init weights only.
*   **Open:** pretrained weights not yet downloaded/tested; no real training run; the 1,500-point sample is not drawn; local disk is nearly full (see report).

## Current Status: 
**In Progress: Step 1 (Data Engineering) transitioning into Step 3 (Tensor Generation)**

---

## 1. Geospatial Data Engineering
**Status:** `PARTIALLY COMPLETE (80%)`

### What is the SAME as the plan:
*   **Locations:** Successfully queried and downloaded data for all three target sites: Pillur Dam, Bhavanisagar, and Mettur.
*   **Sensor Setup:** Retrieved Landsat Level-2 Surface Reflectance data with the exact requested bands (Blue, Green, Red, NIR, SWIR1, SWIR2) alongside the 16-bit `QA_PIXEL` CFMask band.
*   **Local Storage:** The raw data is properly staged on disk (as highly-optimized Zarr stores) ready for preprocessing.

### What has CHANGED / ADAPTED:
*   **Dense Time Series vs. Anniversary Dates:** 
    *   *Plan:* Download anniversary-date pairs (March 2015 & March 2023).
    *   *Implementation:* We significantly upgraded this. We pulled a **dense multi-year time series from 2015 all the way to 2026**. This gives the model infinitely more data to fine-tune on, rather than limiting ourselves to two static dates. 
*   **Strict Platform Rules:** We applied a dynamic STAC filter ensuring only Landsat-8 is used before 2022, and both Landsat-8/9 are used from 2022 onwards.
*   **Chunking Fix:** Due to Planetary Computer SAS token timeouts on massive datasets (like Mettur), we implemented a bi-annual (6-month) temporal chunking and Zarr-appending strategy that was not explicitly in the base architecture.
*   **Added Visual QA Phase:** We implemented a `visualize_zarr.py` script to automatically generate 2nd-98th percentile stretched True/False Color composites to visually verify data integrity before feeding it to the model.

### What is PENDING for Step 1:
*   **Morphological Masking:** The 3x3 morphological opening (erosion/dilation) on the `QA_PIXEL` mask has not been executed yet. We deferred this to the PyTorch dataset pipeline (`dataset.py`) so it can be dynamically computed.
*   **Climatic Baselining:** We have not yet compiled the IMD gridded rainfall and CWC reservoir storage data into the tabular framework.

---

## 2. Ground Truth & Labeling Protocol
**Status:** `IN PROGRESS` (updated 2026-10-20)
*   **Training labels done (2026-10-10):** static forest (Esri + phenology RF consensus) and yearly rule/consensus labels for all 3 sites, 2015–2026 (`scripts/build_labels.py`). The KML basin idea below was replaced by the data-derived maximum reservoir extent.
*   **Labeling pilot done (2026-10-13):** 50 Bhavanisagar points labeled by the interpreter (`data/ground_truth/pilot/BHA_labels_filled.csv`; checks in `scripts/pilot_qc.py`; results in `validation_protocol.md` §8). Water reference labels are reliable; land classes are not, from 10 m Sentinel-2. **Decision:** water is the primary endpoint and land classes secondary (PRD §5a). Validation design moved to v2 (7 strata, shoreline/water-change heavy). Labeling aids: `scripts/make_reference_chips.py` (Sentinel-2 chips, no account) and `scripts/make_copernicus_links.py`.
*   **Full sample done (2026-10-20):** 1,500 v2 points drawn, labeled and frozen as v1 (`data/ground_truth/frozen/v1/`). Second-interpreter κ done (2026-10-24, kept as-is, no v1.1).
*   **ISRO Bhuvan LULC maps removed from the pipeline (for now).** Our data access request to Bhuvan has not been granted, so we are not depending on it.
    *   *Plan:* Bhuvan LULC maps resampled to 30m as (a) weak supervision labels for LoRA training and (b) static topographical basin masks.
    *   *Adaptation:* Training labels will come from the dynamic MNDWI + Otsu masks generated from the Landsat stack itself. The basin constraint can use the reservoir boundary KMLs we already have (`*.kml`), rasterized to the 30m grid. If Bhuvan access is granted later, it can be added back as an extra prior.

---

## 3. HLS-Aligned Tensor Patch Generation
**Status:** `DONE (2026-10-10)`
*   `PrithviReservoirDataset` and the chip store/index/splits are implemented (see the update at the top and `EDD.md` §3.3). T=2 pairs are consecutive-year Jan–Apr medoid composites (11 per site).

---

## Steps 4, 5, 6, and 7
**Status:** `NOT STARTED`
*   Foundation model benchmarking, LoRA fine-tuning, McNemar's testing, and Open Science packaging remain completely untouched.
