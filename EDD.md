# Engineering Design Document (EDD)

**Companion to:** `PRD.md` · **Status:** Draft v1 (2026-10-09)

---

## 1. Architecture Overview

A sequence of stages driven by config files. Each stage reads versioned artifacts from disk and writes new ones, so any stage can be rerun on its own.

```
 STAC catalog ──► [1 acquire] ──► raw Zarr cubes
                                      │
                                      ▼
                  [2 preprocess] ──► masked cubes + MNDWI + Otsu labels (+ basin mask)
                                      │
 KMLs ───────────────────────────────►┤
                                      ▼
                  [3 chip] ─────────► (T1, T2) 256×256 chip index
                                      │
                                      ▼
                  [4 train] ────────► model checkpoints (A, B, C, D)
                                      │
                                      ▼
                  [5 predict] ──────► transition maps (GeoTIFF/Zarr)
                                      │
 1,500 validation points ────────────►┤
 IMD / CWC tables ───────────────────►┤
                                      ▼
                  [6 evaluate] ─────► metrics, statistical tests, figures, tables
                                      │
                                      ▼
                  [7 release] ──────► Zenodo / GitHub / Hugging Face bundle
```

## 2. Recommended Tech Stack

| Concern | Choice | Why |
|---|---|---|
| Language | **Python 3.12** | Every ML and geospatial library supports it. 3.13 works for the data stack, but some ML wheels lag behind. |
| Environment and packaging | **uv** + `pyproject.toml` + lockfile | Fast and reproducible. Replaces the current undocumented `.venv`. |
| Catalog discovery | `pystac-client`, `planetary-computer` | Already in use |
| Loading rasters | `stackstac`, `rioxarray`, `rasterio` | Already in use. `rioxarray` adds CRS-aware I/O for GeoTIFF export. |
| Arrays and storage | `xarray`, `zarr`, `dask` | Already in use. Lazy, chunked processing that scales from a laptop. |
| Vectors (KMLs, points) | `geopandas`, `shapely` | Rasterizing the basin mask, sampling at points, spatial blocks for the bootstrap |
| Classical image processing | `scikit-image`, `scipy.ndimage` | Otsu thresholding and morphological opening |
| Deep learning | **PyTorch** + **Lightning** | Industry standard |
| Foundation model | **TerraTorch** (IBM/NASA) | The official toolkit for Prithvi-EO-2.0, with UPerNet decoders, multi-temporal input and LoRA (via `peft`) built in. Avoids writing the backbone integration yourself. |
| FC-Siam-diff baseline | `torchgeo` (`FCSiamDiff`) | Existing, peer-reviewed implementation |
| Configuration | YAML through Lightning CLI / TerraTorch | One config system across all stages. Every run is recorded by its YAML file. |
| Experiment tracking | TensorBoard (local) + CSV logger | Simple and works offline. Weights & Biases is optional. |
| Statistics | `statsmodels` (McNemar, `multipletests` with Holm), `scikit-learn` (confusion matrices) | Standard tools that reviewers recognize. The spatial block bootstrap is about 50 lines of NumPy. |
| Plotting | `matplotlib` | Already in use. Gives figures at publication quality. |
| Quality checks | `pytest`, `ruff` | Light tests on preprocessing math and data shapes |
| Release | Zenodo (data and code DOI), GitHub, Hugging Face Hub (LoRA weights) | Matches the requirements in architecture step 7 |
| Compute | Apple Silicon for stages 1–3 and 6. A cloud CUDA GPU (Colab Pro, Kaggle or a university cluster) for stages 4–5. | Fine-tuning a 300M-parameter backbone on Apple's MPS backend is slow and has gaps in operator support |

## 3. Modules

### 3.1 `acquisition`: data discovery
- Port of `data_collection.py`: STAC search, date-gated platform filter, 6-month chunks, Zarr append.
- **Changes:** save per-scene metadata (platform, cloud cover, scene ID) to a manifest, because the current flow drops non-dimension coordinates before appending. Make downloads resumable by skipping chunks already written. Replace the automatic `shutil.rmtree` with a `--overwrite` flag.
- Output: `data/raw/<site>.zarr`, plus a scene manifest CSV (scene ID, date, platform, cloud cover).

### 3.2 `preprocessing`: masking and labels
- `qa.py`: decode the QA_PIXEL bits (fill, dilated cloud, cloud, shadow), apply a 3×3 opening, and produce a NaN mask.
- `indices.py`: MNDWI = (Green − SWIR1) / (Green + SWIR1), after Collection-2 scaling (`DN × 2.75e-5 − 0.2`).
- `reservoir_extent.py`: water frequency over all clear scenes (all seasons) = share of clear scenes with MNDWI above a robust per-scene threshold. Permanent body = largest 8-connected component of frequency ≥ 0.75. **Maximum extent** = frequency ≥ 0.05 pixels 8-connected to the permanent body, closed by 1 px. **Study area** = maximum extent buffered by 2 km. (The project KMLs are only the download bounding boxes, so they are not used as a basin.)
- `forest.py`: **Forest is a static class** (one map for 2015–2026), because per-year NDVI forest swings with rainfall and the monsoon has almost no clear scenes. It is the consensus of (A) Esri multi-year Trees (`esri_trees_years` = number of years 2017–2023 with Trees; `forest_esri` = ≥ 4) and (C1) a Random Forest on pooled multi-year Landsat phenology. Per-pixel features from all clear dates (same-day scenes from adjacent rows are averaged first, so a date counts once): NDVI first-harmonic regression (mean, amplitude, phase as cos/sin; accumulated as per-pixel normal equations, so it streams), NDVI p10/p50/p90 and std, inter-annual std of the yearly Jan–Apr median NDVI, median MNDWI. Weak training labels, outside the maximum extent: positives = Esri Trees in ≥ 6 of 7 years, negatives = Crops/Rangeland/Built (any mix) in ≥ 6 of 7 years; balanced sample (≤ 20k per class per site), all three sites pooled. Evaluation: 1 km spatial-block CV and leave-one-site-out (against the weak Esri labels, so these are agreement figures). Outputs per site: `forest_prob_c1`, `forest_c1` (prob ≥ 0.5), `esri_trees_years`, `forest_esri`. **Rule forest** (full coverage, used for strata) = `forest_c1`. **Consensus forest** (training) = forest where `forest_c1 & forest_esri`, non-forest where neither, 255 on disagreement or where \|prob − 0.5\| < 0.15.
- `labels.py`: classes Water > Bare > Forest > Other, per year. Water: composite MNDWI > Otsu t. Bare: inside the maximum extent, not water, composite NDVI below `bare.ndvi_max` (exposed reservoir bed). Forest: the static rule forest (outside water and bare). Other: the rest. Per-scene Otsu thresholds (used for temporal agreement and the frequency map) are tested for bimodality and fall back to the year's composite threshold. The MNDWI confidence margin applies to Water and in-extent Bare.
- Consensus (per year): Forest and Other come from the static consensus forest map (Forest kept where consensus = forest, Other where consensus = non-forest). Water: Esri Water in Esri years (2017–2023), temporal agreement ≥ 75% of clear Jan–Apr scenes otherwise, plus the MNDWI margin. In-extent Bare: temporal agreement + margin (Esri mislabels exposed bed). Other additionally needs Esri ∈ {Crops, Rangeland, Built} in Esri years, temporal agreement otherwise.
- Output: `data/interim/<site>_processed.zarr` (reflectance, valid mask, MNDWI, label, water frequency, extent).
- **Label parameters** live in `configs/preprocessing.yaml`. They are set during the pilot and **frozen before any validation point is scored**:

  | Parameter | Starting value (placeholder) | Notes |
  |---|---|---|
  | Water frequency thresholds | permanent ≥ 0.75, maximum extent ≥ 0.05, closing 1 px | Extent must connect to the permanent body |
  | Study area | maximum extent + 2 km | Region for Otsu and for strata / sampling |
  | Safety bounds on the Otsu threshold | [-0.1, 0.3] on MNDWI | Per-scene value on or outside a bound means fallback |
  | Per-scene bimodality test | Otsu between/total variance ≥ 0.5 | Else fall back to the composite threshold |
  | Confidence margin for MNDWI | \|MNDWI − t\| ≥ 0.1 | Only for Water and in-extent Bare |
  | Static forest, C1 model | RandomForest (150 trees, depth 16, leaf 5, seed 42) on 10 phenology features | `forest.rf`; ≥ 8 clear dates per pixel |
  | Static forest, training labels | positives: Esri Trees ≥ 6/7 yr; negatives: Crops/Rangeland/Built ≥ 6/7 yr; outside max extent | ≤ 20k per class per site |
  | Static forest, Esri | `forest_esri` = Trees in ≥ 4 of 7 years (2017–2023) | |
  | Static forest, rule / consensus | rule = P ≥ 0.5; consensus = both agree, 255 if disagree or \|P − 0.5\| < 0.15 | Replaces the per-year NDVI p90 rule (legacy keys kept for the BHA pilot) |
  | Bare rule (inside max extent) | not water and NDVI < 0.35 | Spec default 0.25 left the pale silt ring (NDVI mode ≈ 0.28) as Other |
  | Bare rule (outside max extent) | off by default (NDVI < 0.15 and mean(red, SWIR1) > 0.25 when on) | Catches almost nothing in the pilot |
  | Mapping of Esri classes to ours | Water→Water, Trees→Forest, Bare→Bare, Crops/Built/Rangeland→Other, Flooded veg→excluded | |
  | Temporal agreement | ≥ 75% of clear Jan–Apr scenes agree | Years without Esri, and in-extent Bare |
  | Minimum clear observations per pixel | 2 | |
  | Shoreline width (stratum S5) | 1 px | |
  | S8 | Bare↔Other only, with \|Δ Jan–Apr composite NDVI\| ≥ 0.15 | Forest is static, so Forest↔Bare/Other changes are excluded; class change alone is not enough |


### 3.3 `datasets`: chips and the dataloader (implemented 2026-10-10; `scripts/make_chips.py`, `configs/chips.yaml`)
- **Store (differs from "materialise every chip"):** one zstd-compressed Zarr per site, `data/processed/chips/<Site>.zarr`, holds the *yearly full-scene* rasters: `refl` int16 (Y,6,H,W) = reflectance × 10000 (HLS units, −32768 = no data), `label_cons` / `label_rule` / `label_esri` uint8, `n_clear`, `date_days` (medoid date), `study_area`, `max_extent`. Chips are windows into it, so stride-128 overlap costs nothing, the Colab payload is ~142 MB, and the same store feeds full-scene inference. (`normalize.py` is the only place for the ×10000 conversion and the Prithvi mean/std, which were checked against the HF `Prithvi-EO-2.0-300M-TL` config.)
- `pairs.py`: consecutive-year Jan–Apr pairs (stride 1 → 11 pairs/site).
- `chips.py`: 256×256 windows; train grid stride 128, eval grid stride 256 (the last row/column is shifted back to the scene edge, so no padding). `<Site>_positions.parquet` (windows) and `<Site>_index.parquet` (chip-pair index: one row per window × year pair with `valid_frac`, `labeled_frac`, water/loss/gain pixel counts, per-chip median medoid date → year/DOY, centre lat/lon). A chip-pair is dropped if `valid_frac` (clear at both dates) < 0.5 or `labeled_frac` (consensus label at both dates) < 0.02.
- `torch_dataset.py`: `PrithviReservoirDataset` returns `image [2,6,256,256]` (normalized, no-data = 0), `mask [2,256,256]` (per-date labels, 255 ignore), `change [256,256]` (0 none / 1 water loss / 2 water gain / 255 if either date unlabeled), `valid [256,256]`, `temporal_coords [2,2]` (year, 0-based DOY), `location_coords [2]` (lat, lon). `label_source` = consensus | rule | esri (E2/E3). Validation-point exclusion (`exclusion.py`: every `data/ground_truth/**/*_key.csv`, plus `validation_points.gpkg` if present; default radius 150 m) is applied at load time to all year pairs.
- `splits.py`: within each site, val/test are whole 256×256 eval windows (containing reservoir extent and true water change, chosen greedily to lose the fewest train windows, seeded); train windows must share no pixel with them (`gap_px` = 0). The "1-chip gap" is configurable but infeasible on these scenes: with 128 px BHA has no within-site val/test left, with 256 px every site is train-only. Pillur (372×367) is too small for any disjoint val/test and is train-only within-site. Leave-one-site-out folds (E1) are columns `loso_<Site>` in `splits.parquet` (held-out site → test, other sites' train/val → train/val).

### 3.4 `models`: four baselines
- `mndwi_otsu.py` (Baseline A, `scripts/run_baseline_a.py`): per-composite Otsu inside the study area (clamp + bimodality fallback of `labels.robust_scene_threshold`), Water vs non-water only (land classes N/A). Full-scene output `data/processed/predictions/baseline_a/<Site>.zarr` (`water` per year, `change` per pair) and `summary.json`.
- `siam_diff.py` (B): torchgeo `FCSiamDiff` with `2 × 4` output channels split into per-date logits.
- `prithvi.py` (C/D): TerraTorch `EncoderDecoderFactory`, backbone `prithvi_eo_v2_300_tl` (T = 2, time + location coords), `SelectIndices [5,11,17,23]` → `ReshapeTokensToImage(effective_time_dim=2)` → `LearnedInterpolateToPyramidal` → UPerNet; head `T × 4` channels → per-date logits `[B,2,4,H,W]`; the water-change map is derived from the two per-date argmaxes (`change_from_logits`). This replaces "T1 and T2 features concatenated" (time is folded into channels inside the neck). Adaptation by config (`model.adaptation`): `frozen` (C), `lora` with `r` ∈ {4, 8, 16}, α = 2r (D = r 8, α 16) on attention q/k/v (split by TerraTorch's `replace_qkv`) and `proj`, or `full` (E4).
- `losses.py`: cross-entropy with `ignore_index = 255`, averaged over the two dates.

### 3.5 `training`
- `training/module.py` (Lightning module + datamodule) and `training/train.py` (CLI): seed, CSV + TensorBoard logging, checkpoint every epoch (`last.ckpt` + best), early stopping on `val_loss`, automatic resume from `<output-dir>/checkpoints/last.ckpt`. Checkpoints of C/D omit the frozen pretrained backbone (~420 MB instead of ~1.3 GB each; rebuilt with `pretrained: true`). One YAML per baseline: `configs/train_{B,C,D}.yaml`; E1/E2/E3/E4 are `--set` overrides (`data.loso_holdout`, `data.label_source`, `model.adaptation.*`). UPerNet's pooling BatchNorm needs a batch of ≥ 2 in train mode.

### 3.6 `inference`
- `inference/predict.py`: tiled full-scene inference (256 tiles, overlap 128, triangular blend of per-date softmax probabilities, tile-wise temporal/location coords). Writes `pred` (P,2,H,W), `change` (P,H,W) and `water_prob` to Zarr (COG export not implemented).

### 3.7 `evaluation`
- `sampling.py`: read predictions at the validation points, using the 15m buffer rule.
- `metrics.py`: confusion matrices, overall accuracy, per-class F1 and F1 on true changes.
- `stats.py`: McNemar's test (overall and on true-change points), Holm-Bonferroni correction across the 6 pairwise comparisons, and a spatial block bootstrap for confidence intervals.
- `context.py`: join the IMD rainfall and CWC storage tables to time series of water area.
- `figures.py` / `tables.py`: outputs ready for the paper.

### 3.8 `qa`
- Port of `visualize_zarr.py`. Also produces per-stage summaries: scene counts, cloud fraction, label balance.

## 4. Directory Structure

```
LandClassifciation/
├── pyproject.toml            # dependencies + CLI entry points
├── uv.lock
├── README.md  PRD.md  EDD.md  progress.md  base_architecture.md
├── configs/
│   ├── sites.yaml            # bounding boxes, KML paths, CRS
│   ├── acquisition.yaml
│   ├── preprocessing.yaml
│   ├── chips.yaml            # + train_{B,C,D}.yaml
│   ├── models/{a_mndwi,b_siamdiff,c_prithvi_frozen,d_prithvi_lora}.yaml
│   └── evaluation.yaml
├── src/reservoir_cd/
│   ├── acquisition/          # stac.py, cube.py
│   ├── preprocessing/        # qa.py, indices.py, labels.py, basin.py
│   ├── datasets/             # pairs.py, chips.py, torch_dataset.py, splits.py
│   ├── models/               # mndwi_otsu.py, siam_diff.py, prithvi.py, losses.py
│   ├── training/             # train.py
│   ├── inference/            # predict.py
│   ├── evaluation/           # sampling.py, metrics.py, stats.py, context.py, figures.py, tables.py
│   ├── qa/                   # composites.py, reports.py
│   └── cli.py                # `rcd acquire|preprocess|chip|train|predict|evaluate|qa`
├── scripts/                  # one-off utilities (salvage_zarr.py)
├── notebooks/                # exploration only; no logic the pipeline depends on
├── data/                     # gitignored
│   ├── boundaries/           # *.kml
│   ├── raw/                  # <site>.zarr + scene manifests
│   ├── interim/              # processed cubes, labels
│   ├── processed/            # chip indices
│   ├── ground_truth/         # validation_points.gpkg
│   └── context/              # IMD rainfall, CWC storage
├── outputs/                  # gitignored: checkpoints/, predictions/, metrics/, figures/, logs/
├── tests/                    # pytest: QA bit decoding, MNDWI, chip shapes, stats
└── Archive/
```

## 5. Key Design Decisions

1. **Stages communicate only through files.** Each stage can be rerun and inspected alone, which helps with the expensive download stage.
2. **Zarr for cubes, Parquet/GeoPackage for tables and vectors, COG for maps.** Each is the native format for its data type.
3. **TerraTorch instead of custom Prithvi code.** Less integration code means fewer reproducibility problems and more credibility with reviewers.
4. **Spatial splits plus independent points.** This is the core defense against circular evaluation.
5. **Config files are the record of each experiment.** Every result file stores the config, commit hash and data manifest that produced it.

## 6. Migrating the Existing Code

| Current | Target |
|---|---|
| `data_collection.py` | `src/reservoir_cd/acquisition/` + `configs/sites.yaml` (removes the commented-out bounding boxes) |
| `visualize_zarr.py` | `src/reservoir_cd/qa/composites.py` |
| `salvage_zarr.py` | `scripts/salvage_zarr.py` |
| `data/*.zarr` | `data/raw/` (no new download needed; consider re-pulling later to recover timestamps) |
| `*.kml` | `data/boundaries/` |

Migrate step by step: add `pyproject.toml` first, then move one module at a time as each stage is built. Don't do a big-bang rewrite.
