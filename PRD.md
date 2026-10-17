# Product Requirements Document (PRD)

**Project:** Reservoir Water-Extent Change Detection with Prithvi-EO-2.0
**Purpose:** Research software behind an ISPRS / TGRS paper submission
**Status:** Draft v2 (2026-10-10). See `progress.md` for live status and `EDD.md` for the engineering design.

---

## 1. Problem Statement

Water extent in South Indian reservoirs changes because of both **structural causes** (siltation, encroachment, operating changes) and **climate causes** (monsoon variability, drought). Current methods either:
- use spectral indices such as MNDWI, which are noisy and easily confused by shadows and wet fields, or
- use static land-cover maps, which are released years apart and miss changes within a year.

No reproducible benchmark yet tests whether a geospatial foundation model (Prithvi-EO-2.0), adapted cheaply with LoRA, detects these changes better than the classical approaches.

## 1a. Novelty and Contributions

**Novelty statement**

> A geospatial foundation model (Prithvi-EO-2.0), adapted with LoRA and trained **only on automatically generated consensus labels** (spectral indices plus global land-cover products, filtered by confidence), detects year-over-year reservoir water-extent transitions **more accurately than its own label sources**. The gain is largest on hard pixels (shorelines, shadows, muddy water) and on true-change pixels. We verify this against 1,500 independent hand-digitized points with statistical testing that accounts for spatial correlation, and we link the detected changes to rainfall and storage records to separate structural change from climate-driven change.

| Contribution | Role in the paper |
|---|---|
| C1. Foundation model beats its weak label sources on hard and true-change pixels | **Core claim.** No prior work found (literature check, 2026-10-10). |
| C2. Multi-year (2015–2026) consecutive-year reservoir change detection with no manual training labels | Main application contribution |
| C3. Evaluation designed against circular labels: independent points, McNemar's test on true-change points, spatial block bootstrap, Holm correction | Methodological rigor |
| C4. Separating structural from climate-driven change (IMD rainfall, CWC storage) | Hydrological insight |
| C5. Open release of validation set, LoRA weights and code | Reproducibility |

**Not claimed as novel:** consensus labels (prior work: RSMA 2024, SinoLC-1, Tanzania IJAEOG 2023, a 2026 European mountains study) or the Prithvi + LoRA + UPerNet components. These are presented as tools.

**Closest prior work to differentiate from:** Wang et al. 2026 (Transactions in GIS, self-supervised long-term LULC labels; full text still needed), arXiv 2610.06704 (Dynamic World labels for Sentinel-1 water), Chen & Bruzzone 2022 (TGRS, spectral-index pseudo-labels).

## 2. Product Positioning

A **headless research pipeline**: a set of command-line stages with no UI. It turns raw satellite archives into:
1. a discovered, analysis-ready Landsat time series for each reservoir,
2. maps of water-extent changes over time, and
3. statistically defensible evidence (tables, figures, released artifacts) for the paper.

"Finding and discovering data" means two things here:
- **Data discovery:** querying satellite catalogs to find usable, cloud-free scenes.
- **Change discovery:** finding where and when reservoir extent changed, and whether rainfall or storage explains the change.

## 3. Users

| User | Need |
|---|---|
| Primary researcher (you) | Run each stage repeatably and iterate on models and experiments |
| Co-authors and advisors | Inspect intermediate outputs (QA composites, metrics) without running code |
| Peer reviewers | Trust the results: no circular evaluation, correct statistics |
| Future researchers | Reproduce the results from the Zenodo/GitHub release |

## 4. Goals and Non-Goals

**Goals**
- G1. Dense 2015–2026 Landsat 8/9 time series for Pillur, Bhavanisagar and Mettur.
- G2. Automatically generated **consensus labels** for Water / Forest / Bare Land, plus **Other** for farmland, towns and scrub: spectral rules (MNDWI + Otsu, NDVI, bare-soil index) agreeing with Esri 10m LULC where available (2017–2023), or with temporal consistency otherwise. Only high-confidence pixels are used for training, and the result is constrained by a basin prior.
- G3. A four-way benchmark: (A) MNDWI + Otsu, (B) FC-Siam-diff, (C) frozen Prithvi, (D) Prithvi + LoRA.
- G4. Evaluation against N = 1,500 independent, hand-digitized points, with McNemar's test, Holm-Bonferroni correction and a spatial block bootstrap.
- G5. Climate context (IMD rainfall, CWC storage) to separate structural changes from climate-driven ones.
- G6. One-command reproducibility and a public release (code, weights, validation set).
- G7. The supporting experiments in §5a, which are needed for a Q1 venue.

**Non-Goals**
- No web app, dashboard or API service. Reports and figures are the outputs.
- No real-time or operational monitoring.
- No reservoirs beyond the three study sites (the design should still allow adding more).
- No dependence on ISRO Bhuvan LULC maps (access not granted; see `progress.md`).

## 5. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR1 | Query Planetary Computer STAC by bounding box, date range, cloud cover and platform rule (L8 only before 2022; L8 + L9 from 2022) | Must |
| FR2 | Write per-reservoir Zarr datacubes with real acquisition timestamps, resumable after interruptions | Must |
| FR3 | Decode the QA_PIXEL cloud mask, apply a 3×3 morphological opening, and set clouds to NaN | Must |
| FR4 | Compute MNDWI, apply Otsu thresholding, and constrain the result with the basin mask derived from the maximum reservoir extent (water frequency) | Must |
| FR5 | Build co-registered pairs (T1, T2) and cut them into 256×256 chips in Prithvi's 6-band order | Must |
| FR6 | Train and run inference for baselines A–D from config files | Must |
| FR7 | Load the 1,500 validation points (paired labels C_T1 and C_T2) and sample predictions at those points | Must |
| FR8 | Produce confusion matrices, McNemar's tests (overall and on true-change points), Holm-Bonferroni correction and a block bootstrap | Must |
| FR9 | Generate visual QA composites and per-stage summaries | Should |
| FR10 | Join the IMD rainfall and CWC storage tables to the detected changes | Should |
| FR11 | Export paper figures and tables (PNG/PDF, CSV/LaTeX) | Should |
| FR12 | Package code, weights and the validation set for Zenodo and GitHub | Must (at submission) |

## 5a. Supporting Experiments (needed for Q1)

| ID | Experiment | Question it answers | Priority |
|---|---|---|---|
| E1 | **Leave-one-site-out:** train on 2 reservoirs and test on the 3rd, rotating through all three | Does the method generalize beyond the training sites? | Must |
| E2 | **Label-source ablation:** consensus vs. spectral rules only vs. Esri only | Is consensus what drives the gain? | Must |
| E3 | **Confidence-filter ablation:** train on high-confidence pixels only vs. all pixels | Does filtering out uncertain pixels matter? | Must |
| E4 | **Adaptation ablation:** frozen backbone vs. LoRA (r = 4/8/16) vs. full fine-tuning | Is LoRA sufficient and efficient? | Must |
| E5 | **Results split by difficulty:** easy vs. hard (shoreline, shadow, muddy water) vs. true-change pixels | Direct evidence for claim C1 | Must |
| E6 | **Comparison with existing water products:** JRC Global Surface Water (to 2020), DSWE | Why use this model instead of existing products? | Must |
| E7 | **A second foundation model** (e.g. Clay or a smaller ViT) | Is the result specific to Prithvi? | Should |
| E8 | **Label-noise sensitivity:** inject controlled noise into the labels and track accuracy | How robust is the claim to label quality? | Could |

The validation points must include enough true-change locations for E5 and the McNemar's test on true-change points to have statistical power. The sampling design is in `validation_protocol.md` (v2: 7 strata, 900 shoreline and water-change points).

**Primary vs. secondary endpoint (decided 2026-10-13, after the pilot).**
- **Primary: water.** Water vs. non-water at T1 and T2, and the derived water transition (stable water, stable land, loss, gain). All headline claims (C1, E1–E6) and McNemar tests use this endpoint. The pilot showed the reference labels are reliable for water: the rule agreed with the interpreter on 29/30 rule-Water pixels and 8/8 water-gain points.
- **Secondary: the four land classes** (Water / Forest / Bare / Other). They are still trained and predicted, but reported only as a secondary result, computed on points whose land class was labeled from high-resolution Google Earth imagery. Reason: in the pilot, 81% of reference dates had only 10 m Sentinel-2. At that resolution, dry-season deciduous forest, thorn scrub and bare ground could not be told apart reliably (automatic vs. human Forest/Other agreement was poor).

## 6. Non-Functional Requirements

- **Reproducibility:** pinned dependencies, fixed random seeds, all settings in config files, and a record of the data source for every run.
- **Robustness:** survive Planetary Computer token expiry (chunked, resumable downloads). Never delete existing data without an explicit flag.
- **Scientific validity:** validation points are never used in training. Train/test splits are spatially separated.
- **Compute (decided 2026-10-10):** the **local Mac** handles light work: downloads, preprocessing, compositing, labels, sampling, evaluation and figures. **Google Colab** handles heavy work: Prithvi/FC-Siam-diff training and full-scene inference. Data moves to Colab as chip files via Google Drive, not as raw Zarr cubes. Training must checkpoint regularly and resume after a session disconnects.
- **Traceability:** every metric in the paper can be traced to a config, a commit hash and a data version.

## 7. Success Metrics

- Baseline D beats Baselines A–C on water accuracy and on water-change (loss/gain) F1, with McNemar's p < 0.05 after Holm correction.
- **Baseline D beats every one of its label sources on hard water pixels (E5: shoreline, muddy/weedy water, true change).** This is the go/no-go test for claim C1. If it fails, reposition the paper as an application study.
- The gain holds in leave-one-site-out testing (E1).
- One command rebuilds every paper table from released artifacts.
- The release passes the ISPRS/TGRS reproducibility checklist.

## 8. Milestones

| # | Milestone | Maps to architecture step |
|---|---|---|
| M1 | Datacubes and QA finished (cloud masking, climate tables) | Step 1 |
| M2 | Labels: MNDWI/Otsu masks and the 1,500 digitized points | Step 2 |
| M3 | Chip dataset and dataloader | Step 3 |
| M4 | Baselines A–C running | Step 4 |
| M5 | Prithvi + LoRA trained | Step 5 |
| M6 | Statistical evaluation finished | Step 6 |
| M7 | Release and manuscript artifacts | Step 7 |

## 9. Open Questions and Risks

1. ~~**The label classes don't match.**~~ **Decided (2026-10-09):** all three classes come from consensus labels (see G2). **Decided (2026-10-10):** farmland, towns and scrub go into a 4th class, **Other**, for now. The validation points include an Other stratum (Option A); see `validation_protocol.md`.
2. ~~**Choosing pairs from a dense series.**~~ **Decided (2026-10-09):** pair the same season in consecutive years (stride 1: 2015→2016, 2016→2017, …, 2025→2026). Season window: **January–April** (dry season, fewest clouds). Still to define: medoid vs. median compositing (EDD §3.2).
3. **The Baseline A paradox.** *Direction chosen (2026-10-09): consensus labels (spectral rules + Esri LULC where available) trained only on high-confidence pixels, with results reported separately for easy, hard and true-change pixels.* Baseline D is trained on the output of Baseline A. The paper must explain how D can beat its own teacher (spatial context, denoising). This is defensible but needs to be argued explicitly.
4. ~~**Where the KMLs come from.**~~ **Resolved (2026-10-10):** the KMLs are only bounding-box rectangles, not Bhuvan outlines, so there's nothing to cite. The reservoir extent and basin prior are derived from the data instead (water frequency 2015–2026).
5. **Colab limits.** Sessions time out and GPU type varies, and E1, E4 and E7 multiply the number of training runs. Mitigations: checkpointing with resume, mixed precision, a small batch size, and a run log. Estimate the total GPU hours after the pilot to judge whether Colab Pro compute units are enough.
6. **The core claim may fail.** If D only matches its label sources, C1 disappears. Run a small pilot (one site, a few year pairs) before building at full scale. *Labeling pilot done (2026-10-13, 50 BHA points; see `validation_protocol.md` §8):* the rule labels miss water on 9 of 70 rule-non-water labels (shorelines, muddy edges, river banks). These are the hard pixels where D can beat its teacher.
9. **2015 reference imagery.** Sentinel-2 has no January–April 2015 images, so T1 of the 2015→2016 pair relies on Google Earth or, as a last resort, Landsat (the model's own input, at the same resolution). Points labeled from Landsat are flagged, and results are reported with and without them.
7. **Target venue.** ISPRS J. / TGRS / RSE need E1–E7. IJAEOG, JSTARS and GIScience & RS are the realistic Q1 fallbacks. Check current quartiles before submitting.
8. **Wang et al. 2026 full text** is still needed to confirm the overlap with C1 (behind a paywall; request it from the author).
