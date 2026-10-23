# Validation Sampling Protocol (N = 1,500)

**Status:** v2 (2026-10-13, after the pilot; v1 2026-10-10) · **Implements:** architecture Step 2, PRD G4 / E5 / M2
**Classes:** Water · Forest · Bare Land · Other (farmland, towns, scrub)
**Pairs:** January–April medoid composites, consecutive years (2015→2016 … 2025→2026, 11 pairs)

---

## 1. Principles

1. **Independent of training.** The points are never used for training. Pixels within 5 px (150 m) of a validation point are masked out of the training loss.
2. **Stratified random sampling**, as recommended by Olofsson et al. (2014). Strata come from a preliminary map; points are drawn at random within each stratum. Accuracy is reported with **area-weighted estimators**, so the unequal allocation does not bias the results.
3. **Strata come from the rule-label map, not from any model.** All four baselines are scored on the same points, so the sample cannot favor any one model.
4. **One point = one location + one year pair.** Each point is labeled at T1 and T2. Points are independent, which McNemar's test requires.

## 2. Strata and allocation (per site; × 3 sites)

**v2 (2026-10-13, after the pilot):** water is the primary endpoint (PRD §5a), so the allocation favours shoreline and water-change points. S8 (land change) is dropped: its pixels join S4, and its points go to the water strata.

The strata partition the (pixel, year pair) population, using the **rule labels** at T1 and T2. The rule labels cover every pixel. The consensus labels leave hard pixels unlabeled, and an area-weighted estimate needs every pixel to belong to a stratum.

| # | Stratum | Definition | Points per site | Total |
|---|---|---|---|---|
| S1 | Stable Water | Water at T1 and T2, interior | 50 | 150 |
| S2 | Stable Forest | Forest at T1 and T2 | 30 | 90 |
| S3 | Stable Bare | Bare (exposed bed) at T1 and T2 | 40 | 120 |
| S4 | Other land | Non-water at T1 and T2, not S2/S3 (includes Bare↔Other) | 30 | 90 |
| S5 | Stable Shoreline | Same class at T1 and T2, but within 1 px of a water edge on either date (takes priority over S1–S4) | 100 | 300 |
| S6 | Water loss | Water → any non-water | 125 | 375 |
| S7 | Water gain | Any non-water → Water | 125 | 375 |
| | **Total** | | **500** | **1,500** |

- **Water-change strata (S6–S7):** 750 points. With the shoreline (S5), 1,050 points sit on hard or changing water.
- **Hard pixels for E5:** S5 plus the shoreline points that fall inside S6/S7, plus any point the interpreter marks as muddy, weedy or mixed water.
- **Year pairs:** points are drawn from all (pixel, year pair) combinations pooled within a stratum. Year pairs with more change therefore get more points. The pilot showed why this matters: loss points drawn only from wet years were all rule errors.
- **Small strata:** if a stratum has too few candidate pixels at a site, take every candidate and move the shortfall to that site's S5. Known case: Pillur S3 has about 28 spaced candidates. Record the final counts.
- **Replaces** v1 (S1–S4 50 each, S5 50, S6/S7 100, S8 50) and the original equal 166/167/167 split in `base_architecture.md`.

### Why 750 water-change points (statistical power)

McNemar's test, two-sided, α = 0.05 / 6 (worst case under Holm correction), power 0.8. *ψ* is the share of points where the two models disagree; *δ* is the difference in accuracy between them.

| ψ | δ | True-change points needed |
|---|---|---|
| 0.20 | 0.10 | 239 |
| 0.20 | 0.07 | 491 |
| 0.30 | 0.10 | 360 |
| 0.30 | 0.07 | 738 |

The pilot confirmed 8 of 8 gain points; loss points need pooled pairs (§8). If about 70% of the 750 water-change points are real change (about 525), the test can detect a 7-point difference when the models disagree on about 20% of points. Adding the 300 shoreline points gives about 825 hard points for E5.

## 3. Spatial rules

- **Study area:** each reservoir's maximum water extent (water frequency 2015–2026), buffered by 2 km (configurable). The KMLs are only download rectangles.
- **Snap to pixel centers** on the 30 m EPSG:32643 grid. This removes ambiguity in point position.
- **Minimum spacing:** 90 m (3 px) between points of the same year pair.
- **Spatial blocks:** each point is tagged with a 1 km × 1 km block ID, used by the block bootstrap.
- **Mixed pixels are flagged, not excluded.** The original plan's 15 m buffer rule would remove shoreline pixels, which are exactly the hard cases E5 needs. Instead, the interpreter marks a `mixed` flag, and results are reported with and without mixed pixels.

## 4. Labeling procedure

1. **Reference imagery, in order of preference:**
   1. Google Earth Pro historical imagery within ±15 days of the pixel's source date (from the medoid date layer).
   2. Sentinel-2 imagery (10 m, available from 2016) on the same date or the closest one.
   3. The Landsat composite plus its time series, TimeSync-style (Cohen et al. 2010).

   Record which source was used. Google Earth coverage of rural Tamil Nadu in January–April may be sparse for some years, so check this in the pilot.
2. **Blinding:** the interpreter sees no model output and no stratum, and points are shown in random order.
3. **Fields recorded per point:** see §5.
4. **Quality control:** a second interpreter independently labels a random 10% (150 points). Report Cohen's κ. Disagreements are resolved by discussion and logged.
5. **Effort estimate:** about 1–2 minutes per point → **25–50 hours** for 1,500 points, plus about 5 hours for the second-interpreter subset.

## 5. Labeling sheet

The labeler fills in `<SITE>_labels.csv`, which has no stratum column (blinding). The sampling script keeps a separate **key file** (`<SITE>_key.csv`: `point_id` → stratum, x, y, block, year pair). After labeling, the two are merged into `data/ground_truth/validation_points.gpkg`. Step-by-step labeling instructions are in `LABELING_GUIDE.md`.

Merged schema:

| Field | Type | Notes |
|---|---|---|
| `point_id` | str | e.g. `BHA-P-0042`. Random order; the ID never reveals the stratum. |
| `site` | str | Pillur / Bhavanisagar / Mettur |
| `x`, `y` | float | EPSG:32643 pixel center |
| `lon`, `lat` | float | WGS84, for Google Earth |
| `stratum` | str | S1–S8 (key file only) |
| `year_t1`, `year_t2` | int | |
| `date_t1`, `date_t2` | date | Landsat source date of this pixel |
| `label_t1`, `label_t2` | str | Water / Forest / Bare / Other |
| `ref_source_t1`, `ref_source_t2` | str | GE / S2 / Landsat |
| `ref_date_t1`, `ref_date_t2` | date | Date of the reference image used |
| `confidence_t1`, `confidence_t2` | int | 0 = no imagery, 1 = guess, 2 = probable, 3 = clear |
| `mixed_t1`, `mixed_t2` | bool | No class covers more than half of the pixel |
| `interpreter` | str | |
| `minutes` | float | Time spent on the point |
| `block_id` | str | 1 km block, for the bootstrap |
| `notes` | str | |

## 6. Workflow

1. Build the automatic labels for all sites/years: `.venv/bin/python scripts/build_labels.py` (writes `data/interim/<Site>_labels.zarr`).
2. Draw the full v2 sample (seed, allocation, spacing and pilot-exclusion radius are in `configs/validation.yaml`; read-only on the label stores, writes only `data/ground_truth/full/`):
   `.venv/bin/python scripts/draw_validation_sample.py`
   This writes per site `<SITE>/<SITE>_points.kml`, the blind `<SITE>_labels.csv`, the `<SITE>_key.csv` (strata + automatic labels; the labeler must not open it), `strata_weights.csv` (pixel-pair population per site/stratum/pair, for the Olofsson estimator), `draw_summary.json`, and the second-interpreter subset (`second_interpreter/<SITE>_labels_2nd.csv` + KML, 50 per site).
3. Build the labeler helpers, one site at a time (the chips make about 1,000 Planetary Computer reads per site; resumable via `reference_chips/<SITE>_manifest.json`, rerun the same command after an interruption):
   ```
   for c in BHA MET PIL; do
     .venv/bin/python scripts/make_copernicus_links.py data/ground_truth/full/$c/${c}_labels.csv
     .venv/bin/python scripts/make_reference_chips.py  data/ground_truth/full/$c/${c}_labels.csv
   done
   ```
4. Label the points (§4), then run the second-interpreter check: `.venv/bin/python scripts/interpreter_agreement.py --version v1` (κ for water/non-water, 4 classes and water transition; writes `second_interpreter/adjudication.csv`). For v1 (2026-10-24) the disagreements were **not** adjudicated: both interpretations are kept, KC is the reference and the κ is reported as measured.
5. Freeze the dataset: version and hash it before any model is scored (`.venv/bin/python scripts/freeze_validation.py --version v1` → `data/ground_truth/frozen/v1/`, tracked manifest `validation_freeze/v1.json`; refuses to overwrite, corrections go into a new version). Never relabel after seeing model results. Load-time exclusion (`configs/chips.yaml`, `data/ground_truth/**/*_key.csv`, 150 m) automatically covers both the pilot and the full keys.

## 7. Pilot (before full-scale labeling)

Take 50 points at Bhavanisagar, spread across all strata, using 2–3 year pairs. Use them to check:
- how often Google Earth has imagery within ±15 days,
- time per point,
- whether the strata contain what they claim (for example, the share of S6/S7 that is real change),
- interpreter confidence on shoreline pixels.

Adjust the allocation and rules based on what the pilot shows.

## 8. Pilot results (2026-10-13, 50 Bhavanisagar points, pairs 2017→18, 2018→19, 2024→25)

The pilot used the v1 strata and the older per-year forest rule. `scripts/pilot_qc.py` repeats the checks below.

- **Effort:** 2.3 min per point on average (3.2 for the first 10, 2.0 for the last 10), so the full set takes about 55–60 h.
- **Imagery:** Google Earth was usable within ±15 days for 19% of dates; Sentinel-2 for the rest (via `reference_chips/`). Confidence: 3 on 39%, 2 on 55%, 1 on 6%. Mixed pixels: 33% of dates.
- **Water is reliable.**
  - 29 of 30 rule-Water labels were Water to the interpreter.
  - All 8 water-gain (S7) points were real gains.
  - 9 rule-non-water labels were Water to the interpreter (shorelines, muddy edges, river banks): the rule misses water there.
  - All 8 water-loss (S6) points were rule errors. They came from wet-year pairs with almost no real loss, a sampling artefact fixed by pooling all 11 pairs.
- **Land classes are not reliable from 10 m imagery.**
  - The interpreter labeled Forest on 26 pixels the rule called Other. They were described as scattered trees or bushes, which the guide calls Other.
  - Bare soil away from the reservoir was labeled Bare 8 times, while the rule calls it Other.
  - Result: water became the primary endpoint, the land classes secondary (GE-only); the class rules were clarified in `LABELING_GUIDE.md` §9.
- **Sheet issues:** 2 reference dates outside ±15 days; 1 mixed pixel with confidence 3. The blanket "S2 ≤ 2" confidence cap was replaced by "3 only if the square is uniform".
- The pilot points are **not** part of the 1,500. They are still excluded from training.

## References

- Olofsson et al. (2014). Good practices for estimating area and assessing accuracy of land change. *Remote Sensing of Environment* 148, 42–57.
- Cohen et al. (2010). TimeSync: tools for calibration and validation. *Remote Sensing of Environment* 114, 2911–2924.
- Foody (2004). Thematic map comparison. *PE&RS* 70(5), 627–633.
- Connor (1987). Sample size for testing differences in proportions for the paired-sample design. *Biometrics* 43, 207–211.
