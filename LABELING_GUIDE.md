# Labeling Guide: Validation Points

A step-by-step manual for labeling the validation points in Google Earth Pro.
The design behind it (strata, counts, statistics) is in `validation_protocol.md`. You don't need to read that file to label.

---

## 0. Ground rules (read once, keep in mind)

1. **Label what you see, not what you expect.** Don't guess from neighboring points, earlier years or "what the reservoir usually looks like."
2. **Never look at model outputs or label maps while labeling.** You also won't be told why a point was chosen. That's deliberate (blinding).
3. **Each point is one 30 m × 30 m Landsat pixel.** Label the class that covers **more than half of that square**, not just the pin location.
4. **Label two dates per point:** T1 (earlier year) and T2 (later year). Label them independently.
5. **Once the dataset is frozen, never change a label**, even if a model result later makes you doubt it. Note doubts in `notes` *before* freezing.

---

## 1. One-time setup (about 30 minutes)

1. **Install Google Earth Pro** (desktop, free): https://www.google.com/earth/about/versions/ → *Earth Pro on desktop*. The web version has no historical imagery.
2. Open Google Earth Pro and set:
   - **Tools → Options → 3D View → Show Lat/Long:** *Decimal Degrees*
   - **Tools → Options → 3D View → Units of Measurement:** *Meters, Kilometers*
   - **View → Historical Imagery:** turn on. A time slider appears at the top left.
   - **Layers panel:** turn off everything except *Borders and Labels* (fewer distractions).
3. Bookmark the fallback imagery source: **Copernicus Browser**, https://browser.dataspace.copernicus.eu (free account). It provides Sentinel-2 (10 m, from 2016) and Landsat 8/9.
4. Open the labeling workbook (`<SITE>_labels.xlsx`, made by `scripts/label_workbook.py build`) in Excel or Numbers. Keep it next to Google Earth. It has dropdowns, and typing `S2` or `LS` in `ref_source` fills `ref_date` with the reference chip's image date (grey `chip_*` columns). For `GE`, type the date over the formula. When done, save as .xlsx (Numbers: File → Export To → Excel) and run `scripts/label_workbook.py export <SITE>_labels.xlsx` to get `<SITE>_labels_filled.csv`.
5. **Optional: keyboard labeling app.** Instead of typing into the workbook, run `.venv/bin/python scripts/label_app.py data/ground_truth/full/<SITE>/<SITE>_labels.xlsx --interpreter <INITIALS>` (close the workbook in Excel first). It opens a browser page with the T1 and T2 chips side by side, starts at the first incomplete point, saves each point to the workbook as you go (backup in `backups/`) and refreshes `<SITE>_labels_filled.csv`. `ref_source`, `ref_date`, `interpreter` and `minutes` are filled automatically. A banner on each date says whether Google Earth has an image within ±15 days (green: which date; grey: none, label from the chip), from `scripts/fetch_ge_dates.py` (Google Earth's historical image dates, via the GEHistoricalImagery tool in `tools/`). Open Google Earth only on green. Keys apply to the highlighted date:

| Key | Action |
|---|---|
| `W` `F` `B` `O` | Water / Forest / Bare / Other |
| `N` | NA (confidence 0, date complete) |
| `1` `2` `3` | Confidence; completes the date (T1 then goes to T2, T2 saves and goes to the next point) |
| `M` | Toggle mixed (a second class is visible in the square); then press the second class (recorded in notes as `T1 mixed Forest/Water`) |
| `G` | Fly Google Earth Pro to the point; source becomes GE with the GE image date prefilled. Set the time slider to that date, check it, `Enter` |
| `S` | Back to the chip as reference source |
| `/` | Type a note, `Enter` to return |
| `Tab` | Switch T1/T2 |
| `←` `→` | Previous / next point (saves what is filled) |
| `U` | Next unlabeled point |
| `Backspace` | Clear the active date |

---

## 2. What you receive per batch

| File | What it is |
|---|---|
| `<SITE>_points.kml` | Each point is a pin **and** a 30 m square showing the exact pixel. The pin name is the `point_id`; clicking it shows `date_t1` and `date_t2`. |
| `<SITE>_labels.csv` | The sheet you fill in. One row per point, in the same order as the KML. |

You will **not** see the stratum, the model predictions or the automatic labels.

---

## 3. Labeling one point, step by step

**Before you start a point**, note the time (for the `minutes` column).

### Step 1: Go to the point
- In the **Places** panel, double-click the point (e.g. `BHA-P-0007`). Google Earth flies to it.
- Zoom until the **30 m square fills about 1/4 of the screen** (eye altitude roughly 300–600 m).

### Step 2: Label T1
1. Read `date_t1` from the sheet (e.g. 2018-02-14).
2. Drag the **historical imagery slider** to the image closest to that date.
3. **Is there an image within ±15 days of `date_t1`?**
   - **Yes:** use it. Write its date in `ref_date_t1` and `GE` in `ref_source_t1`.
   - **No:** go to **§4 Fallback imagery**.
   - **The image is cloudy over the square:** treat it as "no image" and use the fallback.
4. Look at the 30 m square and pick the class covering more than half of it, using **§5 Class definitions**.
5. Fill in:
   - `label_t1`: Water / Forest / Bare / Other
   - `confidence_t1`: how sure you are that `label_t1` (the class covering more than half) is right: 3 = clearly visible, 2 = probably right, 1 = guess
   - `mixed_t1`: TRUE if a **second class is visibly present** inside the square (e.g. the square touches the shoreline, even 90/10); FALSE if the square is a single class. With TRUE, note the second class (and the split if you can, e.g. `Forest/Water 90/10`)

### Step 3: Label T2
Repeat Step 2 with `date_t2`, filling the `_t2` columns. **Don't copy T1** if the image looks similar; look again.

### Step 4: Finish the row
- `interpreter`: your initials
- `minutes`: time spent on this point
- `notes`: anything unusual (e.g. "floating weeds", "only 2017-05 image available, outside window", "cloud edge")

**Then move to the next point.** Save the sheet every 10 points.

---

## 4. Fallback imagery (when Google Earth has nothing within ±15 days)

**Easiest: the reference-chip gallery (no account needed).** Open `reference_chips/<SITE>_gallery.html` next to the sheet. It is made by `scripts/make_reference_chips.py` from Planetary Computer and is blind (built from the sheet only). For every point and date it shows the closest cloud-free Sentinel-2 image within ±15 days, with the 30 m square outlined in yellow: true colour on the left, SWIR false colour on the right (water is black). Copy the `ref_source` and `ref_date` it prints into the sheet. If it uses Landsat (`LS`), lower the confidence by one level.

**Alternative: Copernicus Browser** (needs a free account):

1. Open **Copernicus Browser** and search for the point's coordinates (`lat, lon` from the sheet).
2. Set the date range to `date ± 15 days`.
3. Choose **Sentinel-2 L2A** (2016 onward; not available for 2015) and set **max cloud cover** to about 30%.
4. Pick the clearest image, using **True color**, and also check **False color (urban)** or **SWIR**: water is very dark in SWIR.
5. At 10 m resolution, a 30 m pixel is about **3 × 3 Sentinel-2 pixels**. Label the class covering more than half of that 3 × 3 block.
6. Record `ref_source = S2` and the image date.
7. **If Sentinel-2 has no clear image either** (e.g. 2015): use **Landsat 8/9 L2** in the same browser. Record `ref_source = LS` and lower the confidence by one level, since you're judging at the same resolution as the model.
8. **If nothing usable exists:** set the label to `NA`, `confidence = 0`, and explain in `notes`. Don't guess.

---

## 5. Class definitions

| Class | Includes | Typical look (true color) |
|---|---|---|
| **Water** | Open water, **muddy/turbid water**, water with **floating weeds** if water is visible between them, shallow water over the reservoir bed | Dark blue/black, or brown and smooth (muddy); very dark in SWIR |
| **Forest** | Natural forest, **tree plantations** (teak, eucalyptus, coconut), dense riparian trees: **mostly continuous** tree crowns covering more than half of the square. Scattered trees or bushes with ground between them are **Other** | Dark green, textured canopy, visible crown shadows. **In Jan–Apr, deciduous forest may look brown or grey but still has tree crowns: label it Forest.** |
| **Bare** | **Exposed reservoir bed** (dry mud, cracked silt), sand bars, rock outcrops, quarries. **Not** bare soil away from the reservoir: that is Other (decided 2026-10-13) | Light brown, grey or white; smooth or cracked; no rows, no crowns |
| **Other** | **Farmland** (crops or ploughed fields, **including cultivation on the dry reservoir bed**), towns, villages, roads, buildings, **scrub and grassland**, **scattered trees or bushes on bare ground**, bare or fallow soil away from the reservoir, orchards with sparse trees | Field patterns and rows, regular shapes, roofs, mottled low vegetation |

### Tricky cases (decided rules, apply them consistently)

| Situation | Label |
|---|---|
| Dry reservoir bed with **crops or ploughed rows** | **Other** (farmland) |
| Dry reservoir bed, **bare or cracked**, no crops | **Bare** |
| Dry reservoir bed with **sparse grass** | **Bare** if more than half is visible soil; otherwise **Other** |
| Water **completely covered** by floating weeds (no water visible) | **Water**, with `notes: "full weed cover"`, `confidence` ≤ 2 |
| Wet mud right at the shoreline | **Bare** if no standing water is visible; otherwise **Water** |
| Hill or cloud **shadow** over the square | Label the class *under* the shadow if you can tell; otherwise use fallback imagery |
| Leafless deciduous trees (dry season) | **Forest** |
| Scattered trees over grass/scrub (canopy < 50%) | **Other** |
| A second class is visible in the square (e.g. 90/10 or 70/30) | Label the larger one, `mixed = TRUE`. Confidence 3 is allowed if the larger class is clearly the majority |
| Square split about evenly (≈ 50/50, you can't tell which is larger) | Pick the one you think is larger, `mixed = TRUE`, `confidence` ≤ 2 |

**New tricky case?** Write it in `notes` and add it to the **§9 Decision log** so later points are labeled the same way.

---

## 6. Second interpreter (quality check)

- A second person labels a **random 10%** of points (marked in a separate sheet they receive) **without seeing your labels**.
- They follow this same guide.
- Afterwards, the agreement (Cohen's κ) is computed. Disagreements are discussed and the agreed label is recorded, with both original labels kept in the log.

---

## 7. Pace and habits

- **Expect 1–2 minutes per point** once you're used to it (slower in the first 20).
- Work in **sessions of 45–60 minutes** with breaks. Accuracy drops when you're tired.
- At the start of each session, **re-read §5** (2 minutes). It keeps you consistent across weeks.
- **Back up the sheet** (e.g. a copy in Google Drive) after every session.

---

## 8. Quick reference card

```
1. Double-click point → zoom so the 30 m square fills ~1/4 of the screen
2. Slider to date_t1 (±15 days) → none or cloudy? → Copernicus S2 → Landsat → NA
3. Class covering >50% of the square: Water / Forest / Bare / Other
4. Fill label_t1, confidence_t1 (1–3), mixed_t1, ref_source_t1, ref_date_t1
5. Repeat for T2 (look again, don't copy)
6. interpreter, minutes, notes → save every 10 points
Tricky: crops on dry bed = Other · bare dry bed = Bare · muddy water = Water
        leafless continuous trees = Forest · scattered trees/scrub/grass = Other
        bare soil away from reservoir = Other · water first: be careful at shorelines
```

---

## 9. Decision log

Add any new rule here with the date, so every later point follows it.

| Date | Situation | Rule decided |
|---|---|---|
| 2026-10-10 | (initial rules in §5) | — |
| 2026-10-13 | Scattered trees or bushes with bare ground between them ("Forest + Bare with patches") | **Other**, not Forest. Forest only when crowns are mostly continuous. If you can't tell trees from scrub at 10 m: label Other, confidence ≤ 2, note `trees or scrub?` |
| 2026-10-13 | Bare soil **away from** the reservoir (not the dry bed, sand bar, rock or quarry) | **Other** |
| 2026-10-13 | Priority | **Water vs. non-water is the main result.** Spend your care there: shorelines, muddy or weedy water, wet mud. Still give a land class for non-water, but don't agonise over Forest vs. Other on 10 m images |
| 2026-10-13 | Confidence with Sentinel-2 | Confidence 3 is allowed with S2 when the whole square is uniform and unambiguous (e.g. open water, uniform dry bed). Otherwise S2 ≤ 2. (Pilot: replaces the blanket "S2 ≤ 2".) |
| 2026-10-12 | Google Earth's nearest clear image is outside ±15 days, and the in-window Sentinel-2 picture is ambiguous (e.g. dry-season brown) | Label from the in-window Sentinel-2 picture (`ref_source = S2`). The out-of-window GE image may be used to judge **structure** (tree crowns vs fields vs soil) if S2 shows nothing contradicting it (clearing, burn, water). Confidence ≤ 2. Note: `structure from GE <date>`. |
| 2026-10-19 | Mixed definition | `mixed = TRUE` whenever a **second class is visibly present** in the square, however small its share (as used for PIL). Considered a > 25% threshold and rejected it, to keep all three sites consistent without re-checking PIL. |
| 2026-10-19 | Confidence on mixed squares | Confidence is about the **main label** (the class covering more than half), not about both classes being present. Mixed squares may get confidence 3 when the main class is clearly the majority (e.g. 70/30), also with S2 (overrides the 2026-10-13 "uniform square" condition for this case). ≈ 50/50 stays ≤ 2. |
