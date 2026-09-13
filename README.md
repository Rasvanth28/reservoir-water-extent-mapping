# Automated Geospatial Pipeline for Reservoir Water Extent Mapping (2015–2026)

An end-to-end geospatial pipeline that fine-tunes the **Prithvi-EO-2.0** vision transformer foundation model to dynamically map water extent changes at three reservoirs in Tamil Nadu, India — **Pillur Dam**, **Bhavanisagar**, and **Mettur** — across a dense, multi-year Landsat 8/9 time series (2015–2026).

## Motivation

Government Land Use/Land Cover (LULC) maps are static and released years apart, which fails to capture the high-frequency temporal dynamics of critical hydrological infrastructure. This project builds an automated pipeline that produces a continuously updated, high-fidelity view of reservoir structural and climate-driven changes, aimed at supporting water resource management, drought monitoring, and flood risk assessment.

## Study Sites

| Reservoir     | Basin              |
|---------------|---------------------|
| Pillur Dam    | Bhavani river basin |
| Bhavanisagar  | Bhavani river basin |
| Mettur        | Cauvery river basin |

These three sites were chosen for their distinct morphological, climatic, and storage characteristics, providing varied test cases for the model's robustness.

## Methodology

1. **Geospatial Data Engineering** — Query Landsat 8/9 Level-2 Surface Reflectance imagery (Bands: Blue, Green, Red, NIR, SWIR1, SWIR2, plus the 16-bit `QA_PIXEL` cloud mask) from Microsoft's Planetary Computer STAC API, and consolidate the results into chunked Zarr data cubes.
2. **Dynamic Ground Truth** — Generate water masks via the Modified Normalized Difference Water Index (MNDWI) with automatic Otsu thresholding.
3. **Static Basin Masks** — Use ISRO Bhuvan LULC maps as topographical priors and weak supervision, constraining predictions to plausible reservoir boundaries.
4. **Independent Validation** — Evaluate against N=1,500 manually digitized points (500 per site), cross-verified against historical Google Earth Pro imagery (±15 days of the Landsat overpass), to avoid circular evaluation logic.
5. **Modeling** — Fine-tune the Prithvi-EO-2.0 backbone with Low-Rank Adaptation (LoRA), feeding concatenated multi-temporal feature maps into a UPerNet decoder to predict multi-class semantic water-extent transitions, benchmarked against an MNDWI+Otsu baseline and an FC-Siam-diff baseline trained from scratch.

See [`base_architecture.md`](base_architecture.md) for the full 7-step architecture and [`progress.md`](progress.md) for current status against that plan.

## Repository Structure

```
.
├── data_collection.py    # Queries Planetary Computer STAC API, builds per-reservoir Zarr data cubes
├── salvage_zarr.py        # Recovery utility for Zarr stores left in a fragmented/broken state
├── visualize_zarr.py      # Generates True/False Color composite PNGs for visual QA
├── *.kml                  # Reservoir boundary geometries (Bhuvan)
├── visualizations/        # Generated QA composite images
├── Reference Papers/      # Background literature referenced by the methodology
├── base_architecture.md   # Full 7-step pipeline architecture
├── progress.md            # Current progress vs. the architecture plan
├── Questions.md           # Design-decision rationale (Q&A format)
└── Technical_Defense_QA.md
```

`data/` (Zarr data cubes, ~3.7GB) and `Archive/` (superseded datasets/scripts, ~5.8GB) are generated/local-only and excluded from version control — see `.gitignore`.

## Setup

Requires Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install dask stackstac planetary-computer pystac-client xarray zarr rasterio numpy matplotlib tqdm
```

## Usage

```bash
# Download Landsat imagery and build Zarr data cubes for the configured reservoirs
python data_collection.py

# Repair a Zarr store left with fragmented arrays (only needed after an interrupted run)
python salvage_zarr.py

# Generate True/False Color QA composites from the Zarr stores
python visualize_zarr.py
```

## Current Status

Data engineering (Step 1) is largely complete for all three reservoirs, with a dense 2015–2026 time series downloaded (rather than the originally planned anniversary-date pairs). Ground truth labeling and tensor patch generation are the next phases. See [`progress.md`](progress.md) for the full breakdown of what's done, adapted, and pending.

## References

Background literature is collected in [`Reference Papers/`](Reference%20Papers), covering cloud-cover/patch-size considerations and the 30m resolution rationale used throughout this pipeline.
