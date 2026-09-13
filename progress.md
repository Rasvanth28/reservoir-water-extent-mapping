# Project Progress & Architecture Alignment

This document tracks our current progress against the pipeline defined in `base_architecture.md`, highlighting what has been strictly followed and where we have made strategic architectural adaptations.

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
**Status:** `NOT STARTED`
*   No ISRO Bhuvan weak supervision maps have been ingested yet.
*   The N=1,500 validation points have not been digitized or formatted yet.

---

## 3. HLS-Aligned Tensor Patch Generation
**Status:** `NOT STARTED (Next Immediate Phase)`
*   We have not yet built the `PrithviReservoirDataset` PyTorch class to chip the Zarr stores into $256 \times 256$ spatial windows.
*   We need to ensure the $T=2$ sequences are generated from our expanded 2015-2026 temporal stacks.

---

## Steps 4, 5, 6, and 7
**Status:** `NOT STARTED`
*   Foundation model benchmarking, LoRA fine-tuning, McNemar's testing, and Open Science packaging remain completely untouched.
