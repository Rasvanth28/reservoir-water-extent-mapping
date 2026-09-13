# Geospatial Foundation Model Pipeline Architecture

## 1. Geospatial Data Engineering & Hydrological Contextualization
*Landsat extraction and climatic baselining*

Download anniversary-date (March 2015 and March 2023) Landsat 8/9 Level-2 Surface Reflectance imagery for Pillur Dam, Bhavanisagar, and Mettur. Apply bitwise unpacking on the 16-bit QA_PIXEL band combined with a 3x3 morphological opening to mask out clouds and shadows (NaN). Simultaneously, compile IMD gridded rainfall and CWC storage data for the preceding 12 months into a descriptive tabular framework to contextualize whether detected changes are structural or climate-driven.

## 2. Ground Truth & Labeling Protocol
*Breaking circularity with strict spatial rules*

Reproject and resample ISRO Bhuvan LULC maps to a 30m grid to serve strictly as weak supervision for training. For testing, manually digitize an independent validation set of N=1,500 points (500 per site) using historical Google Earth Pro imagery (±15 days). Enforce equal allocation stratification (166 Water, 167 Forest, 167 Bare Land) and a 15-meter spatial buffer to avoid pixel bleed. Assign paired temporal labels ($C_{T1}$, $C_{T2}$) to each point to establish ground-truth transition classes.

## 3. HLS-Aligned Tensor Patch Generation
*Preparing inputs for the foundation model*

Stack the masked Landsat arrays into the exact 6-band configuration required by Prithvi (Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2). Slide a spatial window across the scenes to chip them into 256x256 PyTorch training tensors, ensuring the 2015 and 2023 sequences are spatially identical for the $T=2$ temporal ingestion.

## 4. Architectural Setup & Baselines
*Configuring the 4-way benchmark*

Initialize the four benchmark pipelines:
*   **Baseline A:** MNDWI + Otsu Automatic Thresholding.
*   **Baseline B:** FC-Siam-diff (trained from scratch).
*   **Baseline C:** Frozen Prithvi-EO-2.0 backbone with a UPerNet decoder initialized from OSCD weights (resampled to 30m), reinitializing the final output head for multi-class transitions.
*   **Baseline D (Ours):** Prithvi-EO-2.0 backbone. Concatenate $T_1$ and $T_2$ multi-scale feature maps ($F_{concat} = [F_{T1}, F_{T2}]$) and feed them into a UPerNet decoder to predict a multi-class semantic transition matrix.

## 5. Parameter-Efficient Fine-Tuning
*LoRA implementation*

Train Baseline D using Low-Rank Adaptation (LoRA). Inject trainable rank decomposition matrices (rank $r = 8$, alpha $\alpha = 16$) exclusively into the transformer attention blocks of the Prithvi backbone. Train the network using a Masked Cross-Entropy loss function (ignoring NaN cloud pixels) against the weak Bhuvan labels to prevent overfitting on the regional dataset.

## 6. Statistical Evaluation & Matrix Generation
*Conditioned testing and spatial correction*

Flatten the final predictions and evaluate against the N=1,500 human-verified points. Generate a comprehensive multi-class Confusion Matrix to report performance on rare trajectories. Run McNemar's Test in two passes: once on overall agreement, and once conditioned specifically on true-change validation points. Apply a spatial block bootstrap to correct for spatial autocorrelation, and use a Holm-Bonferroni correction across the six pairwise McNemar tests to control the family-wise error rate.

## 7. Open Science Packaging
*Preparing for publication*

Clean the PyTorch training loops, evaluation scripts, and statistical R/Python notebooks. Upload the N=1,500 validation dataset, the adapted Prithvi-LoRA weights, and the code repository to Zenodo and GitHub to satisfy the reproducibility requirements for ISPRS/TGRS submission.
