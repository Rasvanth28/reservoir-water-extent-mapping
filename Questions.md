1. **How is this project going to help the government?**
**Answer:** This project automates the mapping of water extents over an 11-year timeframe (2015-2026). Currently, government Land Use/Land Cover (LULC) maps are static and released years apart, failing to capture high-frequency changes in critical hydrological infrastructure. Our dynamic pipeline provides continuous, high-fidelity data that can aid the government in real-time water resource management, drought monitoring, flood risk assessment, and proactive policy formulation.

2. **Why Specifically Pillur, Mettur, Bhavani Sagar Dam?**
**Answer:** These three dams represent critical hydrological infrastructure within Tamil Nadu (specifically the Cauvery and Bhavani river basins). They exhibit distinct morphological, climatic, and storage characteristics, making them ideal varied case studies to test the robustness of the foundation model across different geographic and topological settings.

3. **What is the Zarr File and why we are using it?**
**Answer:** Zarr is a cloud-optimized, chunked, and compressed N-dimensional array storage format. We use it instead of traditional GeoTIFFs because it natively supports chunking and lazy loading through `xarray` and `Dask`. This allows us to handle massive geospatial datasets (like a decade of multi-band Landsat imagery) efficiently without overwhelming local RAM, and it allows us to bypass Planetary Computer's SAS token expiration timeouts by appending data in bi-annual chunks.

4. **Why specifically prithvi model is used?**
**Answer:** Prithvi-EO-2.0 is a Geospatial Foundation Model specifically pre-trained on massive amounts of multi-spectral Earth Observation data (like Landsat and Sentinel). Unlike standard computer vision models (like ResNet) which only understand RGB images, Prithvi natively understands the 6-band optical/infrared spectral signatures (Blue, Green, Red, NIR, SWIR1, SWIR2) and the spatial-temporal dynamics unique to satellite imagery.

5. **What are we going to do when there is missing of data? like bhuvan only has few years not every years.**
**Answer:** Bhuvan LULC data is only being used as a static topographical basin mask (to prevent false positives outside the reservoir basin) and as "weak supervision" during training. Our primary, dynamic ground-truth generation actually relies on the continuous Landsat time-series itself, using MNDWI indexing + Otsu's automatic thresholding. Furthermore, we rely on human-verified historical imagery from Google Earth (our N=1,500 validation points) to serve as a robust, independent ground truth, entirely bypassing Bhuvan's temporal gaps.

6. **What are we actually changing in the existing prithvi architecture?**
**Answer:** We are implementing Parameter-Efficient Fine-Tuning using Low-Rank Adaptation (LoRA). Instead of updating all the millions of parameters, we freeze the Prithvi backbone and only inject trainable rank decomposition matrices into its transformer attention blocks. Additionally, we are modifying the UPerNet decoder head to predict a multi-class semantic transition matrix (change detection) by concatenating the multi-scale feature maps from two different times ($T_1$ and $T_2$).

7. **Why specifically these march? Are we having any plan on doing it in other months?**
**Answer:** Our initial plan targeted anniversary dates in March because March represents the pre-monsoon dry season. This provides cloud-free imagery and represents the base storage level of the reservoirs, making structural changes easier to identify. **However**, as noted in our recent progress updates, we have already expanded our scope to pull a **dense multi-year time series from 2015 to 2026** (covering all months), vastly increasing the data available for the model to learn temporal dynamics across both wet and dry seasons.

8. **What are the metrics and indexes are we using? Like NDVI , those indexes?**
**Answer:** For dynamic mask generation, we are primarily using **MNDWI** (Modified Normalized Difference Water Index) combined with Otsu's automatic thresholding. Unlike NDVI (which is used for vegetation), MNDWI uses Green and SWIR bands to effectively suppress built-up land noise while strongly highlighting open water. We also use the `QA_PIXEL` bitmask band to filter out clouds and cloud shadows.