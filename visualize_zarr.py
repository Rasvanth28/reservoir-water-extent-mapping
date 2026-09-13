import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

ZARR_DIR = "data"
OUTPUT_DIR = "visualizations"

def apply_stretch(array, p_low=2, p_high=98):
    """Applies a percentile contrast stretch to normalize pixels to [0, 1]."""
    valid_pixels = array[~np.isnan(array)]
    if len(valid_pixels) == 0:
        return np.zeros_like(array)
    
    p2, p98 = np.percentile(valid_pixels, (p_low, p_high))
    if p98 == p2:
        return np.zeros_like(array)
        
    stretched = np.clip((array - p2) / (p98 - p2), 0, 1)
    return stretched

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    if not os.path.exists(ZARR_DIR):
        print(f"Data directory {ZARR_DIR} does not exist.")
        return

    zarr_stores = [f for f in os.listdir(ZARR_DIR) if f.endswith(".zarr")]
    if not zarr_stores:
        print(f"No Zarr stores found in {ZARR_DIR}/")
        return
        
    for store in tqdm(zarr_stores, desc="Processing Reservoirs"):
        reservoir_name = store.replace("_datacube.zarr", "")
        store_path = os.path.join(ZARR_DIR, store)
        
        # Load Zarr directly to bypass xarray metadata dimension constraints
        try:
            import zarr
            store_zarr = zarr.open(store_path, mode="r")
            if "landsat_data" not in store_zarr:
                print(f"\nNo 'landsat_data' array in {store}")
                continue
            data_arr = store_zarr["landsat_data"]
        except Exception as e:
            print(f"\nFailed to open {store_path}: {e}")
            continue
            
        if len(data_arr.shape) != 4 or data_arr.shape[0] == 0:
            print(f"\nInvalid shape {data_arr.shape} in {store}")
            continue
            
        # Get first and last indices (time axis is 0)
        t_first = 0
        t_last = data_arr.shape[0] - 1
        indices = [t_first, t_last]
        labels = ["First", "Last"]
        
        # TARGET_BANDS: ["blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel"]
        # Indexes: 0=Blue, 1=Green, 2=Red, 3=NIR
        
        for idx, label in zip(indices, labels):
            try:
                # Shape is [band, y, x]
                img_data = data_arr[idx, :, :, :]
            except IndexError:
                continue
                
            time_val = f"Index_{idx}"  # Time metadata was lost in salvage, using index as proxy
            
            # Extract specific bands (axis 0 is band in img_data)
            red = img_data[2, :, :]
            green = img_data[1, :, :]
            blue = img_data[0, :, :]
            nir = img_data[3, :, :]
            
            # Apply Landsat C2 scaling (DN * 0.0000275 - 0.2)
            red = (red * 0.0000275) - 0.2
            green = (green * 0.0000275) - 0.2
            blue = (blue * 0.0000275) - 0.2
            nir = (nir * 0.0000275) - 0.2
            
            # Stack for True Color (RGB)
            true_color = np.dstack((red, green, blue))
            true_color_stretched = np.zeros_like(true_color)
            for b in range(3):
                true_color_stretched[:, :, b] = apply_stretch(true_color[:, :, b])
                
            # Stack for False Color (NIR, R, G)
            false_color = np.dstack((nir, red, green))
            false_color_stretched = np.zeros_like(false_color)
            for b in range(3):
                false_color_stretched[:, :, b] = apply_stretch(false_color[:, :, b])
                
            # Plot and Save True Color
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(true_color_stretched)
            ax.axis('off')
            ax.set_title(f"{reservoir_name} - True Color (RGB) - {time_val}", fontsize=14)
            tc_out = os.path.join(OUTPUT_DIR, f"{reservoir_name}_{label}_TrueColor.png")
            plt.savefig(tc_out, bbox_inches='tight', dpi=300)
            plt.close()
            
            # Plot and Save False Color
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(false_color_stretched)
            ax.axis('off')
            ax.set_title(f"{reservoir_name} - False Color (NIR-R-G) - {time_val}", fontsize=14)
            fc_out = os.path.join(OUTPUT_DIR, f"{reservoir_name}_{label}_FalseColor.png")
            plt.savefig(fc_out, bbox_inches='tight', dpi=300)
            plt.close()

if __name__ == "__main__":
    main()
