import os
import shutil
import zarr
import numpy as np

output_dir = "data"

for reservoir in ["Pillur", "Bhavanisagar", "Mettur"]:
    zarr_path = os.path.join(output_dir, f"{reservoir}_datacube.zarr")
    if not os.path.exists(zarr_path):
        continue
        
    print(f"\nSalvaging {reservoir}...")
    
    dirs = [d for d in os.listdir(zarr_path) if d.startswith("stackstac-")]
    if not dirs:
        print(f"No stackstac arrays found for {reservoir}. Maybe already fixed?")
        continue
        
    # Sort by modification time to reconstruct chronological order
    dirs.sort(key=lambda x: os.path.getmtime(os.path.join(zarr_path, x)))
    
    store = zarr.open(zarr_path, mode="a")
    
    first_arr = store[dirs[0]]
    
    # Create a unified landsat_data array
    if "landsat_data" in store:
        del store["landsat_data"]
        
    dest = store.create("landsat_data", shape=first_arr.shape, chunks=first_arr.chunks, dtype=first_arr.dtype, overwrite=True)
    dest[:] = first_arr[:]
    
    # Iteratively append chunks to save RAM
    for d in dirs[1:]:
        arr = store[d]
        dest.append(arr[:], axis=0)
        
    print(f"Merged array shape: {dest.shape}")
    
    # Tell xarray how to map the dimensions
    dest.attrs["_ARRAY_DIMENSIONS"] = ["time", "band", "y", "x"]
    
    # Clean up the old hashed arrays
    for d in dirs:
        del store[d]
        
    # Re-consolidate metadata for xarray
    zarr.consolidate_metadata(zarr_path)
    
    print(f"{reservoir} successfully salvaged!")
