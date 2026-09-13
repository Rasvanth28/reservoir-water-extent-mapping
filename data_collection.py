import os
import shutil
from typing import List, Optional
import pystac_client
import planetary_computer
import stackstac
import xarray as xr
from dask.diagnostics import ProgressBar

# --- STAC Configuration & Constants ---
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "landsat-c2-l2"

TARGET_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel"]

# Verified Reservoir Coordinates (EPSG:4326 [min_lon, min_lat, max_lon, max_lat])
RESERVOIR_BBOXES = {
    # "Pillur": [76.75, 11.20, 76.85, 11.30],
    "Bhavanisagar": [76.950, 11.36766, 77.179989, 11.53099],
    # "Mettur": [77.720, 11.750, 77.920, 11.980]
}

def query_and_stack_reservoir(
    reservoir_name: str, 
    bbox: List[float], 
    time_range: str,
    year: int
) -> Optional[xr.DataArray]:
    """Queries STAC for Landsat 8 & 9 L2 imagery and lazily loads the stack."""
    print(f"\n[{reservoir_name} | {time_range}] Querying STAC API...")
    
    catalog = pystac_client.Client.open(
        STAC_API_URL, 
        modifier=planetary_computer.sign_inplace
    )
    
    platforms = ["landsat-8"] if year < 2022 else ["landsat-8", "landsat-9"]
    
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=time_range,
        query={
            "eo:cloud_cover": {"lt": 20},
            "platform": {"in": platforms}
        }
    )
    items = search.item_collection()
    
    if len(items) == 0:
        print(f"[{reservoir_name} | {time_range}] No scenes found.")
        return None
        
    print(f"[{reservoir_name} | {time_range}] Found {len(items)} scenes (Landsat 8/9).")
    
    stack = stackstac.stack(
        items,
        assets=TARGET_BANDS,
        epsg=32643, 
        resolution=30,
        chunksize=256,
        bounds_latlon=bbox
    )
    
    return stack

def main():
    """Main execution block: downloads and saves reservoir data as Zarr stores."""
    output_dir = "data"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for name, bbox in RESERVOIR_BBOXES.items():
        print(f"\n{'='*60}\nProcessing Reservoir: {name}\n{'='*60}")
        zarr_path = os.path.join(output_dir, f"{name}_datacube.zarr")
        
        # In this updated script, we remove the existing store if it exists 
        # so we can cleanly pull the full 2015-2026 record. 
        if os.path.exists(zarr_path):
            print(f"[{name}] Removing existing Zarr store to ensure clean 2015-2026 build...")
            shutil.rmtree(zarr_path)
            
        is_first_chunk = True
        
        # ==============================================================================
        # FIX FOR SAS TOKEN TIMEOUT (The Mettur Issue)
        # ==============================================================================
        # The Issue: Planetary Computer SAS tokens strictly expire after 45-60 minutes. 
        # Because Mettur is a massive spatial area, downloading an entire year of data 
        # took longer than 45 minutes for Dask to compute, causing the token to expire 
        # mid-download and throwing a RasterioIOError (TIFFFillTile read error).
        #
        # The Fix: We chunk the temporal loop into smaller 6-month intervals (bi-annually). 
        # This guarantees each Dask block finishes well within the 45-minute SAS lifespan. 
        # We then append these 6-month blocks into the master Zarr store.
        # ==============================================================================
        
        for year in range(2015, 2027): # Extended up to 2026
            # Split into H1 (Jan-Jun) and H2 (Jul-Dec)
            half_years = [
                (f"{year}-01-01", f"{year}-06-30"),
                (f"{year}-07-01", f"{year}-12-31")
            ]
            
            for start_date, end_date in half_years:
                time_range = f"{start_date}/{end_date}"
                
                data_cube = query_and_stack_reservoir(name, bbox, time_range, year)
                
                if data_cube is None:
                    continue
                    
                # Enforce a consistent array name so xarray appends to the same variable
                data_cube.name = "landsat_data"
                
                # Strip non-dimensional coordinates to prevent Zarr append mismatch errors
                coords_to_drop = [c for c in data_cube.coords if c not in data_cube.dims]
                data_cube = data_cube.drop_vars(coords_to_drop)
                
                data_cube.attrs.clear()
                for coord in data_cube.coords:
                    data_cube.coords[coord].attrs.clear()
                
                # Download with visual progress bar
                with ProgressBar():
                    if is_first_chunk:
                        print(f"[{name} | {time_range}] Initializing Zarr store (Mode: WRITE)...")
                        data_cube.to_zarr(zarr_path, compute=True, mode="w")
                        is_first_chunk = False
                    else:
                        print(f"[{name} | {time_range}] Appending to Zarr store (Mode: APPEND)...")
                        data_cube.to_zarr(zarr_path, compute=True, mode="a", append_dim="time")
                
        print(f"[{name}] Successfully completed temporal chunking and saved data to disk.")
        
    print("\nData collection phase complete.")

if __name__ == "__main__":
    main()
