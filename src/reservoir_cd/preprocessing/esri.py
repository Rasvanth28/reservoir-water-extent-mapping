"""Esri 10 m Annual LULC (``io-lulc-annual-v02``) resampled to the 30 m cube grid.

Class codes (verified from the STAC ``file:values`` of the collection):
1 Water, 2 Trees, 4 Flooded vegetation, 5 Crops, 7 Built area, 8 Bare ground,
9 Snow/ice, 10 Clouds, 11 Rangeland, 0 No data.
"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from reservoir_cd.grid import Grid

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "io-lulc-annual-v02"
UNLABELED = 255


def _cache_path(cache_dir: Path, site: str, year: int) -> Path:
    return Path(cache_dir) / f"{site}_esri_{year}_30m.tif"


def fetch_esri_year(year: int, grid: Grid, bbox_lonlat: list[float], cache_dir: str | Path,
                    site: str) -> np.ndarray:
    """Raw Esri class codes (uint8) mode-resampled to ``grid``; cached as GeoTIFF."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, site, year)
    if path.exists():
        with rasterio.open(path) as src:
            return src.read(1)

    import planetary_computer as pc
    import pystac_client

    cat = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    items = list(cat.search(collections=[COLLECTION], bbox=bbox_lonlat,
                            datetime=f"{year}-01-01/{year}-12-31").items())
    if not items:
        raise RuntimeError(f"No Esri LULC item for {year}")
    out = np.zeros(grid.shape, dtype=np.uint8)
    for item in items:  # mosaic if the bbox spans several tiles (nodata=0 is overwritten)
        href = item.assets["data"].href
        with rasterio.open(href) as src:
            with WarpedVRT(src, crs=grid.crs, transform=grid.transform, width=grid.width,
                           height=grid.height, resampling=Resampling.mode,
                           src_nodata=0, nodata=0) as vrt:
                arr = vrt.read(1)
        out = np.where(out == 0, arr, out)
    with rasterio.open(path, "w", driver="GTiff", height=grid.height, width=grid.width, count=1,
                       dtype="uint8", crs=grid.crs, transform=grid.transform, nodata=0,
                       compress="deflate") as dst:
        dst.write(out, 1)
    return out


def map_esri(raw: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Map raw Esri codes to project classes (unmapped / nodata -> 255)."""
    lut = np.full(256, UNLABELED, dtype=np.uint8)
    for k, v in mapping.items():
        lut[int(k)] = int(v)
    return lut[raw]
