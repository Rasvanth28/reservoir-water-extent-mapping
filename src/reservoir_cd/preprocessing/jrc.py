"""JRC Global Surface Water 'extent' layer (ever-water) on the cube grid, for an independent
extent comparison. Cached as GeoTIFF."""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from reservoir_cd.grid import Grid


def fetch_jrc_extent(grid: Grid, bbox: list[float], cache_dir: str | Path, site: str) -> np.ndarray:
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{site}_jrc_extent_30m.tif"
    if path.exists():
        with rasterio.open(path) as src:
            return src.read(1)
    import planetary_computer as pc
    import pystac_client
    from reservoir_cd.preprocessing.esri import STAC_URL
    cat = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    items = list(cat.search(collections=["jrc-gsw"], bbox=bbox).items())
    j = np.zeros(grid.shape, np.uint8)
    for it in items:
        with rasterio.open(it.assets["extent"].href) as src:
            with WarpedVRT(src, crs=grid.crs, transform=grid.transform, width=grid.width,
                           height=grid.height, resampling=Resampling.nearest) as vrt:
                j = np.maximum(j, vrt.read(1).astype(np.uint8))
    cache_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=grid.height, width=grid.width, count=1,
                       dtype="uint8", crs=grid.crs, transform=grid.transform, compress="deflate") as dst:
        dst.write(j, 1)
    return j


def compare_extent(j: np.ndarray, ext: np.ndarray, study: np.ndarray, res: float) -> dict:
    jw = (j == 1) & study
    inter = (jw & ext).sum()
    return {"jrc_extent_km2_in_study": float(jw.sum() * res ** 2 / 1e6),
            "frac_of_our_extent_in_jrc": float(inter / max(ext.sum(), 1)),
            "frac_of_jrc_in_our_extent": float(inter / max(jw.sum(), 1)),
            "iou": float(inter / max((jw | ext).sum(), 1))}
