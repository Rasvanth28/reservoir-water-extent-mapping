"""Study-area mask from a reservoir KML polygon."""
import re
from pathlib import Path

import numpy as np
from pyproj import Transformer
from rasterio import features
from shapely.geometry import Polygon
from shapely.ops import transform as shp_transform

from reservoir_cd.grid import Grid


def read_kml_polygon(path: str | Path) -> Polygon:
    """First Polygon outer ring in a KML as a lon/lat shapely Polygon."""
    # Regex instead of an XML parser: bhavanisagar.kml has a corrupted XML declaration
    # ('<?xml version="1.0" enc<?xml ...'), which strict parsers reject.
    text = Path(path).read_text()
    m = re.search(r"<coordinates>(.*?)</coordinates>", text, re.S)
    pts = [tuple(float(v) for v in tok.split(",")[:2]) for tok in m.group(1).split()]
    return Polygon(pts)


def basin_mask(kml_path: str | Path, grid: Grid, buffer_m: float = 2000.0) -> np.ndarray:
    """Boolean mask of pixels inside the KML polygon buffered by ``buffer_m`` (in grid CRS)."""
    poly = read_kml_polygon(kml_path)
    tr = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True).transform
    poly_utm = shp_transform(tr, poly).buffer(buffer_m)
    return features.rasterize(
        [(poly_utm, 1)], out_shape=grid.shape, transform=grid.transform,
        fill=0, dtype="uint8", all_touched=False,
    ).astype(bool)
