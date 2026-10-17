"""Raster grid helper for the 30 m EPSG:32643 cube grid."""
from dataclasses import dataclass

import numpy as np
from rasterio.transform import Affine


@dataclass(frozen=True)
class Grid:
    """North-up grid. ``x0``/``y0`` are the TOP-LEFT CORNER of the first pixel.

    stackstac (default ``xy_coords="topleft"``) stores corner coordinates in the
    cube's ``x``/``y`` coords, so pixel centers are offset by half a pixel.
    """

    x0: float
    y0: float
    res: float
    height: int
    width: int
    crs: str = "EPSG:32643"

    @classmethod
    def from_cube_coords(cls, x: np.ndarray, y: np.ndarray, crs: str = "EPSG:32643") -> "Grid":
        res = float(x[1] - x[0])
        return cls(float(x[0]), float(y[0]), res, len(y), len(x), crs)

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0, self.x0, 0, -self.res, self.y0)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def xc(self) -> np.ndarray:
        return self.x0 + self.res * (np.arange(self.width) + 0.5)

    @property
    def yc(self) -> np.ndarray:
        return self.y0 - self.res * (np.arange(self.height) + 0.5)

    def center(self, row: int, col: int) -> tuple[float, float]:
        return self.x0 + self.res * (col + 0.5), self.y0 - self.res * (row + 0.5)


def grid_bbox_lonlat(grid: "Grid", pad_px: int = 0) -> list[float]:
    """[min_lon, min_lat, max_lon, max_lat] of the grid extent (for STAC searches)."""
    from pyproj import Transformer
    tr = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    x0, x1 = grid.x0 - pad_px * grid.res, grid.x0 + (grid.width + pad_px) * grid.res
    y1, y0 = grid.y0 + pad_px * grid.res, grid.y0 - (grid.height + pad_px) * grid.res
    pts = [tr.transform(x, y) for x in (x0, x1) for y in (y0, y1)]
    lons, lats = zip(*pts)
    return [min(lons), min(lats), max(lons), max(lats)]
