"""Pre-fetch Esri LULC (2017-2023) and JRC extent for every site into data/external/."""
import xarray as xr
from reservoir_cd.config import load_yaml
from reservoir_cd.grid import Grid, grid_bbox_lonlat
from reservoir_cd.preprocessing import esri, jrc

b = load_yaml("configs/build.yaml")
for s in load_yaml(b["sites_config"])["sites"]:
    ds = xr.open_zarr(s["zarr"])
    g = Grid.from_cube_coords(ds.x.values, ds.y.values)
    bb = grid_bbox_lonlat(g)
    print(s["name"], bb, flush=True)
    for y in b["esri_years"]:
        a = esri.fetch_esri_year(y, g, bb, b["esri_cache"], s["name"])
        print(" esri", y, (a == 0).mean(), flush=True)
    j = jrc.fetch_jrc_extent(g, bb, b["jrc_cache"], s["name"])
    print(" jrc", (j == 1).sum(), flush=True)
