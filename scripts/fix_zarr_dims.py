"""Add missing zarr v3 ``dimension_names`` to Mettur/Pillur stores (metadata only).

The ``landsat_data`` array carries its dimension names only inside ``attributes``,
so xarray cannot open the store. This script:
  * backs up every zarr.json it edits to ``zarr.json.bak`` (never overwritten),
  * sets top-level ``dimension_names`` on ``landsat_data`` (and on any 1-D coord
    array that lacks them) in both the array's zarr.json and the root
    consolidated metadata,
  * verifies that ``xr.open_zarr`` opens the store and ``time`` decodes.
Chunk data is never touched. Use ``--restore`` to put the backups back.
"""
import json
import shutil
import sys
from pathlib import Path

import xarray as xr

DATA = Path("data")
STORES = ["Mettur_datacube.zarr", "Pillur_datacube.zarr"]
DIMS = {
    "landsat_data": ["time", "band", "y", "x"],
    "time": ["time"], "band": ["band"], "y": ["y"], "x": ["x"],
}


def backup(path: Path) -> None:
    bak = path.with_name(path.name + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)


def fix_store(store: Path) -> None:
    root_json = store / "zarr.json"
    backup(root_json)
    root = json.loads(root_json.read_text())
    cons = root.get("consolidated_metadata", {}).get("metadata", {})
    for name, dims in DIMS.items():
        p = store / name / "zarr.json"
        meta = json.loads(p.read_text())
        if not meta.get("dimension_names"):
            backup(p)
            meta["dimension_names"] = dims
            p.write_text(json.dumps(meta, indent=2))
            print(f"  patched {p}")
        if name in cons and not cons[name].get("dimension_names"):
            cons[name]["dimension_names"] = dims
    root_json.write_text(json.dumps(root, indent=2))


def restore(store: Path) -> None:
    for bak in store.rglob("zarr.json.bak"):
        shutil.copy2(bak, bak.with_name("zarr.json"))
        print(f"  restored {bak.parent}")


def verify(store: Path) -> None:
    ds = xr.open_zarr(store)
    t = ds["time"].values
    print(f"  opened: {dict(ds.sizes)}; time {t.min()} -> {t.max()}")
    assert str(t.min())[:4] >= "2015" and str(t.max())[:4] <= "2026"


def main() -> None:
    for name in STORES:
        store = DATA / name
        print(name)
        if "--restore" in sys.argv:
            restore(store)
            continue
        fix_store(store)
        verify(store)


if __name__ == "__main__":
    main()
