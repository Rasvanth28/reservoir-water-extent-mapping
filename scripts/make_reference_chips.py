"""Make Sentinel-2 reference image chips for a labeling sheet (no account needed).

For every point and date (T1, T2) it searches Sentinel-2 L2A on Planetary Computer
within date ± window days, picks the image closest to the date whose 3×3 S2 pixels
under the 30 m square are cloud-free (SCL), and saves a true-colour and a SWIR
false-colour chip with the 30 m square outlined. Dates before Sentinel-2 (mid-2015)
fall back to Landsat 8/9 L2. Reads only `<SITE>_labels.csv` (never the key file),
so the output stays blind.

Output: <sheet dir>/reference_chips/<point_id>_<t1|t2>.png and <SITE>_gallery.html

Usage: .venv/bin/python scripts/make_reference_chips.py data/ground_truth/pilot/BHA_labels.csv
"""

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from matplotlib.patches import Rectangle
from pyproj import Transformer
from rasterio.windows import from_bounds

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
HALF_CHIP = 150.0  # metres; chip is 300 m × 300 m
HALF_PIXEL = 15.0  # the 30 m Landsat pixel
SCL_BAD = {0, 1, 3, 8, 9, 10}  # nodata, saturated, cloud shadow, cloud med/high, cirrus
LS_CLOUD_BITS = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)  # dilated, cirrus, cloud, shadow

SENSORS = {
    "S2": {
        "collection": "sentinel-2-l2a",
        "rgb": ["B04", "B03", "B02"],
        "swir": ["B12", "B8A", "B04"],
        "qa": "SCL",
    },
    "LS": {
        "collection": "landsat-c2-l2",
        "rgb": ["red", "green", "blue"],
        "swir": ["swir22", "nir08", "red"],
        "qa": "qa_pixel",
    },
}


def read_window(href, bounds_xy, crs_from):
    with rasterio.open(href) as src:
        to_src = Transformer.from_crs(crs_from, src.crs, always_xy=True)
        x0, y0 = to_src.transform(bounds_xy[0], bounds_xy[1])
        x1, y1 = to_src.transform(bounds_xy[2], bounds_xy[3])
        win = from_bounds(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), src.transform)
        return src.read(1, window=win, boundless=True, fill_value=0).astype("float32")


def reflectance(sensor, item, arr):
    if sensor == "S2":
        offset = -1000.0 if float(item.properties.get("s2:processing_baseline", "0")) >= 4.0 else 0.0
        return (arr + offset) / 10000.0
    return arr * 0.0000275 - 0.2  # Landsat C2 L2 DN (these are raw COGs, not our datacube)


def centre_clear(sensor, qa):
    h, w = qa.shape
    cy, cx = h // 2, w // 2
    r = 2 if sensor == "S2" else 1
    block = qa[cy - r : cy + r + 1, cx - r : cx + r + 1].astype(int)
    if block.size == 0 or (block == 0).all():
        return False
    if sensor == "S2":
        return not np.isin(block, list(SCL_BAD)).any()
    return not (block & LS_CLOUD_BITS).any() and not (block & 1).any()


def compose(sensor, item, bands, bounds, crs):
    stack = [reflectance(sensor, item, read_window(item.assets[b].href, bounds, crs)) for b in bands]
    shape = min(a.shape for a in stack)
    stack = [a[: shape[0], : shape[1]] for a in stack]
    return np.clip(np.dstack(stack) / 0.3, 0, 1)


def find_image(catalog, lon, lat, centre, window_days, bounds, crs):
    start, end = centre - timedelta(days=window_days), centre + timedelta(days=window_days)
    for sensor in ("S2", "LS"):
        cfg = SENSORS[sensor]
        query = {"platform": {"in": ["landsat-8", "landsat-9"]}} if sensor == "LS" else None
        items = list(
            catalog.search(
                collections=[cfg["collection"]],
                intersects={"type": "Point", "coordinates": [lon, lat]},
                datetime=f"{start}/{end}",
                query=query,
            ).items()
        )
        items.sort(key=lambda it: abs((it.datetime.date() - centre).days))
        for it in items:
            qa = read_window(it.assets[cfg["qa"]].href, bounds, crs)
            if centre_clear(sensor, qa):
                return sensor, it, len(items)
    return None, None, 0


def retry(fn, tries=3, wait=3.0):
    """Call fn(); retry transient network errors (RasterioIOError, HTTP, STAC) a couple of times."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - any transient read error
            if k == tries - 1:
                raise
            time.sleep(wait * (k + 1))


def fetch_chip(catalog, r, t, window_days):
    """Network part (thread-safe): returns dict(status, ...) with arrays for a clear image."""
    lon, lat = float(r["lon"]), float(r["lat"])
    d = date.fromisoformat(r[f"date_{t}"])
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True).transform(lon, lat)
    bounds = (x - HALF_CHIP, y - HALF_CHIP, x + HALF_CHIP, y + HALF_CHIP)
    sensor, item, _ = retry(lambda: find_image(catalog, lon, lat, d, window_days, bounds, "EPSG:32643"))
    if item is None:
        return {"status": "none", "target": str(d)}
    cfg = SENSORS[sensor]
    rgb = retry(lambda: compose(sensor, item, cfg["rgb"], bounds, "EPSG:32643"))
    swir = retry(lambda: compose(sensor, item, cfg["swir"], bounds, "EPSG:32643"))
    img_date = item.datetime.date()
    return {"status": "ok", "sensor": sensor, "img_date": str(img_date), "off": (img_date - d).days,
            "target": str(d), "rgb": rgb, "swir": swir}


def save_chip(path, rgb, swir, title):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.3))
    for ax, img, name in zip(axes, (rgb, swir), ("True colour", "SWIR false colour (water = black)")):
        ax.imshow(img, extent=(-HALF_CHIP, HALF_CHIP, -HALF_CHIP, HALF_CHIP), interpolation="nearest")
        ax.add_patch(Rectangle((-HALF_PIXEL, -HALF_PIXEL), 30, 30, fill=False, ec="yellow", lw=1.8))
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_csv", type=Path)
    ap.add_argument("--window-days", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.labels_csv.open()))
    site = args.labels_csv.stem.replace("_labels", "")
    out = args.labels_csv.parent / "reference_chips"
    out.mkdir(exist_ok=True)
    catalog = pystac_client.Client.open(STAC, modifier=pc.sign_inplace)

    manifest_path = out / f"{site}_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    tasks = [(r, t) for r in rows for t in ("t1", "t2")
             if f"{r['point_id']}_{t}" not in manifest or
             (manifest[f"{r['point_id']}_{t}"]["status"] == "ok" and not (out / f"{r['point_id']}_{t}.png").exists())]
    print(f"{len(rows) * 2 - len(tasks)} chips already done, {len(tasks)} to fetch", flush=True)

    def work(task):
        r, t = task
        try:
            return task, fetch_chip(catalog, r, t, args.window_days)
        except Exception as e:  # noqa: BLE001
            return task, {"status": "error", "error": repr(e)[:200]}

    n_err = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for k, (task, res) in enumerate(ex.map(work, tasks), 1):
            r, t = task
            key = f"{r['point_id']}_{t}"
            if res["status"] == "ok":
                title = f"{r['point_id']} {t.upper()}  |  {res['sensor']} {res['img_date']} ({res['off']:+d} d from {res['target']})"
                save_chip(out / f"{key}.png", res.pop("rgb"), res.pop("swir"), title)
                manifest[key] = res
            elif res["status"] == "none":
                manifest[key] = res
            else:
                n_err += 1  # not stored -> retried on the next run
                print(f"{key}: ERROR {res['error']}", flush=True)
            if k % 25 == 0 or k == len(tasks):
                manifest_path.write_text(json.dumps(manifest))
                print(f"[{k}/{len(tasks)}] errors so far {n_err}", flush=True)
    manifest_path.write_text(json.dumps(manifest))

    gallery = []
    for r in rows:
        cells = []
        for t in ("t1", "t2"):
            m = manifest.get(f"{r['point_id']}_{t}")
            d = r[f"date_{t}"]
            if m is None:
                cells.append(f"<td><b>{t.upper()}</b> {d}<br><i>not fetched (error), rerun</i></td>")
            elif m["status"] == "none":
                cells.append(f"<td><b>{t.upper()}</b> {d}<br><i>No clear image within ±{args.window_days} days of {d}</i></td>")
            else:
                src = "S2" if m["sensor"] == "S2" else "LS"
                cells.append(
                    f'<td><b>{t.upper()}</b> target {d} · use <code>ref_source={src}</code>, '
                    f"<code>ref_date={m['img_date']}</code> ({m['off']:+d} d)<br>"
                    f'<img src="{r["point_id"]}_{t}.png" width="640"></td>')
        gallery.append(f"<tr><td><b>{escape(r['point_id'])}</b></td>{''.join(cells)}</tr>")

    (out / f"{site}_gallery.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>{site} reference chips</title>
<style>body{{font:14px system-ui;margin:16px}}table{{border-collapse:collapse}}
td{{border:1px solid #ccc;padding:6px;vertical-align:top}}</style>
<h2>{site}: Sentinel-2 reference chips (closest cloud-free image within ±{args.window_days} days)</h2>
<p>Each chip is 300 m across. The <b>yellow square is the 30 m pixel</b> to label (about 3×3 Sentinel-2 pixels).
Left: true colour. Right: SWIR false colour, where water is black, vegetation green, bare soil pink/brown.
Use these only when Google Earth has no clear image within ±15 days.</p>
<table>{''.join(gallery)}</table>"""
    )
    print(f"Wrote {out}/{site}_gallery.html")


if __name__ == "__main__":
    main()
