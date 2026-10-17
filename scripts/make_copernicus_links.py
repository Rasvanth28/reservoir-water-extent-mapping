"""Make Copernicus Browser helpers for a labeling sheet.

Reads only `<SITE>_labels.csv` (never the key file, so the output stays blind)
and writes, next to it in `copernicus/`:
  - `<SITE>_copernicus.html`: one row per point with T1/T2 links that open
    Copernicus Browser centred on the pixel, with the date window
    (date ± window days) and true colour already set.
  - `<SITE>_squares.geojson`: all 30 m pixel squares, for "Upload area of interest".
  - `squares/<point_id>.geojson`: one square per point, if a single AOI is easier.

Usage: .venv/bin/python scripts/make_copernicus_links.py data/ground_truth/pilot/BHA_labels.csv
"""

import argparse
import csv
import json
from datetime import date, timedelta
from html import escape
from pathlib import Path
from urllib.parse import urlencode

from pyproj import Transformer

BROWSER = "https://browser.dataspace.copernicus.eu/"
UTM = "EPSG:32643"
HALF_PIXEL = 15.0  # metres; 30 m Landsat pixel
S2_L2A_START = date(2017, 4, 1)  # CDSE L2A archive; use L1C before this


def browser_url(lat, lon, centre, window_days, zoom):
    start = centre - timedelta(days=window_days)
    end = centre + timedelta(days=window_days)
    if end < date(2015, 7, 1):
        dataset = "LANDSAT_OT_L2"  # no Sentinel-2 before mid-2015
    elif centre < S2_L2A_START:
        dataset = "S2_L1C_CDAS"
    else:
        dataset = "S2_L2A_CDAS"
    params = {
        "zoom": zoom,
        "lat": f"{lat:.6f}",
        "lng": f"{lon:.6f}",
        "themeId": "DEFAULT-THEME",
        "datasetId": dataset,
        "fromTime": f"{start.isoformat()}T00:00:00.000Z",
        "toTime": f"{end.isoformat()}T23:59:59.999Z",
        "layerId": "1_TRUE_COLOR",
    }
    return BROWSER + "?" + urlencode(params), dataset


def pixel_square(lon, lat, to_utm, to_ll):
    x, y = to_utm.transform(lon, lat)
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
    return [list(to_ll.transform(x + dx * HALF_PIXEL, y + dy * HALF_PIXEL)) for dx, dy in corners]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_csv", type=Path)
    ap.add_argument("--window-days", type=int, default=15)
    ap.add_argument("--zoom", type=int, default=16)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.labels_csv.open()))
    site = args.labels_csv.stem.replace("_labels", "")
    out = args.labels_csv.parent / "copernicus"
    (out / "squares").mkdir(parents=True, exist_ok=True)

    to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    to_ll = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)

    features, table = [], []
    for r in rows:
        lon, lat = float(r["lon"]), float(r["lat"])
        feat = {
            "type": "Feature",
            "properties": {"point_id": r["point_id"]},
            "geometry": {"type": "Polygon", "coordinates": [pixel_square(lon, lat, to_utm, to_ll)]},
        }
        features.append(feat)
        (out / "squares" / f"{r['point_id']}.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": [feat]})
        )
        cells = []
        for t in ("t1", "t2"):
            d = date.fromisoformat(r[f"date_{t}"])
            url, dataset = browser_url(lat, lon, d, args.window_days, args.zoom)
            lo, hi = d - timedelta(days=args.window_days), d + timedelta(days=args.window_days)
            cells.append(
                f'<td><a href="{escape(url)}" target="_blank">{d}</a>'
                f"<br><small>{lo} → {hi} · {dataset}</small></td>"
            )
        table.append(
            f"<tr><td><b>{escape(r['point_id'])}</b></td><td>{lat:.6f}, {lon:.6f}</td>{''.join(cells)}</tr>"
        )

    (out / f"{site}_squares.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    (out / f"{site}_copernicus.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>{site} Copernicus links</title>
<style>body{{font:14px system-ui;margin:16px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:6px 10px;text-align:left}}tr:nth-child(even){{background:#f6f6f6}}
a:visited{{color:#7a3e9d}}</style>
<h2>{site}: Copernicus Browser links (±{args.window_days} days, true colour)</h2>
<p>Click a date to open Copernicus Browser on that pixel with the date window set. Pick the clearest
image in the timeline. The 30 m square is the 3×3 block of Sentinel-2 pixels around the centre; to see it
outlined, upload <code>squares/&lt;point_id&gt;.geojson</code> via the "Upload" button in the area-of-interest tool.</p>
<table><tr><th>Point</th><th>Lat, lon</th><th>T1</th><th>T2</th></tr>
{''.join(table)}</table>
"""
    )
    print(f"Wrote {out}/{site}_copernicus.html, {site}_squares.geojson and {len(rows)} per-point squares")


if __name__ == "__main__":
    main()
