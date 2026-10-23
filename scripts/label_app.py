"""Keyboard-driven labeling app for the validation workbook (local, stdlib + openpyxl only).

Serves one page on 127.0.0.1 showing the T1 and T2 reference chips of a point side by side. One
keystroke per decision: class (W/F/B/O/N), then confidence (1/2/3) completes the date; completing T2
saves the point to the workbook and moves to the next. Source/date of the reference come from the chip
unless you press G (opens Google Earth Pro at the point and date; type the GE image date). Reads only
the workbook and `reference_chips/*.png` (never the *_key.csv files or model output).

Each save rewrites `<sheet>.xlsx` atomically and regenerates `<sheet>_filled.csv`. A timestamped copy of
the workbook is made in `<sheet dir>/backups/` at start. Close the workbook in Excel/Numbers first.

Usage:
  .venv/bin/python scripts/label_app.py data/ground_truth/full/PIL/PIL_labels.xlsx --interpreter KC
  options: --port 8765  --force (ignore an Excel lock file)  --no-browser
Keys: see the footer of the page, or LABELING_GUIDE.md section 1.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import webbrowser
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_workbook as lw  # noqa: E402

CHIP_RE = re.compile(r"^([A-Za-z0-9]+-\d+)_(t[12])\.png$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LABELS = {"Water", "Forest", "Bare", "Other", "NA"}
SOURCES = {"GE", "S2", "LS"}
KML_NS = 'xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2"'


class Store:
    """The workbook held in memory; every save is written to disk atomically."""

    def __init__(self, path: Path, interpreter: str | None):
        self.path = path
        self.lock = threading.Lock()
        self.wb = load_workbook(path)
        self.ws = self.wb["labels"]
        header = [c.value for c in self.ws[1]]
        if header[:len(lw.COLUMNS + lw.HELPERS)] != lw.COLUMNS + lw.HELPERS:
            raise SystemExit(f"{path}: unexpected column layout {header}")
        self.cix = {h: i + 1 for i, h in enumerate(header)}
        self.rows = {}  # point_id -> sheet row number
        for r in range(2, self.ws.max_row + 1):
            pid = self.ws.cell(r, 1).value
            if pid:
                self.rows[str(pid)] = r
        self.interpreter = interpreter or self._latest("interpreter")
        if not self.interpreter:
            raise SystemExit("no interpreter in the workbook yet; pass --interpreter")

    def _latest(self, name):
        for r in range(self.ws.max_row, 1, -1):
            v = lw.fmt(self.ws.cell(r, self.cix[name]).value)
            if v:
                return v
        return None

    def state(self) -> dict:
        ge = load_ge_dates(self.path, next(iter(self.rows), "X-").split("-")[0])
        out = []
        for pid, r in self.rows.items():
            rec = {c: lw.fmt(self.ws.cell(r, self.cix[c]).value) for c in lw.COLUMNS + lw.HELPERS}
            for t in ("t1", "t2"):  # formulas count as empty
                if rec[f"ref_date_{t}"].startswith("="):
                    rec[f"ref_date_{t}"] = ""
                # GE images within the window, nearest first; None = dates not fetched for this point
                rec[f"ge_{t}"] = None if pid not in ge else ge_in_window(ge[pid], rec[f"date_{t}"])
            out.append(rec)
        return {"interpreter": self.interpreter, "rows": out}

    def get(self, pid, name):
        return lw.fmt(self.ws.cell(self.rows[pid], self.cix[name]).value)

    def save(self, d: dict):
        pid = d.get("point_id")
        if pid not in self.rows:
            raise ValueError(f"unknown point {pid}")
        r = self.rows[pid]
        vals = {}
        for t in ("t1", "t2"):
            label = d.get(f"label_{t}") or ""
            if label not in LABELS | {""}:
                raise ValueError(f"bad label {label}")
            if not label:  # cleared: restore the chip-date formula
                vals.update({f"label_{t}": None, f"confidence_{t}": None, f"mixed_{t}": None,
                             f"ref_source_{t}": None, f"ref_date_{t}": lw.ref_date_formula(t, r)})
                continue
            conf = d.get(f"confidence_{t}")
            conf = None if conf in (None, "") else int(conf)
            if conf not in (None, 0, 1, 2, 3):
                raise ValueError(f"bad confidence {conf}")
            src = (d.get(f"ref_source_{t}") or "").upper()
            rdate = d.get(f"ref_date_{t}") or ""
            if src not in SOURCES | {""} or (rdate and not DATE_RE.match(rdate)):
                raise ValueError(f"bad source/date {src} {rdate}")
            vals.update({f"label_{t}": label, f"confidence_{t}": conf, f"mixed_{t}": bool(d.get(f"mixed_{t}")),
                         f"ref_source_{t}": src or None, f"ref_date_{t}": rdate or None})
        minutes = int(d.get("minutes") or 1)
        vals.update({"interpreter": self.interpreter, "minutes": max(1, minutes),
                     "notes": (d.get("notes") or "").strip() or None})
        with self.lock:
            for k, v in vals.items():
                self.ws.cell(r, self.cix[k]).value = v
            tmp = self.path.with_name(f".tmp_{self.path.name}")
            self.wb.save(tmp)
            os.replace(tmp, self.path)
            lw.export(self.path)

    def date_of(self, pid, t):
        return self.get(pid, f"date_{t}")

    def lonlat(self, pid):
        return float(self.get(pid, "lon")), float(self.get(pid, "lat"))


def load_ge_dates(sheet: Path, site: str) -> dict:
    """{point_id: [dates]} from <SITE>_ge_dates.json (scripts/fetch_ge_dates.py); re-read on every /state."""
    for d in (sheet.parent, sheet.parent.parent / site):
        p = d / f"{site}_ge_dates.json"
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except ValueError:  # being rewritten by the fetcher
                return {}
    return {}


def ge_in_window(dates: list[str], target: str, window: int = 15) -> list[dict]:
    t = date.fromisoformat(target)
    hits = [{"date": d, "off": (date.fromisoformat(d) - t).days} for d in dates]
    return sorted((h for h in hits if abs(h["off"]) <= window), key=lambda h: abs(h["off"]))


def fly_ge(lon, lat, distance=450) -> bool:
    """Fly Google Earth Pro to the point via its AppleScript API and bring it to the front (no pins added)."""
    cmd = (f'tell application "Google Earth Pro" to SetViewInfo {{latitude:{lat}, longitude:{lon}, '
           f'distance:{distance}, tilt:0, azimuth:0}} speed 6')
    ok = subprocess.run(["osascript", "-e", cmd], capture_output=True, timeout=20).returncode == 0
    if ok:
        subprocess.Popen(["open", "-a", "Google Earth Pro"])
    return ok


def square_lonlat(lon, lat, half=15.0):
    """Corners of the +-15 m square around the point (EPSG:32643, degree approximation if no pyproj)."""
    try:
        from pyproj import Transformer
        fwd = Transformer.from_crs(4326, 32643, always_xy=True)
        inv = Transformer.from_crs(32643, 4326, always_xy=True)
        x, y = fwd.transform(lon, lat)
        return [inv.transform(x + dx, y + dy) for dx, dy in
                ((-half, -half), (half, -half), (half, half), (-half, half), (-half, -half))]
    except ImportError:
        dlat = half / 111320.0
        dlon = half / (111320.0 * math.cos(math.radians(lat)))
        return [(lon + sx * dlon, lat + sy * dlat) for sx, sy in
                ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]


def make_kml(pid, lon, lat, date_str, range_m=450) -> str:
    """KML with the 30 m pixel outline and a LookAt carrying the date, so GE's historical imagery jumps to it."""
    ring = " ".join(f"{x:.7f},{y:.7f},0" for x, y in square_lonlat(lon, lat))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml {KML_NS}>
<Document><name>{pid}</name>
<Style id="px"><LineStyle><color>ff00ffff</color><width>2</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>
<Placemark><name>{pid} {date_str}</name><styleUrl>#px</styleUrl>
<LookAt><gx:TimeStamp><when>{date_str}</when></gx:TimeStamp><longitude>{lon:.7f}</longitude><latitude>{lat:.7f}</latitude><altitude>0</altitude><heading>0</heading><tilt>0</tilt><range>{range_m}</range><altitudeMode>relativeToGround</altitudeMode></LookAt>
<Polygon><tessellate>1</tessellate><outerBoundaryIs><LinearRing><coordinates>{ring}</coordinates></LinearRing></outerBoundaryIs></Polygon>
</Placemark></Document></kml>
"""


def open_in_ge(store: Store, pid, t, tmpdir: Path) -> Path | None:
    lon, lat = store.lonlat(pid)
    if fly_ge(lon, lat):
        return None
    p = tmpdir / f"{pid}_{t}.kml"  # fallback: hand GE a KML (adds a pin to Temporary Places)
    p.write_text(make_kml(pid, lon, lat, store.date_of(pid, t)))
    subprocess.Popen(["open", str(p)])  # macOS: hands the KML to Google Earth Pro
    return p


def find_chip(sheet: Path, name: str) -> Path | None:
    site = name.split("-")[0]
    for d in (sheet.parent / "reference_chips", sheet.parent.parent / site / "reference_chips"):
        if (d / name).is_file():
            return d / name
    return None


def make_handler(store: Store, tmpdir: Path):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def js(self, obj, code=200):
            self.send(code, json.dumps(obj).encode())

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/state":
                with store.lock:
                    self.js(store.state())
            elif path.startswith("/chip/"):
                name = path[len("/chip/"):]
                f = CHIP_RE.match(name) and find_chip(store.path, name)
                if not f:
                    return self.send(404, b"no chip", "text/plain")
                self.send(200, f.read_bytes(), "image/png", {"Cache-Control": "max-age=3600"})
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                d = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/save":
                    store.save(d)
                    self.js({"ok": True})
                elif self.path == "/ge":
                    if d.get("t") not in ("t1", "t2") or d.get("point_id") not in store.rows:
                        raise ValueError("bad point/date")
                    p = open_in_ge(store, d["point_id"], d["t"], tmpdir)
                    self.js({"ok": True, "kml": str(p) if p else None})
                else:
                    self.send(404, b"not found", "text/plain")
            except Exception as e:  # report to the page instead of dropping the connection
                self.js({"ok": False, "error": str(e)}, 400)

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xlsx", type=Path)
    ap.add_argument("--interpreter", help="initials; optional if the workbook already has some")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--force", action="store_true", help="start even if an Excel lock file exists")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    path = args.xlsx.resolve()
    if not path.is_file():
        raise SystemExit(f"{path} not found")
    locks = [path.with_name("~$" + path.name), path.with_name("~$" + path.name[2:])]
    if any(p.exists() for p in locks) and not args.force:
        raise SystemExit(f"Excel lock file found next to {path.name}: close it in Excel first (or --force)")
    store = Store(path, args.interpreter)
    bdir = path.parent / "backups"
    bdir.mkdir(exist_ok=True)
    backup = bdir / f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}"
    shutil.copy2(path, backup)
    tmpdir = Path(tempfile.mkdtemp(prefix="label_app_"))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store, tmpdir))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{path.name}: {len(store.rows)} points, interpreter {store.interpreter}, backup {backup.name}\n{url}  (Ctrl-C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Label app</title>
<style>
body{margin:0;font:14px system-ui,sans-serif;background:#1b1d20;color:#e6e6e6;display:flex;flex-direction:column;height:100vh}
header{display:flex;gap:24px;align-items:baseline;padding:8px 16px;background:#25282c;font-size:15px}
header b{font-size:18px}#msg{margin-left:auto;color:#ffb454}
main{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:10px;min-height:0}
.panel{border:3px solid #333;border-radius:6px;padding:8px;background:#222427;display:flex;flex-direction:column;min-height:0;overflow:auto}
.panel.active{border-color:#4aa3ff;background:#242a31}
.panel h2{margin:0 0 4px;font-size:15px}
.panel img{width:100%;height:auto;display:block;background:#000}
.vals{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:8px;font-size:15px}
.vals span b{color:#8fd18f}.vals .empty{color:#777}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;background:#b8860b;color:#fff;margin-left:8px;font-size:13px}
.warn{background:#a33}.hint{color:#ffb454;margin-top:4px}
.ge{margin:4px 0 6px;padding:6px 10px;border-radius:4px;font-size:15px}
.ge.yes{background:#1f6f3a;color:#fff;font-weight:600}.ge.no{background:#2c2f33;color:#999}.ge.unk{background:#2c2f33;color:#ffb454}
.nochip{padding:40px 10px;text-align:center;color:#ffb454;border:1px dashed #777}
input[type=text],input[type=date]{background:#111;color:#eee;border:1px solid #555;padding:3px 6px;font-size:14px}
footer{padding:6px 16px;background:#25282c;font-size:12px;color:#aaa;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
footer input{flex:1;min-width:240px}kbd{background:#333;border-radius:3px;padding:0 4px;color:#fff}
</style></head><body>
<header><b id="pid"></b><span id="idx"></span><span id="done"></span><span id="who"></span><span id="timer"></span><span id="msg"></span></header>
<main><div class="panel" id="p_t1"></div><div class="panel" id="p_t2"></div></main>
<footer>
<span>Notes <kbd>/</kbd> <input type="text" id="notes" placeholder="notes"></span>
<span><kbd>W</kbd>ater <kbd>F</kbd>orest <kbd>B</kbd>are <kbd>O</kbd>ther <kbd>N</kbd>A &middot; <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> confidence (completes) &middot;
<kbd>M</kbd> mixed (then 2nd class) &middot; <kbd>G</kbd> Google Earth &middot; <kbd>S</kbd> back to chip &middot; <kbd>Tab</kbd> T1/T2 &middot;
<kbd>&larr;</kbd><kbd>&rarr;</kbd> prev/next &middot; <kbd>U</kbd> next unlabeled &middot; <kbd>Backspace</kbd> clear date</span>
</footer>
<script>
const T = ['t1', 't2'];
const CLS = {w: 'Water', f: 'Forest', b: 'Bare', o: 'Other'};
let rows = [], idx = 0, active = 't1', who = '';
const sec = {};          // seconds on screen this session, per point
let shownAt = Date.now(), q = Promise.resolve();

const done = (r, t) => r['label_' + t] !== '' && r['confidence_' + t] !== '';
const complete = r => done(r, 't1') && done(r, 't2');
const cur = () => rows[idx];
function msg(s, err) { const m = document.getElementById('msg'); m.textContent = s; m.style.color = err ? '#ff6b6b' : '#ffb454'; }

function norm(r) {
  for (const t of T) {
    r['confidence_' + t] = r['confidence_' + t] === '' ? '' : parseInt(r['confidence_' + t]);
    r['mixed_' + t] = String(r['mixed_' + t]).toUpperCase() === 'TRUE';
  }
  r._base = parseInt(r.minutes) || 0; r._pend = {t1: false, t2: false}; r._dirty = false;
  return r;
}
function elapsed(pid) { return (sec[pid] || 0) + (cur() && cur().point_id === pid ? (Date.now() - shownAt) / 1000 : 0); }

function chipOf(r, t) { return [r['chip_source_' + t], r['chip_date_' + t]]; }
function ensureSource(r, t) {
  if (!r['ref_source_' + t]) { const [s, d] = chipOf(r, t); r['ref_source_' + t] = s; r['ref_date_' + t] = d; }
}
function setSeg(r, t, text) {   // "T1 mixed A/B" segment inside notes
  const tag = t.toUpperCase() + ' mixed ';
  const parts = r.notes.split(';').map(s => s.trim()).filter(s => s && !s.startsWith(tag));
  if (text) parts.push(tag + text);
  r.notes = parts.join('; ');
}

const sgn = n => (n >= 0 ? '+' : '') + n;
function geBanner(r, t) {   // Google Earth images within +-15 days of the target (scripts/fetch_ge_dates.py)
  const g = r['ge_' + t];
  if (g === null || g === undefined) return '<div class="ge unk">GE dates not fetched for this point &mdash; check GE yourself if needed</div>';
  if (!g.length) return '<div class="ge no">No Google Earth image within &plusmn;15 days &mdash; label from the chip</div>';
  return '<div class="ge yes">Check Google Earth: ' + g.map(h => h.date + ' (' + sgn(h.off) + ' d)').join(', ') +
    ' &mdash; press <kbd>G</kbd></div>';
}

function render() {
  const r = cur();
  document.getElementById('pid').textContent = r.point_id;
  document.getElementById('idx').textContent = (idx + 1) + ' / ' + rows.length;
  document.getElementById('done').textContent = rows.filter(complete).length + ' complete';
  document.getElementById('who').textContent = who;
  for (const t of T) {
    const [cs, cd] = chipOf(r, t), el = document.getElementById('p_' + t);
    const lab = r['label_' + t], src = r['ref_source_' + t];
    const v = (n, x) => '<span>' + n + ': ' + (x === '' || x === undefined ? '<span class="empty">-</span>' : '<b>' + x + '</b>') + '</span>';
    const pend = r._pend[t];
    el.className = 'panel' + (t === active ? ' active' : '');
    el.innerHTML = '<h2>' + t.toUpperCase() + ' &middot; target ' + r['date_' + t] +
      (cs ? ' &middot; chip ' + cs + ' ' + cd : '') +
      (cs === 'LS' ? '<span class="badge">LS &mdash; lower confidence by 1</span>' : '') +
      (done(r, t) ? ' &#10003;' : '') + '</h2>' + geBanner(r, t) +
      (cs ? '<img src="/chip/' + r.point_id + '_' + t + '.png" onerror="this.outerHTML=\'<div class=nochip>no chip &mdash; use GE (G) or NA</div>\'">'
          : '<div class="nochip">no chip &mdash; use GE (G) or NA</div>') +
      '<div class="vals">' + v('label', lab) + v('conf', r['confidence_' + t]) + v('mixed', r['mixed_' + t] ? 'yes' : 'no') +
      v('source', src) + '<span>date: ' + (src === 'GE'
        ? '<input type="date" id="ge_' + t + '" value="' + r['ref_date_' + t] + '">' : (r['ref_date_' + t] ? '<b>' + r['ref_date_' + t] + '</b>' : '<span class="empty">-</span>')) +
      '</span></div>' +
      (pend ? '<div class="hint">mixed: pick ' + (pend === 'first' ? 'main class, then 2nd class' : '2nd class') + '</div>' : '') +
      (lab && lab !== 'NA' && src === 'GE' && !r['ref_date_' + t] ? '<div class="hint">type the GE image date</div>' : '');
    const gi = document.getElementById('ge_' + t);
    if (gi) gi.addEventListener('input', () => { r['ref_date_' + t] = gi.value; r._dirty = true; });
  }
  document.getElementById('notes').value = r.notes;
  [rows[idx + 1], rows[idx - 1]].forEach(n => { if (n) for (const t of T) new Image().src = '/chip/' + n.point_id + '_' + t + '.png'; });
}

function payload(r) {
  const p = {point_id: r.point_id, notes: r.notes, minutes: Math.max(1, r._base + Math.ceil(elapsed(r.point_id) / 60))};
  for (const t of T) for (const k of ['label_', 'confidence_', 'mixed_', 'ref_source_', 'ref_date_']) p[k + t] = r[k + t];
  return p;
}
function save(r) {
  if (!r._dirty) return;
  r._dirty = false;
  const body = JSON.stringify(payload(r));
  q = q.then(() => fetch('/save', {method: 'POST', body}).then(x => x.json()).then(j => {
    if (!j.ok) throw new Error(j.error); msg('saved ' + r.point_id);
  })).catch(e => { r._dirty = true; msg('SAVE FAILED ' + r.point_id + ': ' + e.message, true); });
}
function go(i) {
  if (i < 0 || i >= rows.length) { msg(i < 0 ? 'first point' : 'last point'); save(cur()); return; }
  const r = cur(); sec[r.point_id] = elapsed(r.point_id); shownAt = Date.now();
  save(r); idx = i; r._pend = {t1: false, t2: false};
  active = !done(cur(), 't1') ? 't1' : (!done(cur(), 't2') ? 't2' : 't1');
  shownAt = Date.now(); render();
}
function nextUnlabeled(from) {
  for (let k = 1; k <= rows.length; k++) { const i = (from + k) % rows.length; if (!complete(rows[i])) return i; }
  return -1;
}
function finish(t) {   // the date is complete
  if (t === 't1') { active = 't2'; render(); }
  else { const r = cur(); if (idx + 1 < rows.length) go(idx + 1); else { save(r); msg('last point saved'); render(); } }
}

function setClass(key) {
  const r = cur(), t = active, c = CLS[key], pend = r._pend[t];
  r._dirty = true;
  if (pend === 'second') {
    if (c === r['label_' + t]) { msg('2nd class must differ'); return; }
    setSeg(r, t, r['label_' + t] + '/' + c); r._pend[t] = false;
  } else {
    r['label_' + t] = c; ensureSource(r, t);
    if (pend === 'first') r._pend[t] = 'second';
  }
  render();
}
function setNA() {
  const r = cur(), t = active;
  r['label_' + t] = 'NA'; r['confidence_' + t] = 0; r._pend[t] = false; ensureSource(r, t); r._dirty = true;
  finish(t);
}
function setConf(n) {
  const r = cur(), t = active;
  if (!r['label_' + t]) { msg('pick a class first'); return; }
  r['confidence_' + t] = n; r._pend[t] = false; r._dirty = true;
  finish(t);
}
function toggleMixed() {
  const r = cur(), t = active;
  r['mixed_' + t] = !r['mixed_' + t]; r._dirty = true;
  r._pend[t] = r['mixed_' + t] ? (r['label_' + t] ? 'second' : 'first') : false;
  if (!r['mixed_' + t]) setSeg(r, t, null);
  render();
}
function clearDate() {
  const r = cur(), t = active;
  for (const k of ['label_', 'confidence_', 'ref_source_', 'ref_date_']) r[k + t] = '';
  r['mixed_' + t] = false; r._pend[t] = false; setSeg(r, t, null); r._dirty = true; render();
}
function resetSource() {
  const r = cur(), t = active;
  r['ref_source_' + t] = ''; r['ref_date_' + t] = ''; ensureSource(r, t); r._dirty = true; render();
}
function openGE() {
  const r = cur(), t = active;
  fetch('/ge', {method: 'POST', body: JSON.stringify({point_id: r.point_id, t})}).then(x => x.json())
    .then(j => { if (!j.ok) throw new Error(j.error); }).catch(e => msg('GE failed: ' + e.message, true));
  const g = r['ge_' + t];   // prefill the nearest in-window GE image date; correct it if that image was cloudy
  r['ref_source_' + t] = 'GE'; r['ref_date_' + t] = g && g.length ? g[0].date : r['date_' + t]; r._dirty = true; render();
  msg('Google Earth: set the time slider to ' + r['ref_date_' + t]);
  const gi = document.getElementById('ge_' + t); gi.focus(); gi.select && gi.select();
}

document.addEventListener('keydown', e => {
  const a = document.activeElement;
  if (a && a.tagName === 'INPUT') {
    if (e.key === 'Enter' || e.key === 'Escape') { e.preventDefault(); a.blur(); }
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (CLS[k]) setClass(k);
  else if (k === 'n') setNA();
  else if (k === '1' || k === '2' || k === '3') setConf(parseInt(k));
  else if (k === 'm') toggleMixed();
  else if (k === 'g') openGE();
  else if (k === 's') resetSource();
  else if (k === '/') { e.preventDefault(); document.getElementById('notes').focus(); return; }
  else if (k === 'tab') { active = active === 't1' ? 't2' : 't1'; render(); }
  else if (k === 'arrowleft') go(idx - 1);
  else if (k === 'arrowright') go(idx + 1);
  else if (k === 'backspace') clearDate();
  else if (k === 'u') { const i = nextUnlabeled(idx); if (i < 0) msg('all points complete'); else go(i); }
  else return;
  e.preventDefault();
});
document.getElementById('notes').addEventListener('input', e => { cur().notes = e.target.value; cur()._dirty = true; });
window.addEventListener('beforeunload', () => { const r = cur(); if (r && r._dirty) navigator.sendBeacon('/save', JSON.stringify(payload(r))); });
setInterval(() => {
  const s = Math.floor(elapsed(cur().point_id));
  document.getElementById('timer').textContent = Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
}, 500);

fetch('/state').then(x => x.json()).then(s => {
  who = s.interpreter; rows = s.rows.map(norm);
  const i = rows.findIndex(r => !complete(r));
  idx = i < 0 ? 0 : i; active = i >= 0 && done(rows[i], 't1') ? 't2' : 't1';
  shownAt = Date.now(); render();
  if (i < 0) msg('all points complete');
});
</script></body></html>
"""

if __name__ == "__main__":
    main()
