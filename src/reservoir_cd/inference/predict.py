"""Full-scene inference: overlapping tiles blended with a triangular window (EDD 3.6).

For each site and year pair the scene is cut into 256x256 tiles (stride = tile - overlap, last
tile shifted to the edge), the model's per-date softmax probabilities are accumulated with a
triangular weight (centre > edge, which hides tile seams) and normalised. Writes
<out_dir>/<Site>.zarr with
  pred   uint8 (P, 2, H, W)  per-date class (0 Water 1 Forest 2 Bare 3 Other, 255 = no data at that date)
  change uint8 (P, H, W)     0 none, 1 water loss, 2 water gain, 255 = no data at either date
  water_prob float16 (P, 2, H, W)  blended P(Water) per date
and attrs (years, pairs, run info). No validation-point exclusion here (it only affects labels).

CLI: python -m reservoir_cd.inference.predict --config <run>/config.yaml --ckpt <run>/checkpoints/last.ckpt \
        --out data/processed/predictions/<name> [--sites Mettur] [--overlap 128] [--device cuda]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import zarr

from reservoir_cd.datasets import normalize as nz
from reservoir_cd.datasets.chips import _doy_year, open_store, window_starts
from reservoir_cd.datasets.pairs import consecutive_pairs


def tri_window(n: int, floor: float = 0.05) -> np.ndarray:
    t = 1.0 - np.abs(2.0 * (np.arange(n) + 0.5) / n - 1.0)
    w = np.maximum(t, floor)
    return np.outer(w, w).astype(np.float32)


def load_trained(cfg: dict, ckpt: str | Path | None, device: str = "cpu"):
    """Build the model from the training config and load a (possibly trainable-only) checkpoint."""
    from reservoir_cd.training.module import ChangeModule
    m = ChangeModule(cfg)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
    return m.model.to(device).eval()


@torch.no_grad()
def predict_site(model, store_path: str | Path, out_path: str | Path, pair_step: int = 1, tile: int = 256,
                 overlap: int = 128, batch_size: int = 4, device: str = "cpu") -> None:
    g = open_store(store_path)
    a = dict(g.attrs)
    H, W, years = a["height"], a["width"], a["years"]
    yi = {y: i for i, y in enumerate(years)}
    pairs = consecutive_pairs(years, pair_step)
    from pyproj import Transformer
    tr = Transformer.from_crs(a["crs"], "EPSG:4326", always_xy=True)
    o = zarr.open_group(str(out_path), mode="w")
    codec = zarr.codecs.ZstdCodec(level=5)
    pred = o.create_array("pred", shape=(len(pairs), 2, H, W), chunks=(1, 1, tile, tile), dtype="u1", compressors=codec)
    chg = o.create_array("change", shape=(len(pairs), H, W), chunks=(1, tile, tile), dtype="u1", compressors=codec)
    wp = o.create_array("water_prob", shape=(len(pairs), 2, H, W), chunks=(1, 1, tile, tile), dtype="f2", compressors=codec)
    win = torch.from_numpy(tri_window(tile))
    rows, cols = window_starts(H, tile, tile - overlap), window_starts(W, tile, tile - overlap)
    tiles = [(r, c) for r in rows for c in cols]
    refl_all = g["refl"]
    for p, (t1, t2) in enumerate(pairs):
        acc = torch.zeros(2, 4, H, W)
        wsum = torch.zeros(H, W)
        valid = np.zeros((2, H, W), bool)
        for b0 in range(0, len(tiles), batch_size):
            batch = tiles[b0:b0 + batch_size]
            imgs, tcs, lcs = [], [], []
            for r, c in batch:
                sl = (slice(r, r + tile), slice(c, c + tile))
                refl = np.stack([refl_all[yi[t1], :, sl[0], sl[1]], refl_all[yi[t2], :, sl[0], sl[1]]])
                valid[:, sl[0], sl[1]] = nz.int16_to_valid(refl)
                imgs.append(nz.normalize_hls(refl))
                tc = []
                for y in (t1, t2):
                    d = g["date_days"][yi[y], sl[0], sl[1]]
                    d = d[d > 0]
                    tc.append(_doy_year(float(np.median(d))) if d.size else (y, 60))
                tcs.append(tc)
                lon, lat = tr.transform(a["x0"] + a["res"] * (c + tile / 2), a["y0"] - a["res"] * (r + tile / 2))
                lcs.append([lat, lon])
            x = torch.from_numpy(np.stack(imgs)).to(device)
            logits = model(x, temporal_coords=torch.tensor(tcs, dtype=torch.float32, device=device),
                           location_coords=torch.tensor(lcs, dtype=torch.float32, device=device))
            prob = logits.softmax(2).float().cpu()
            for k, (r, c) in enumerate(batch):
                acc[:, :, r:r + tile, c:c + tile] += prob[k] * win
                wsum[r:r + tile, c:c + tile] += win
        prob = acc / wsum.clamp(min=1e-6)
        cls = prob.argmax(1).numpy().astype(np.uint8)  # (2,H,W)
        cls[~valid] = 255
        pred[p] = cls
        wp[p] = prob[:, 0].numpy().astype(np.float16)
        ch = np.zeros((H, W), np.uint8)
        ch[(cls[0] == 0) & (cls[1] != 0)] = 1
        ch[(cls[0] != 0) & (cls[1] == 0)] = 2
        ch[~valid.all(0)] = 255
        chg[p] = ch
        print(f"  pair {t1}->{t2} done", flush=True)
    o.attrs.update(years=years, pairs=[list(q) for q in pairs], overlap=overlap, tile=tile,
                   classes={"0": "Water", "1": "Forest", "2": "Bare", "3": "Other", "255": "no data"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chips-dir", default="data/processed/chips")
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    model = load_trained(cfg, a.ckpt, a.device)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for z in sorted(Path(a.chips_dir).glob("*.zarr")):
        if a.sites and z.stem not in a.sites:
            continue
        print(z.stem, flush=True)
        predict_site(model, z, Path(a.out) / z.name, overlap=a.overlap, batch_size=a.batch_size, device=a.device)


if __name__ == "__main__":
    main()
