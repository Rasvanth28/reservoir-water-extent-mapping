"""QA figures for the label pipeline."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from reservoir_cd.grid import Grid

CLASS_COLORS = {0: "#1f78b4", 1: "#1b5e20", 2: "#d2b48c", 3: "#ff7f00", 255: "#9e9e9e"}
LEGEND = [Patch(color=CLASS_COLORS[k], label=n) for k, n in
          zip((0, 1, 2, 3, 255), ["Water", "Forest", "Bare", "Other", "Unlabeled"])]


def lab_cmap() -> ListedColormap:
    return ListedColormap([CLASS_COLORS[k] for k in (0, 1, 2, 3, 255)])


def lab_img(a: np.ndarray) -> np.ndarray:
    return np.where(a == 255, 4, a)


def true_color(bands: np.ndarray) -> np.ndarray:
    rgb = np.stack([bands[2], bands[1], bands[0]], axis=-1)
    out = np.ones_like(rgb)
    ok = np.isfinite(rgb).all(axis=-1)
    if not ok.any():
        return out
    for i in range(3):
        v = rgb[..., i]
        lo, hi = np.percentile(v[ok], [2, 92])  # 92: residual cloud patches must not dominate the stretch
        out[..., i] = np.where(ok, np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1), 1.0)
    return out


def extent_of(g: Grid) -> list[float]:
    return [g.x0, g.x0 + g.res * g.width, g.y0 - g.res * g.height, g.y0]


def year_overview(path: Path, site: str, year: int, bands: np.ndarray, rule: np.ndarray, cons: np.ndarray,
                  g: Grid, n_scenes: int) -> None:
    e = extent_of(g)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.5))
    axs[0].imshow(true_color(bands), extent=e)
    axs[0].set_title(f"{site} {year} Jan-Apr medoid true colour ({n_scenes} scenes)")
    for ax, lab, name in ((axs[1], rule, "rule label"), (axs[2], cons, "consensus label")):
        ax.imshow(lab_img(lab), cmap=lab_cmap(), vmin=0, vmax=4, interpolation="nearest", extent=e)
        ax.set_title(f"{year} {name}")
    axs[2].legend(handles=LEGEND, loc="lower right", fontsize=7)
    for ax in axs:
        ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def extent_figure(path: Path, site: str, ext: dict, bands: np.ndarray, g: Grid, year: int) -> None:
    e = extent_of(g)
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    axs[0].imshow(true_color(bands), extent=e)
    axs[0].set_title(f"{site} true colour ({year} Jan-Apr medoid)")
    axs[1].imshow(true_color(bands), extent=e)
    fr = np.ma.masked_where(~(ext["freq"] > 0), ext["freq"])
    h = axs[1].imshow(fr, cmap="viridis", vmin=0, vmax=1, extent=e, alpha=0.85, interpolation="nearest")
    for key, col, ls, lw in (("max_extent", "red", "-", 0.8), ("study", "white", "--", 0.8),
                             ("permanent", "cyan", "-", 0.6)):
        if ext[key].any():
            axs[1].contour(g.xc, g.yc, ext[key].astype(float), levels=[0.5], colors=col, linewidths=lw,
                           linestyles=ls)
    plt.colorbar(h, ax=axs[1], shrink=0.7, label="water frequency")
    axs[1].set_title("Water frequency; red = max extent, cyan = permanent, white dashed = study area")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def static_forest_figure(path: Path, site: str, bands: np.ndarray, prob: np.ndarray, trees_years: np.ndarray,
                         maps: dict, g: Grid, max_extent: np.ndarray) -> None:
    e = extent_of(g)
    c1, es = maps["forest_c1"], maps["forest_esri"]
    agree = np.zeros(c1.shape, np.uint8)       # 0 neither, 1 both, 2 C1 only, 3 Esri only
    agree[c1 & es], agree[c1 & ~es], agree[~c1 & es] = 1, 2, 3
    aspect = prob.shape[0] / prob.shape[1]
    fig, axs = plt.subplots(2, 3, figsize=(20, 2 * 6.5 * aspect + 1.5))
    axs[0, 0].imshow(true_color(bands), extent=e)
    axs[0, 0].set_title("true colour")
    im = axs[0, 1].imshow(prob, vmin=0, vmax=1, cmap="RdYlGn", extent=e, interpolation="nearest")
    plt.colorbar(im, ax=axs[0, 1], shrink=0.6)
    axs[0, 1].set_title("C1 forest probability (RF on phenology)")
    im = axs[0, 2].imshow(trees_years, vmin=0, vmax=7, cmap="Greens", extent=e, interpolation="nearest")
    plt.colorbar(im, ax=axs[0, 2], shrink=0.6)
    axs[0, 2].set_title("Esri Trees: # years of 7 (2017-2023)")
    cm = ListedColormap(["#eeeeee", "#1b5e20", "#e66101", "#5e3c99"])
    axs[1, 0].imshow(agree, cmap=cm, vmin=0, vmax=3, extent=e, interpolation="nearest")
    axs[1, 0].legend(handles=[Patch(color=c, label=l) for c, l in zip(
        ["#eeeeee", "#1b5e20", "#e66101", "#5e3c99"], ["neither", "both", "C1 only", "Esri only (>=4 yr)"])],
        loc="lower right", fontsize=7)
    axs[1, 0].set_title("agreement C1 vs Esri")
    cons = maps["forest_consensus"]
    cc = ListedColormap(["#d9d9d9", "#1b5e20", "#fdb863"])
    axs[1, 1].imshow(np.select([cons == 0, cons == 1], [0, 1], 2), cmap=cc, vmin=0, vmax=2, extent=e,
                     interpolation="nearest")
    axs[1, 1].legend(handles=[Patch(color=c, label=l) for c, l in zip(
        ["#d9d9d9", "#1b5e20", "#fdb863"], ["non-forest (confident)", "forest (consensus)", "unlabeled"])],
        loc="lower right", fontsize=7)
    axs[1, 1].set_title("static forest consensus (training map)")
    axs[1, 2].imshow(true_color(bands), extent=e)
    ov = np.ma.masked_where(~c1, np.ones(c1.shape))
    axs[1, 2].imshow(ov, cmap=ListedColormap(["#00ff00"]), alpha=0.45, extent=e, interpolation="nearest")
    ov2 = np.ma.masked_where(trees_years < 4, np.ones(c1.shape))
    axs[1, 2].contour(g.xc, g.yc, (trees_years >= 4).astype(float), levels=[0.5], colors="magenta", linewidths=0.5)
    axs[1, 2].contour(g.xc, g.yc, max_extent.astype(float), levels=[0.5], colors="red", linewidths=0.7)
    axs[1, 2].set_title("C1 forest (green) + Esri Trees>=4yr outline (magenta) + max extent (red)")
    for ax in axs.ravel():
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{site} static forest")
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def phenology_figure(path: Path, site: str, feats: dict) -> None:
    names = ["c0", "amp", "phase", "p10", "p90", "std", "ia_std", "mndwi_med"]
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for ax, n in zip(axs.ravel(), names):
        im = ax.imshow(feats[n], cmap="twilight" if n == "phase" else "viridis", interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title(n)
        ax.axis("off")
    fig.suptitle(f"{site} phenology features (all clear dates 2015-2026)")
    fig.savefig(path, dpi=70, bbox_inches="tight")
    plt.close(fig)


def importance_figure(path: Path, imp: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    k = sorted(imp, key=imp.get)
    ax.barh(k, [imp[i] for i in k])
    ax.set_title("RF feature importance (pooled)")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
