"""Figures for the MaxEnt report (static PNGs)."""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

from .environment import describe  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Single-hue sequential ramp (light -> dark) for habitat suitability.
SUITABILITY = LinearSegmentedColormap.from_list("suit", ["#f3f6f1", "#b7d7a8", "#5aa25a", "#1f6b35", "#0b3a1c"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#c0392b", "#f2c2b8", "#e9e8e4", "#a9c8ee", "#1f5fae"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 150,
})


def _extent(template):
    t = template["transform"]
    return [t.c, t.c + t.a * template["width"], t.f + t.e * template["height"], t.f]


def _map_axes(ax):
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(False)
    ax.set_aspect("equal")


def _draw_boundaries(ax, boundaries, template=None):
    """Neighbouring countries (thin), provinces (fine), the focal country (bold), keeping the map extent."""
    if not boundaries:
        return
    if not isinstance(boundaries, dict):  # a plain GeoDataFrame
        boundaries = {"custom": boundaries}
    style = {
        "countries": dict(color=INK_2, linewidth=0.5),
        "provinces": dict(color=INK_2, linewidth=0.35, linestyle=(0, (3, 2))),
        "custom": dict(color=INK_2, linewidth=0.6),
        "country": dict(color=INK, linewidth=1.3),
    }
    for key in ("countries", "provinces", "custom", "country"):
        gdf = boundaries.get(key)
        if gdf is not None and not gdf.empty:
            gdf.boundary.plot(ax=ax, zorder=4, **style[key])
    if template is not None:
        e = _extent(template)
        ax.set_xlim(e[0], e[1])
        ax.set_ylim(e[2], e[3])


def _points(ax, pres, label="Occurrences"):
    ax.scatter(pres["lon"], pres["lat"], s=14, c=ORANGE, edgecolors="white", linewidths=0.7,
               label=label, zorder=3)


def suitability_map(path, grid, template, pres, title, boundaries=None):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(grid, extent=_extent(template), cmap=SUITABILITY, vmin=0, vmax=1, interpolation="nearest")
    _draw_boundaries(ax, boundaries, template)
    if pres is not None:
        _points(ax, pres)
        ax.legend(loc="lower left", frameon=True)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Habitat suitability (cloglog, 0–1)")
    ax.set_title(title)
    _map_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def binary_map(path, binary, template, pres, title, threshold, boundaries=None):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    cmap = ListedColormap(["#e9e8e4", "#1f6b35"])
    ax.imshow(binary, extent=_extent(template), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    _draw_boundaries(ax, boundaries, template)
    _points(ax, pres)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#1f6b35", label=f"Suitable (≥ {threshold:.3f})"),
               plt.Rectangle((0, 0), 1, 1, color="#e9e8e4", label="Unsuitable"),
               ax.collections[-1]]
    ax.legend(handles=handles, loc="lower left", frameon=True)
    ax.set_title(title)
    _map_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def change_map(path, change, template, title, label="Change in suitability (future − current)", boundaries=None):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    lim = np.nanmax(np.abs(change)) or 1
    im = ax.imshow(change, extent=_extent(template), cmap=DIVERGING, vmin=-lim, vmax=lim, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    _draw_boundaries(ax, boundaries, template)
    ax.set_title(title)
    _map_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def agreement_map(path, agreement, n_models, template, pres, title, boundaries=None):
    """Number of climate models (0..n) predicting suitable habitat in each cell."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = SUITABILITY(np.linspace(0.0, 0.85, n_models + 1))
    colors[0] = matplotlib.colors.to_rgba("#e9e8e4")
    cmap = ListedColormap(colors)
    im = ax.imshow(agreement, extent=_extent(template), cmap=cmap, vmin=-0.5, vmax=n_models + 0.5,
                   interpolation="nearest")
    _points(ax, pres)
    ax.legend(loc="lower left", frameon=True)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, ticks=range(n_models + 1))
    cb.set_label(f"Models predicting suitable (of {n_models})")
    _draw_boundaries(ax, boundaries, template)
    ax.set_title(title)
    _map_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def occurrence_map(path, raw, thinned, template, background_layer, bg, boundaries=None):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.imshow(np.where(np.isnan(background_layer), np.nan, 1), extent=_extent(template),
              cmap=ListedColormap(["#e9e8e4"]), interpolation="nearest")
    ax.scatter(bg["lon"], bg["lat"], s=1, c="#a3a29c", label=f"Background ({len(bg):,})", zorder=2)
    ax.scatter(raw["lon"], raw["lat"], s=10, facecolors="none", edgecolors=BLUE, linewidths=0.8,
               label=f"Cleaned records ({len(raw):,})", zorder=3)
    _points(ax, thinned, label=f"Thinned, used in model ({len(thinned):,})")
    _draw_boundaries(ax, boundaries, template)
    ax.legend(loc="lower left", frameon=True, markerscale=1.5)
    ax.set_title("Occurrence and background points")
    _map_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def correlation_heatmap(path, corr, kept):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.45 * n + 3, 0.45 * n + 2.5))
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    labels = [f"{c} ✓" if c in kept else c for c in corr.columns]
    ax.set_xticks(range(n), labels, rotation=90)
    ax.set_yticks(range(n), labels)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    ax.set_title("Correlation between bioclimatic variables (✓ = retained)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def tuning_plot(path, tuning, best):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fcs = list(dict.fromkeys(tuning["features"]))
    colors = [BLUE, ORANGE, "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    for i, fc in enumerate(fcs):
        d = tuning[tuning["features"] == fc]
        ax.plot(d["rm"], d["test_auc"], "-o", color=colors[i % len(colors)], lw=2, ms=5, label=fc.replace(",", "+"))
    ax.scatter([best["rm"]], [best["test_auc"]], s=160, facecolors="none", edgecolors=INK, linewidths=1.5,
               zorder=5, label="Selected")
    ax.set_xlabel("Regularization multiplier")
    ax.set_ylabel("Mean cross-validated test AUC")
    ax.set_title("Model tuning (spatial cross-validation)")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def roc_plot(path, fpr, tpr, auc_value):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"Training AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color=INK_2, lw=1, label="Random (AUC = 0.5)")
    ax.set_xlabel("1 − specificity (fraction of background predicted)")
    ax.set_ylabel("Sensitivity")
    ax.set_title("ROC curve (presence vs background)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def importance_plot(path, importance):
    imp = importance.sort_values("importance_pct")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(imp) + 1.5))
    labels = [f"{v} – {describe(v)}" for v in imp.index]
    ax.barh(labels, imp["importance_pct"], color=BLUE, height=0.6)
    for y, v in enumerate(imp["importance_pct"]):
        ax.text(v + 0.5, y, f"{v:.1f}%", va="center", color=INK_2, fontsize=9)
    ax.set_xlabel("Permutation importance (% of total AUC loss)")
    ax.set_title("Variable importance")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def response_plot(path, curves, pres_env):
    n = len(curves)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for ax, (var, df) in zip(axes.flat, curves.items()):
        ax.plot(df["value"], df["suitability"], color=BLUE, lw=2)
        ax.plot(pres_env[var], np.full(len(pres_env), -0.03), "|", color=ORANGE, ms=6, alpha=0.6)
        ax.set_ylim(-0.06, 1.02)
        ax.set_title(f"{var}: {describe(var)}", fontsize=9)
        ax.set_ylabel("Suitability")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle("Response curves (other variables at median; ticks = presence values)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
