"""Figures for the lake analysis. Every function returns a Matplotlib figure."""

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from skimage.measure import find_contours

from .config import FIGURES_DIR
from .io import Scene
from .pipeline import LakeDetection
from .segmentation import label_components

INDEX_CMAP = "RdYlBu"
BOUNDARY_COLORS = ("yellow", "darkorange", "red")
BAR_COLORS = ("#8FBADD", "#4A88B7", "#1D5382")
COMPONENT_COLORS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0))


# --- Image helpers ----------------------------------------------------------------------


def percentile_stretch(image: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Linearly stretch an image so its [low, high] percentiles map to [0, 1]."""
    p_low, p_high = np.percentile(image, (low, high))
    span = p_high - p_low
    stretched = (image - p_low) / span if span > 0 else image - p_low
    return np.clip(stretched, 0.0, 1.0)


def true_color(bands: Mapping[str, np.ndarray]) -> np.ndarray:
    """Red-green-blue (B04, B03, B02) composite, percentile-stretched."""
    return percentile_stretch(np.dstack((bands["B04"], bands["B03"], bands["B02"])))


def false_color(bands: Mapping[str, np.ndarray]) -> np.ndarray:
    """NIR-red-green (B08, B04, B03) composite, percentile-stretched."""
    return percentile_stretch(np.dstack((bands["B08"], bands["B04"], bands["B03"])))


def bounding_box(mask: np.ndarray, padding: int = 0) -> tuple[slice, slice]:
    """Row and column slices around the mask, padded and clipped to the image."""
    coords = np.argwhere(mask)
    (row_min, col_min), (row_max, col_max) = coords.min(axis=0), coords.max(axis=0)
    return (
        slice(max(0, row_min - padding), min(mask.shape[0], row_max + padding)),
        slice(max(0, col_min - padding), min(mask.shape[1], col_max + padding)),
    )


def save_figure(fig: Figure, name: str, figures_dir: Path = FIGURES_DIR, dpi: int = 150) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# --- Figures ----------------------------------------------------------------------------


def plot_false_color_composites(scenes: Mapping[int, Scene]) -> Figure:
    fig, axes = plt.subplots(1, len(scenes), figsize=(18, 6), squeeze=False)
    for ax, (year, scene) in zip(axes[0], scenes.items(), strict=True):
        ax.imshow(false_color(scene.bands))
        ax.set_title(f"False Color Composite ({year})", fontsize=14, fontweight="bold")
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_index_maps(detections: Mapping[int, LakeDetection]) -> Figure:
    fig, axes = plt.subplots(2, len(detections), figsize=(18, 10), squeeze=False)
    for col, (year, det) in enumerate(detections.items()):
        for row, (name, values) in enumerate((("NDWI", det.ndwi), ("MNDWI", det.mndwi))):
            ax = axes[row, col]
            im = ax.imshow(values, cmap=INDEX_CMAP, vmin=-1.0, vmax=1.0)
            ax.set_title(f"{name} ({year})", fontsize=14, fontweight="bold")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_index_histograms(
    detections: Mapping[int, LakeDetection],
    index: str,
    xlim: tuple[float, float],
    show_otsu: bool = False,
    color: str = "steelblue",
) -> Figure:
    """Log-scale histograms of ``index`` ("NDWI" or "MNDWI") for each year.

    Marks either the Otsu threshold (``show_otsu``) or zero.
    """
    fig, axes = plt.subplots(1, len(detections), figsize=(18, 5), squeeze=False)
    for ax, (year, det) in zip(axes[0], detections.items(), strict=True):
        values = det.ndwi if index == "NDWI" else det.mndwi
        ax.hist(values.ravel(), bins=200, range=(-1.0, 1.0), color=color, alpha=0.8)
        ax.set_xlim(*xlim)
        ax.set_yscale("log")
        if show_otsu:
            line_at, label = det.otsu_threshold, f"Otsu Thresh ({det.otsu_threshold:.4f})"
        else:
            line_at, label = 0.0, "Zero Threshold"
        ax.axvline(line_at, color="red", linestyle="--", linewidth=1.5, label=label)
        ax.set_title(f"{index} Distribution ({year})", fontsize=13, fontweight="bold")
        ax.set_xlabel(f"{index} Value")
        ax.set_ylabel("Pixel Frequency Count (Log Scale)")
        ax.legend()
    fig.tight_layout()
    return fig


def plot_results_grid(detections: Mapping[int, LakeDetection]) -> Figure:
    """True color, false color, MNDWI with Otsu contour, and isolated lake mask per year."""
    titles = ("True Color", "False Color", "MNDWI (Otsu Threshold)", "Isolated Water Mask")
    fig, axes = plt.subplots(len(detections), 4, figsize=(20, 15), squeeze=False)

    for row, (year, det) in enumerate(detections.items()):
        ax_tc, ax_fc, ax_mndwi, ax_mask = axes[row]

        ax_tc.imshow(true_color(det.scene.bands))
        ax_tc.set_xticks([])
        ax_tc.set_yticks([])
        ax_tc.set_ylabel(str(year), fontsize=18, fontweight="bold", labelpad=15)
        for spine in ax_tc.spines.values():
            spine.set_visible(False)

        ax_fc.imshow(false_color(det.scene.bands))
        ax_fc.axis("off")

        im = ax_mndwi.imshow(det.mndwi, cmap=INDEX_CMAP, vmin=-1.0, vmax=1.0)
        ax_mndwi.contour(det.mndwi, levels=[det.otsu_threshold], colors=["lime"], linewidths=1.5)
        fig.colorbar(im, ax=ax_mndwi, fraction=0.046, pad=0.04)
        ax_mndwi.axis("off")

        ax_mask.imshow(det.lake_mask, cmap="gray")
        ax_mask.axis("off")

        if row == 0:
            for ax, title in zip(axes[0], titles, strict=True):
                ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    fig.tight_layout()
    return fig


def plot_area_bars(areas_km2: Mapping[int, float]) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    years = sorted(areas_km2)
    values = [areas_km2[y] for y in years]
    bars = ax.bar(
        [str(y) for y in years],
        values,
        color=BAR_COLORS[: len(years)],
        edgecolor="black",
        linewidth=1.2,
        width=0.6,
    )
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_title(
        f"Imja Tsho Surface Area Expansion ({years[0]}–{years[-1]})",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylabel("Surface Area (km²)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    return fig


def plot_boundary_evolution(
    detections: Mapping[int, LakeDetection], zoom_padding: int | None = None
) -> Figure:
    """Lake outlines of every year over the latest true-color image.

    With ``zoom_padding``, the view is cropped to the latest lake plus that many pixels.
    """
    years = sorted(detections)
    if len(years) > len(BOUNDARY_COLORS):
        raise ValueError(f"at most {len(BOUNDARY_COLORS)} years can be overlaid")
    background = true_color(detections[years[-1]].scene.bands)
    window = (slice(None), slice(None))
    if zoom_padding is not None:
        window = bounding_box(detections[years[-1]].lake_mask, zoom_padding)

    fig, ax = plt.subplots(figsize=(10, 8) if zoom_padding is not None else (12, 10))
    ax.imshow(background[window])
    for year, color in zip(years, BOUNDARY_COLORS, strict=False):
        contours = find_contours(detections[year].lake_mask[window].astype(float), level=0.5)
        for i, contour in enumerate(contours):
            ax.plot(
                contour[:, 1],
                contour[:, 0],
                linewidth=2.5 if zoom_padding is not None else 2.0,
                color=color,
                label=str(year) if i == 0 else None,
            )

    if zoom_padding is not None:
        ax.set_title("Shoreline Segmentation Detail (Zoomed)", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", title="Boundary Year", fontsize=11, title_fontsize=12)
    else:
        ax.set_title(
            f"Lake Boundary Evolution: {years[0]}–{years[-1]} (overlaid on {years[-1]} True Color)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="upper right", title="Boundary Year", fontsize=12, title_fontsize=13)
    ax.axis("off")
    return fig


def plot_fragmentation(detections: Mapping[int, LakeDetection]) -> Figure:
    """Pre-isolation water mask, its three largest components and true color, per year."""
    titles = (
        "Full Pre-Isolation Mask\n(All Components)",
        "Top 3 Components Identified\n(Red=1st, Green=2nd, Cyan=3rd)",
        "True Color Reference\n(Spatial Context)",
    )
    fig, axes = plt.subplots(len(detections), 3, figsize=(18, 6 * len(detections)), squeeze=False)

    for row, (year, det) in enumerate(detections.items()):
        ax_raw, ax_components, ax_tc = axes[row]

        ax_raw.imshow(det.water_mask, cmap="gray")
        ax_raw.text(
            -0.1,
            0.5,
            str(year),
            transform=ax_raw.transAxes,
            fontsize=18,
            fontweight="bold",
            va="center",
            ha="right",
        )

        labels, sizes = label_components(det.water_mask)
        colored = np.zeros((*labels.shape, 3))
        for label, color in zip(np.argsort(sizes)[::-1], COMPONENT_COLORS, strict=False):
            if sizes[label] > 0:
                colored[labels == label] = color
        ax_components.imshow(colored)

        ax_tc.imshow(true_color(det.scene.bands))

        for ax in axes[row]:
            ax.axis("off")
        if row == 0:
            for ax, title in zip(axes[0], titles, strict=True):
                ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    fig.tight_layout()
    return fig
