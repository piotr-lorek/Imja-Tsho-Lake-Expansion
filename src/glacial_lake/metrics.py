"""Area, shape and change metrics for binary lake masks."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.measure import find_contours

from .segmentation import EIGHT_CONNECTIVITY, label_components

M2_PER_KM2 = 1_000_000.0


def area_km2(mask: np.ndarray, pixel_area_m2: float) -> float:
    return float(np.count_nonzero(mask)) * pixel_area_m2 / M2_PER_KM2


@dataclass(frozen=True)
class ShapeDescriptors:
    centroid: tuple[float, float]  # (row, col) in pixels
    col_range: tuple[int, int]  # (westernmost, easternmost) column
    bbox_px: tuple[int, int]  # (rows, cols)
    bbox_m: tuple[float, float]  # (height, width)
    aspect_ratio: float  # bounding-box width / height


def shape_descriptors(mask: np.ndarray, pixel_size_m: tuple[float, float]) -> ShapeDescriptors:
    """Centroid, extent, bounding box and aspect ratio of a non-empty mask."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("mask is empty")
    centroid_row, centroid_col = ndimage.center_of_mass(mask)
    (_, col_min), (_, col_max) = coords.min(axis=0), coords.max(axis=0)
    rows, cols = coords.max(axis=0) - coords.min(axis=0) + 1
    pixel_width, pixel_height = pixel_size_m
    return ShapeDescriptors(
        centroid=(float(centroid_row), float(centroid_col)),
        col_range=(int(col_min), int(col_max)),
        bbox_px=(int(rows), int(cols)),
        bbox_m=(rows * pixel_height, cols * pixel_width),
        aspect_ratio=float(cols / rows),
    )


def containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of ``inner`` pixels that also belong to ``outer``."""
    return float(np.count_nonzero(inner & outer) / np.count_nonzero(inner))


def nearby_component_sizes(
    water_mask: np.ndarray, lake_mask: np.ndarray, distance_px: int
) -> np.ndarray:
    """Pixel counts of water components, other than the lake, within ``distance_px`` of it.

    Used to test whether the lake was split into pieces (e.g. by floating ice) that the
    largest-component step would have discarded. Sorted largest first.
    """
    labels, sizes = label_components(water_mask)
    near = ndimage.binary_dilation(lake_mask, EIGHT_CONNECTIVITY, iterations=distance_px)
    nearby = np.unique(labels[near & water_mask & ~lake_mask])
    return np.sort(sizes[nearby[nearby > 0]])[::-1]


def boundary_length_px(mask: np.ndarray) -> int:
    """Number of vertices on the mask's sub-pixel contour, a proxy for boundary pixels."""
    return sum(len(contour) for contour in find_contours(mask.astype(float), level=0.5))


def boundary_uncertainty_km2(mask: np.ndarray, pixel_area_m2: float) -> float:
    """Area uncertainty assuming each boundary pixel may be misclassified (±1 pixel)."""
    return boundary_length_px(mask) * pixel_area_m2 / M2_PER_KM2


@dataclass(frozen=True)
class Change:
    start: int
    end: int
    absolute_km2: float
    percent: float


def area_changes(areas_km2: Mapping[int, float]) -> list[Change]:
    """Changes between consecutive years and over the whole period, in chronological order."""
    years = sorted(areas_km2)
    pairs = list(zip(years, years[1:], strict=False))
    if len(years) > 2:
        pairs.append((years[0], years[-1]))
    return [
        Change(
            start=a,
            end=b,
            absolute_km2=areas_km2[b] - areas_km2[a],
            percent=(areas_km2[b] - areas_km2[a]) / areas_km2[a] * 100,
        )
        for a, b in pairs
    ]


def compound_annual_growth_rate(areas_km2: Mapping[int, float]) -> float:
    """Compound annual growth rate between the first and last year, as a fraction."""
    years = sorted(areas_km2)
    span = years[-1] - years[0]
    return (areas_km2[years[-1]] / areas_km2[years[0]]) ** (1 / span) - 1
