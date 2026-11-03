"""Water segmentation and connected-component isolation."""

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

EIGHT_CONNECTIVITY = np.ones((3, 3), dtype=bool)


def water_mask(mndwi: np.ndarray, ndwi: np.ndarray) -> tuple[np.ndarray, float]:
    """Classify water as ``MNDWI > Otsu threshold`` and ``NDWI > 0``.

    Returns the boolean mask and the Otsu threshold computed on the MNDWI.
    """
    threshold = float(threshold_otsu(mndwi.ravel()))
    return (mndwi > threshold) & (ndwi > 0), threshold


def label_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label 8-connected components.

    Returns the label image and the pixel count of each label, where index 0
    (background) is set to 0.
    """
    labels, _ = ndimage.label(mask, structure=EIGHT_CONNECTIVITY)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels, sizes


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected component of a boolean mask."""
    labels, sizes = label_components(mask)
    if sizes.max(initial=0) == 0:
        return np.zeros_like(mask, dtype=bool)
    return labels == sizes.argmax()
