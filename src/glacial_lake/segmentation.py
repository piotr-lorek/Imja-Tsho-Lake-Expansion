"""Water segmentation and connected-component isolation."""

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

EIGHT_CONNECTIVITY = np.ones((3, 3), dtype=bool)

# Liquid water absorbs near-infrared, while snow and ice reflect it strongly. Pixels
# brighter than this NIR (B08) reflectance are rejected even if their MNDWI is high,
# because MNDWI uses the same formula as the snow index NDSI.
NIR_CEILING = 0.20


def water_mask(
    mndwi: np.ndarray,
    ndwi: np.ndarray,
    nir: np.ndarray,
    nir_ceiling: float = NIR_CEILING,
    threshold: float | None = None,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Classify open water as ``MNDWI > threshold``, ``NDWI > 0`` and ``NIR < nir_ceiling``.

    The MNDWI threshold is found with Otsu's method unless ``threshold`` is given (for
    example one pooled over several dates). Pixels outside ``valid`` (no data) are left
    out of the Otsu histogram and are never water.

    Returns the boolean mask and the MNDWI threshold used.
    """
    if valid is None:
        valid = np.ones(mndwi.shape, dtype=bool)
    if threshold is None:
        if not valid.any():
            raise ValueError("no valid pixels to threshold")
        threshold = float(threshold_otsu(mndwi[valid]))
    return valid & (mndwi > threshold) & (ndwi > 0) & (nir < nir_ceiling), threshold


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
