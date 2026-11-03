"""Normalised-difference spectral water indices."""

from collections.abc import Mapping

import numpy as np


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return ``(a - b) / (a + b)`` bounded to [-1, 1], with 0 where undefined."""
    with np.errstate(divide="ignore", invalid="ignore"):
        nd = (a - b) / (a + b)
    nd = np.nan_to_num(nd, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(nd, -1.0, 1.0)


def ndwi(bands: Mapping[str, np.ndarray]) -> np.ndarray:
    """Normalised Difference Water Index (McFeeters, 1996): green vs. near-infrared."""
    return normalized_difference(bands["B03"], bands["B08"])


def mndwi(bands: Mapping[str, np.ndarray]) -> np.ndarray:
    """Modified NDWI (Xu, 2006): green vs. shortwave-infrared."""
    return normalized_difference(bands["B03"], bands["B11"])
