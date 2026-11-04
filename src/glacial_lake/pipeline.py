"""End-to-end lake detection for one scene."""

from dataclasses import dataclass

import numpy as np

from . import indices, segmentation
from .io import Scene
from .metrics import area_km2
from .segmentation import NIR_CEILING


@dataclass(frozen=True)
class LakeDetection:
    scene: Scene
    ndwi: np.ndarray
    mndwi: np.ndarray
    otsu_threshold: float  # MNDWI threshold used
    nir_ceiling: float
    water_mask: np.ndarray  # all pixels classified as water
    lake_mask: np.ndarray  # largest connected water body only

    @property
    def area_km2(self) -> float:
        return area_km2(self.lake_mask, self.scene.pixel_area_m2)

    @property
    def snow_ice_mask(self) -> np.ndarray:
        """Pixels above the MNDWI threshold but rejected by the NIR test: snow or ice."""
        return (
            self.scene.valid_mask
            & (self.mndwi > self.otsu_threshold)
            & (self.scene.bands["B08"] >= self.nir_ceiling)
        )


def detect_lake(
    scene: Scene, nir_ceiling: float = NIR_CEILING, mndwi_threshold: float | None = None
) -> LakeDetection:
    """Detect the lake in one scene.

    ``mndwi_threshold`` replaces the per-scene Otsu threshold, e.g. with one pooled
    over all dates.
    """
    ndwi = indices.ndwi(scene.bands)
    mndwi = indices.mndwi(scene.bands)
    water, threshold = segmentation.water_mask(
        mndwi,
        ndwi,
        scene.bands["B08"],
        nir_ceiling=nir_ceiling,
        threshold=mndwi_threshold,
        valid=scene.valid_mask,
    )
    return LakeDetection(
        scene=scene,
        ndwi=ndwi,
        mndwi=mndwi,
        otsu_threshold=threshold,
        nir_ceiling=nir_ceiling,
        water_mask=water,
        lake_mask=segmentation.largest_component(water),
    )
