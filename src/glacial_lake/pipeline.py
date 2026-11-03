"""End-to-end lake detection for one scene."""

from dataclasses import dataclass

import numpy as np

from . import indices, segmentation
from .io import Scene
from .metrics import area_km2


@dataclass(frozen=True)
class LakeDetection:
    scene: Scene
    ndwi: np.ndarray
    mndwi: np.ndarray
    otsu_threshold: float
    water_mask: np.ndarray  # all pixels classified as water
    lake_mask: np.ndarray  # largest connected water body only

    @property
    def area_km2(self) -> float:
        return area_km2(self.lake_mask, self.scene.pixel_area_m2)


def detect_lake(scene: Scene) -> LakeDetection:
    ndwi = indices.ndwi(scene.bands)
    mndwi = indices.mndwi(scene.bands)
    water, threshold = segmentation.water_mask(mndwi, ndwi)
    return LakeDetection(
        scene=scene,
        ndwi=ndwi,
        mndwi=mndwi,
        otsu_threshold=threshold,
        water_mask=water,
        lake_mask=segmentation.largest_component(water),
    )
