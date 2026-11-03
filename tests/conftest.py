import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine

from glacial_lake.io import Scene

# Surface reflectance per band for each synthetic surface type.
LAND = {"B02": 0.10, "B03": 0.12, "B04": 0.14, "B08": 0.25, "B11": 0.30}
WATER = {"B02": 0.08, "B03": 0.10, "B04": 0.07, "B08": 0.05, "B11": 0.01}
# High MNDWI like water, but bright in the near-infrared.
SNOW = {"B02": 0.85, "B03": 0.80, "B04": 0.78, "B08": 0.70, "B11": 0.05}

SHAPE = (60, 80)
LAKE = (slice(10, 30), slice(10, 50))  # 20 x 40 = 800 px
POND = (slice(45, 48), slice(10, 13))  # 3 x 3 = 9 px
SNOW_PATCH = (slice(40, 58), slice(55, 78))  # 18 x 23 = 414 px


def make_scene(*, include_snow: bool = True) -> Scene:
    regions = [(LAKE, WATER), (POND, WATER)]
    if include_snow:
        regions.append((SNOW_PATCH, SNOW))
    bands = {}
    for band, land_value in LAND.items():
        image = np.full(SHAPE, land_value)
        for window, surface in regions:
            image[window] = surface[band]
        bands[band] = image
    return Scene(
        date="2000-01-01",
        bands=bands,
        transform=Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 3_000_000.0),
        crs=CRS.from_epsg(32645),
    )


@pytest.fixture
def scene() -> Scene:
    return make_scene()


@pytest.fixture
def lake_mask() -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[LAKE] = True
    return mask
