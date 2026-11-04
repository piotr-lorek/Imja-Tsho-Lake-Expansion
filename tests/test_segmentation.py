import numpy as np
import pytest

from glacial_lake.indices import mndwi, ndwi
from glacial_lake.segmentation import label_components, largest_component, water_mask

from conftest import LAKE, POND, SNOW_PATCH


def test_water_mask_keeps_water_and_rejects_snow(scene):
    mask, threshold = water_mask(mndwi(scene.bands), ndwi(scene.bands), scene.bands["B08"])

    assert mask[LAKE].all()
    assert mask[POND].all()
    assert not mask[SNOW_PATCH].any()  # high MNDWI, but bright in NIR
    assert np.count_nonzero(mask) == 800 + 9
    assert -1.0 < threshold < 1.0


def test_snow_passes_without_nir_ceiling(scene):
    mask, _ = water_mask(
        mndwi(scene.bands), ndwi(scene.bands), scene.bands["B08"], nir_ceiling=np.inf
    )
    assert mask[SNOW_PATCH].all()


def one_pixel(mndwi: float, ndwi: float, nir: float) -> bool:
    """Classify a single pixel with a fixed MNDWI threshold of 0.25."""
    mask, _ = water_mask(np.array([mndwi]), np.array([ndwi]), np.array([nir]), threshold=0.25)
    return bool(mask[0])


@pytest.mark.parametrize(
    ("mndwi", "ndwi", "nir", "is_water"),
    [
        (0.8, 0.3, 0.19, True),  # just below the NIR ceiling
        (0.8, 0.3, 0.21, False),  # just above the NIR ceiling
        (0.8, -0.05, 0.10, False),  # fails NDWI > 0
        (0.2, 0.3, 0.10, False),  # below the MNDWI threshold
    ],
)
def test_water_mask_applies_every_test(mndwi, ndwi, nir, is_water):
    assert one_pixel(mndwi, ndwi, nir) is is_water


def test_otsu_threshold_falls_between_the_two_modes():
    mndwi = np.r_[np.full(900, 0.4), np.full(100, 0.8)]
    ones = np.ones_like(mndwi)
    mask, threshold = water_mask(mndwi, ones, ones * 0.1)
    assert 0.4 <= threshold < 0.8
    assert mask.sum() == 100


def test_invalid_pixels_are_ignored():
    mndwi = np.r_[np.full(10, 0.4), np.full(10, 0.8), np.zeros(1000)]
    valid = np.r_[np.ones(20, dtype=bool), np.zeros(1000, dtype=bool)]
    ones = np.ones_like(mndwi)
    mask, threshold = water_mask(mndwi, ones, ones * 0.1, valid=valid)
    assert 0.4 <= threshold < 0.8  # the no-data spike at 0 does not pull the threshold down
    assert mask.sum() == 10


def test_water_mask_rejects_all_invalid_input():
    zeros = np.zeros(4)
    with pytest.raises(ValueError, match="no valid"):
        water_mask(zeros, zeros, zeros, valid=np.zeros(4, dtype=bool))


def test_label_components_uses_eight_connectivity():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = mask[1, 1] = True  # diagonal neighbours
    mask[3, 3] = True
    _, sizes = label_components(mask)
    assert sizes[0] == 0
    assert sorted(sizes[sizes > 0]) == [1, 2]


def test_largest_component_keeps_only_the_biggest_body(lake_mask):
    mask = lake_mask.copy()
    mask[POND] = True
    np.testing.assert_array_equal(largest_component(mask), lake_mask)


def test_largest_component_of_empty_mask_is_empty():
    result = largest_component(np.zeros((5, 5), dtype=bool))
    assert result.dtype == bool and not result.any()
