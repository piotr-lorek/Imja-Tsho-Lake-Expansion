import numpy as np

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
