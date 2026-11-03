import numpy as np
import pytest

from glacial_lake import metrics

from conftest import POND, SHAPE


def test_area_km2(lake_mask):
    assert metrics.area_km2(lake_mask, pixel_area_m2=100.0) == pytest.approx(0.08)


def test_shape_descriptors(lake_mask):
    shape = metrics.shape_descriptors(lake_mask, pixel_size_m=(10.0, 10.0))
    assert shape.centroid == pytest.approx((19.5, 29.5))
    assert shape.col_range == (10, 49)
    assert shape.bbox_px == (20, 40)
    assert shape.bbox_m == (200.0, 400.0)
    assert shape.aspect_ratio == pytest.approx(2.0)


def test_shape_descriptors_rejects_empty_mask():
    with pytest.raises(ValueError):
        metrics.shape_descriptors(np.zeros(SHAPE, dtype=bool), (10.0, 10.0))


def test_containment(lake_mask):
    grown = lake_mask.copy()
    grown[:, 50:60] = True
    assert metrics.containment(lake_mask, grown) == 1.0
    assert metrics.containment(grown, lake_mask) == pytest.approx(800 / grown.sum())


def test_nearby_component_sizes_only_counts_components_close_to_the_lake(lake_mask):
    water = lake_mask.copy()
    water[POND] = True  # 15 px below the lake
    water[31, 20] = True  # 1 px gap below the lake

    assert metrics.nearby_component_sizes(water, lake_mask, distance_px=3).tolist() == [1]
    assert metrics.nearby_component_sizes(water, lake_mask, distance_px=20).tolist() == [9, 1]


def test_area_changes_consecutive_then_overall():
    changes = metrics.area_changes({2018: 1.0, 2021: 1.5, 2024: 1.8})
    assert [(c.start, c.end) for c in changes] == [(2018, 2021), (2021, 2024), (2018, 2024)]
    assert changes[0].absolute_km2 == pytest.approx(0.5)
    assert changes[0].percent == pytest.approx(50.0)
    assert changes[2].percent == pytest.approx(80.0)


def test_area_changes_with_two_years_has_no_duplicate_overall_entry():
    assert len(metrics.area_changes({2018: 1.0, 2024: 2.0})) == 1


def test_compound_annual_growth_rate():
    assert metrics.compound_annual_growth_rate({2000: 1.0, 2002: 1.21}) == pytest.approx(0.1)


def test_boundary_uncertainty_scales_with_pixel_area(lake_mask):
    small = metrics.boundary_uncertainty_km2(lake_mask, pixel_area_m2=100.0)
    large = metrics.boundary_uncertainty_km2(lake_mask, pixel_area_m2=400.0)
    assert small > 0
    assert large == pytest.approx(4 * small)
