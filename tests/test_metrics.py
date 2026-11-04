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


def test_containment_rejects_empty_inner_mask(lake_mask):
    with pytest.raises(ValueError, match="empty"):
        metrics.containment(np.zeros(SHAPE, dtype=bool), lake_mask)


def test_nearby_component_sizes_only_counts_components_close_to_the_lake(lake_mask):
    water = lake_mask.copy()
    water[POND] = True  # 15 px below the lake
    water[31, 20] = True  # 1 px gap below the lake

    assert metrics.nearby_component_sizes(water, lake_mask, distance_px=3).tolist() == [1]
    assert metrics.nearby_component_sizes(water, lake_mask, distance_px=20).tolist() == [9, 1]


def test_nearby_component_distance_is_euclidean(lake_mask):
    water = lake_mask.copy()
    water[32, 52] = True  # 3 px down and 3 px right of the lake's corner: 4.2 px away
    assert metrics.nearby_component_sizes(water, lake_mask, distance_px=3).size == 0
    assert metrics.nearby_component_sizes(water, lake_mask, distance_px=5).tolist() == [1]


def test_area_changes_consecutive_then_overall():
    changes = metrics.area_changes({2018: 1.0, 2021: 1.5, 2024: 1.8})
    assert [(c.start, c.end) for c in changes] == [(2018, 2021), (2021, 2024), (2018, 2024)]
    assert changes[0].absolute_km2 == pytest.approx(0.5)
    assert changes[0].percent == pytest.approx(50.0)
    assert changes[2].percent == pytest.approx(80.0)


def test_area_changes_combine_uncertainties_in_quadrature():
    changes = metrics.area_changes({2018: 1.0, 2021: 1.5}, {2018: 0.3, 2021: 0.4})
    assert changes[0].uncertainty_km2 == pytest.approx(0.5)
    assert metrics.area_changes({2018: 1.0, 2021: 1.5})[0].uncertainty_km2 is None


def test_area_changes_with_two_years_has_no_duplicate_overall_entry():
    assert len(metrics.area_changes({2018: 1.0, 2024: 2.0})) == 1


def test_area_changes_rejects_zero_start_area():
    with pytest.raises(ValueError, match="zero area"):
        metrics.area_changes({2018: 0.0, 2021: 1.0})


def test_compound_annual_growth_rate():
    assert metrics.compound_annual_growth_rate({2000: 1.0, 2002: 1.21}) == pytest.approx(0.1)


def test_compound_annual_growth_rate_uses_acquisition_dates():
    areas = {2000: 1.0, 2002: 1.21}
    dates = {2000: "2000-07-01", 2002: "2001-07-01"}  # one year apart, not two
    assert metrics.compound_annual_growth_rate(areas, dates) == pytest.approx(0.21, rel=1e-3)


@pytest.mark.parametrize(
    ("areas", "dates"),
    [
        ({2000: 1.0}, None),  # a single year
        ({2000: 0.0, 2002: 1.0}, None),  # zero start area
        ({2000: 1.0, 2002: 2.0}, {2000: "2002-01-01", 2002: "2002-01-01"}),  # no time elapsed
    ],
)
def test_compound_annual_growth_rate_rejects_undefined_cases(areas, dates):
    with pytest.raises(ValueError):
        metrics.compound_annual_growth_rate(areas, dates)


def test_boundary_vertex_count_of_a_rectangle(lake_mask):
    # A 20 x 40 px rectangle has 2 * (20 + 40) = 120 shoreline pixel edges; the closed
    # contour repeats its first vertex at the end.
    assert metrics.boundary_vertex_count(lake_mask) == 121
    assert metrics.boundary_uncertainty_km2(lake_mask, 100.0) == pytest.approx(0.0121)


def test_boundary_vertex_count_includes_holes(lake_mask):
    holed = lake_mask.copy()
    holed[20, 30] = False  # one-pixel hole: 4 edges + closing vertex
    assert metrics.boundary_vertex_count(holed) == 121 + 5


def test_boundary_uncertainty_scales_with_pixel_area(lake_mask):
    small = metrics.boundary_uncertainty_km2(lake_mask, pixel_area_m2=100.0)
    large = metrics.boundary_uncertainty_km2(lake_mask, pixel_area_m2=400.0)
    assert small > 0
    assert large == pytest.approx(4 * small)
