import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from glacial_lake import plotting
from glacial_lake.pipeline import detect_lake

from conftest import LAKE, SNOW_PATCH, make_scene

matplotlib.use("Agg")


def test_detect_lake_isolates_the_lake(scene, lake_mask):
    detection = detect_lake(scene)
    np.testing.assert_array_equal(detection.lake_mask, lake_mask)
    assert detection.area_km2 == pytest.approx(0.08)
    assert np.count_nonzero(detection.water_mask) == 800 + 9  # lake + pond, no snow


def test_snow_ice_mask_is_the_snow_patch(scene):
    detection = detect_lake(scene)
    assert detection.snow_ice_mask[SNOW_PATCH].all()
    assert np.count_nonzero(detection.snow_ice_mask) == 414


def test_detect_lake_overrides(scene):
    no_ceiling = detect_lake(scene, nir_ceiling=np.inf)
    assert np.count_nonzero(no_ceiling.water_mask) == 800 + 9 + 414  # snow passes

    too_strict = detect_lake(scene, mndwi_threshold=0.9)  # above the lake's MNDWI of 0.82
    assert too_strict.otsu_threshold == 0.9
    assert not too_strict.lake_mask.any()


def test_no_data_pixels_are_never_water(scene):
    for band in scene.bands.values():
        band[LAKE] = 0.0
    detection = detect_lake(scene)
    assert not scene.valid_mask[LAKE].any()
    assert not detection.water_mask[LAKE].any()


@pytest.fixture
def detections():
    return {2018: detect_lake(make_scene()), 2021: detect_lake(make_scene(include_snow=False))}


@pytest.mark.parametrize(
    "make_figure",
    [
        lambda d: plotting.plot_false_color_composites({y: det.scene for y, det in d.items()}),
        plotting.plot_index_maps,
        lambda d: plotting.plot_index_histograms(d, "NDWI", xlim=(-0.3, 0.7)),
        lambda d: plotting.plot_index_histograms(d, "MNDWI", xlim=(-0.5, 1.0), show_otsu=True),
        plotting.plot_results_grid,
        lambda d: plotting.plot_area_bars({y: det.area_km2 for y, det in d.items()}),
        lambda d: plotting.plot_area_bars(
            {y: det.area_km2 for y, det in d.items()}, {2018: 0.01, 2021: 0.02}, "Test Lake"
        ),
        plotting.plot_boundary_evolution,
        lambda d: plotting.plot_boundary_evolution(d, zoom_padding=5),
        plotting.plot_fragmentation,
    ],
)
def test_figures_render(detections, make_figure, tmp_path):
    fig = make_figure(detections)
    assert plotting.save_figure(fig, "figure.jpg", tmp_path).stat().st_size > 0
    plt.close(fig)


def test_index_histograms_reject_unknown_index(detections):
    with pytest.raises(ValueError, match="NDWI"):
        plotting.plot_index_histograms(detections, "NDSI", xlim=(-1, 1))


def test_bounding_box_includes_the_last_row_and_column():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True
    rows, cols = plotting.bounding_box(mask)
    assert (rows, cols) == (slice(2, 5), slice(3, 7))
    assert mask[rows, cols].all()


def test_bounding_box_padding_is_clipped_to_the_image():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 9] = True
    assert plotting.bounding_box(mask, padding=3) == (slice(0, 4), slice(6, 10))


def test_bounding_box_rejects_empty_mask():
    with pytest.raises(ValueError, match="empty"):
        plotting.bounding_box(np.zeros((5, 5), dtype=bool))


def test_boundary_evolution_rejects_more_years_than_colors(scene):
    detection = detect_lake(scene)
    with pytest.raises(ValueError, match="at most"):
        plotting.plot_boundary_evolution({year: detection for year in range(4)})
