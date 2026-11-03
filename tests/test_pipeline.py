import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from glacial_lake import plotting
from glacial_lake.pipeline import detect_lake

from conftest import make_scene

matplotlib.use("Agg")


def test_detect_lake_isolates_the_lake(scene, lake_mask):
    detection = detect_lake(scene)
    np.testing.assert_array_equal(detection.lake_mask, lake_mask)
    assert detection.area_km2 == pytest.approx(0.08)
    assert np.count_nonzero(detection.water_mask) == 800 + 9  # lake + pond, no snow


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
        plotting.plot_boundary_evolution,
        lambda d: plotting.plot_boundary_evolution(d, zoom_padding=5),
        plotting.plot_fragmentation,
    ],
)
def test_figures_render(detections, make_figure, tmp_path):
    fig = make_figure(detections)
    assert plotting.save_figure(fig, "figure.jpg", tmp_path).stat().st_size > 0
    plt.close(fig)


def test_boundary_evolution_rejects_more_years_than_colors(scene):
    detection = detect_lake(scene)
    with pytest.raises(ValueError, match="at most"):
        plotting.plot_boundary_evolution({year: detection for year in range(4)})
