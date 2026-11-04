"""Run the full Imja Tsho analysis: detect the lake, save figures and a results table.

    uv run python scripts/run_analysis.py

Expects the data downloaded by scripts/download_data.py.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from glacial_lake import metrics, plotting
from glacial_lake.config import ACQUISITION_DATES, LAKE_NAME
from glacial_lake.io import load_scene
from glacial_lake.pipeline import LakeDetection, detect_lake

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"
DEFAULT_RESULTS_PATH = REPO_ROOT / "results" / "lake_areas.csv"


def uncertainties_km2(detections: dict[int, LakeDetection]) -> dict[int, float]:
    return {
        year: metrics.boundary_uncertainty_km2(det.lake_mask, det.scene.pixel_area_m2)
        for year, det in detections.items()
    }


def build_figures(detections: dict[int, LakeDetection]) -> dict[str, plt.Figure]:
    areas = {year: det.area_km2 for year, det in detections.items()}
    return {
        "false_color_composites.jpg": plotting.plot_false_color_composites(
            {year: det.scene for year, det in detections.items()}
        ),
        "index_maps.jpg": plotting.plot_index_maps(detections),
        "ndwi_histograms.png": plotting.plot_index_histograms(detections, "NDWI", xlim=(-0.3, 0.7)),
        "mndwi_histograms_otsu.png": plotting.plot_index_histograms(
            detections, "MNDWI", xlim=(-0.5, 1.0), show_otsu=True, color="teal"
        ),
        "final_results_grid.jpg": plotting.plot_results_grid(detections),
        "area_comparison_bar_chart.png": plotting.plot_area_bars(
            areas, uncertainties_km2(detections), lake_name=LAKE_NAME
        ),
        "lake_boundary_evolution_full.jpg": plotting.plot_boundary_evolution(detections),
        "lake_boundary_evolution_zoomed.jpg": plotting.plot_boundary_evolution(
            detections, zoom_padding=30
        ),
        "fragmentation_diagnostic.jpg": plotting.plot_fragmentation(detections),
    }


def write_results(detections: dict[int, LakeDetection], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "year",
                "date",
                "otsu_threshold",
                "water_pixels",
                "lake_pixels",
                "pixel_area_m2",
                "area_km2",
                "boundary_uncertainty_km2",
            ]
        )
        uncertainties = uncertainties_km2(detections)
        for year, det in detections.items():
            writer.writerow(
                [
                    year,
                    det.scene.date,
                    f"{det.otsu_threshold:.4f}",
                    int(det.water_mask.sum()),
                    int(det.lake_mask.sum()),
                    f"{det.scene.pixel_area_m2:.2f}",
                    f"{det.area_km2:.4f}",
                    f"{uncertainties[year]:.4f}",
                ]
            )


def print_summary(detections: dict[int, LakeDetection]) -> None:
    uncertainties = uncertainties_km2(detections)
    print(f"{'Year':<6}{'Date':<12}{'Otsu':>8}{'Lake px':>10}{'Area (km²)':>19}")
    for year, det in detections.items():
        print(
            f"{year:<6}{det.scene.date:<12}{det.otsu_threshold:>8.4f}"
            f"{int(det.lake_mask.sum()):>10}{det.area_km2:>11.3f} ± {uncertainties[year]:.3f}"
        )

    areas = {year: det.area_km2 for year, det in detections.items()}
    print()
    for change in metrics.area_changes(areas, uncertainties):
        print(
            f"{change.start}–{change.end}: {change.absolute_km2:+.3f} ± "
            f"{change.uncertainty_km2:.3f} km² ({change.percent:+.2f}%)"
        )
    dates = {year: det.scene.date for year, det in detections.items()}
    print(f"Compound annual growth rate: {metrics.compound_annual_growth_rate(areas, dates):.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    detections = {
        year: detect_lake(load_scene(date, args.data_dir))
        for year, date in ACQUISITION_DATES.items()
    }
    print_summary(detections)

    for filename, fig in build_figures(detections).items():
        plotting.save_figure(fig, filename, args.figures_dir)
        plt.close(fig)
    write_results(detections, args.results)

    print(f"\nFigures saved to {args.figures_dir}")
    print(f"Results saved to {args.results}")


if __name__ == "__main__":
    main()
