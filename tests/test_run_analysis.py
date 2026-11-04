"""Checks the analysis script on synthetic scenes."""

import csv
import importlib.util
from pathlib import Path

import pytest

from glacial_lake.pipeline import detect_lake

from conftest import make_scene

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_analysis.py"
spec = importlib.util.spec_from_file_location("run_analysis", SCRIPT)
run_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_analysis)


@pytest.fixture
def detections():
    return {
        2018: detect_lake(make_scene(date="2018-10-30")),
        2021: detect_lake(make_scene(include_snow=False, date="2021-11-08")),
    }


def test_default_paths_are_in_the_repository():
    repo = SCRIPT.parents[1]
    assert repo / "data" == run_analysis.DATA_DIR
    assert repo / "figures" == run_analysis.FIGURES_DIR
    assert repo / "results" / "lake_areas.csv" == run_analysis.DEFAULT_RESULTS_PATH


def test_write_results(detections, tmp_path):
    path = tmp_path / "results" / "areas.csv"
    run_analysis.write_results(detections, path)

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["year"] for row in rows] == ["2018", "2021"]
    assert rows[0]["date"] == "2018-10-30"
    assert rows[0]["water_pixels"] == "809"  # lake + pond, snow rejected
    assert rows[0]["lake_pixels"] == "800"
    assert rows[0]["pixel_area_m2"] == "100.00"
    assert float(rows[0]["area_km2"]) == pytest.approx(0.08)
    assert float(rows[0]["boundary_uncertainty_km2"]) > 0


def test_print_summary(detections, capsys):
    run_analysis.print_summary(detections)
    out = capsys.readouterr().out
    assert "0.080" in out
    assert "2018–2021" in out
    assert "Compound annual growth rate: 0.00%" in out
