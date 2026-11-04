"""Regression test on the real Sentinel-2 data, skipped when it has not been downloaded."""

import csv
from pathlib import Path

import pytest

from glacial_lake.config import ACQUISITION_DATES, BANDS, band_path
from glacial_lake.io import load_scene
from glacial_lake.pipeline import detect_lake

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results" / "lake_areas.csv"

have_data = all(
    band_path(date, band, DATA_DIR).exists()
    for date in ACQUISITION_DATES.values()
    for band in BANDS
)


@pytest.mark.skipif(not have_data, reason="real data not downloaded (scripts/download_data.py)")
def test_results_match_the_committed_table():
    with RESULTS.open(newline="") as f:
        expected = {int(row["year"]): row for row in csv.DictReader(f)}

    for year, date in ACQUISITION_DATES.items():
        detection = detect_lake(load_scene(date, DATA_DIR))
        assert int(detection.lake_mask.sum()) == int(expected[year]["lake_pixels"])
        assert int(detection.water_mask.sum()) == int(expected[year]["water_pixels"])
        assert detection.otsu_threshold == pytest.approx(
            float(expected[year]["otsu_threshold"]), abs=1e-4
        )
