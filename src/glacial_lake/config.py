"""Study-site configuration: area of interest, acquisition dates, bands and paths."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"

# Imja Tsho and its surroundings (Sagarmatha National Park, Nepal) in
# WGS 84 / UTM zone 45N. Bounds are aligned to the 10 m Sentinel-2 pixel grid.
CRS_EPSG = 32645
BBOX = (489_120, 3_082_640, 497_070, 3_088_860)  # (min_x, min_y, max_x, max_y) in metres
RESOLUTION_M = 10

ACQUISITION_DATES: dict[int, str] = {
    2018: "2018-10-30",
    2021: "2021-11-08",
    2024: "2024-10-13",
}
YEARS = tuple(ACQUISITION_DATES)

# B02 blue, B03 green, B04 red, B08 near-infrared, B11 shortwave-infrared (1.6 µm)
BANDS = ("B02", "B03", "B04", "B08", "B11")

# Sentinel-2 L2A digital numbers are surface reflectance multiplied by this factor.
REFLECTANCE_SCALE = 10_000


def band_path(date: str, band: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / date / f"{band}.tif"
