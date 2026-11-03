import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from glacial_lake.io import load_scene

DATE = "2000-01-01"
TRANSFORM = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 3_000_000.0)


def write_band(path, values, transform=TRANSFORM):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="uint16",
        crs=CRS.from_epsg(32645),
        transform=transform,
    ) as dst:
        dst.write(values.astype(np.uint16), 1)


def test_load_scene_converts_to_reflectance(tmp_path):
    dn = np.array([[0, 2_500], [10_000, 12_000]])
    for band in ("B03", "B08"):
        write_band(tmp_path / DATE / f"{band}.tif", dn)

    scene = load_scene(DATE, tmp_path, bands=("B03", "B08"))

    np.testing.assert_allclose(scene.bands["B03"], [[0.0, 0.25], [1.0, 1.0]])  # clipped to 1
    assert scene.shape == (2, 2)
    assert scene.pixel_size_m == (10.0, 10.0)
    assert scene.pixel_area_m2 == 100.0
    assert scene.crs.to_epsg() == 32645


def test_load_scene_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_scene(DATE, tmp_path, bands=("B03",))


def test_load_scene_rejects_mismatched_grids(tmp_path):
    values = np.ones((2, 2))
    write_band(tmp_path / DATE / "B03.tif", values)
    write_band(
        tmp_path / DATE / "B08.tif",
        values,
        transform=Affine(10.0, 0.0, 500_010.0, 0.0, -10.0, 3_000_000.0),
    )
    with pytest.raises(ValueError, match="same grid"):
        load_scene(DATE, tmp_path, bands=("B03", "B08"))
