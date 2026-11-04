"""Loading Sentinel-2 bands as surface reflectance."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from .config import BANDS, REFLECTANCE_SCALE, band_path


@dataclass(frozen=True)
class Scene:
    """Co-registered surface-reflectance bands of a single acquisition."""

    date: str
    bands: dict[str, np.ndarray]
    transform: Affine
    crs: CRS

    @property
    def shape(self) -> tuple[int, int]:
        return next(iter(self.bands.values())).shape

    @property
    def valid_mask(self) -> np.ndarray:
        """Pixels with data. Sentinel Hub fills pixels without data with 0 in every band."""
        return np.any([band > 0 for band in self.bands.values()], axis=0)

    @property
    def pixel_size_m(self) -> tuple[float, float]:
        """Pixel (width, height) in metres."""
        return abs(self.transform.a), abs(self.transform.e)

    @property
    def pixel_area_m2(self) -> float:
        width, height = self.pixel_size_m
        return width * height


def load_scene(date: str, data_dir: Path, bands: Iterable[str] = BANDS) -> Scene:
    """Read the given bands for one date and convert them to reflectance in [0, 1].

    Bands are read from ``data_dir/<date>/<band>.tif``, as written by
    ``scripts/download_data.py``. All bands must share the same grid.
    """
    reflectance: dict[str, np.ndarray] = {}
    transform = crs = None

    for band in bands:
        path = band_path(date, band, data_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `uv run python scripts/download_data.py` first."
            )
        with rasterio.open(path) as src:
            if transform is None:
                transform, crs = src.transform, src.crs
            elif src.transform != transform or src.crs != crs:
                raise ValueError(f"{path} is not on the same grid as the other bands of {date}")
            dn = src.read(1)
        reflectance[band] = np.clip(dn.astype(np.float64) / REFLECTANCE_SCALE, 0.0, 1.0)

    return Scene(date=date, bands=reflectance, transform=transform, crs=crs)
