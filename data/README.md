# Data

The satellite imagery is not stored in this repository. Download it with:

```bash
export CDSE_CLIENT_ID=...
export CDSE_CLIENT_SECRET=...
uv run python scripts/download_data.py
```

On Windows (PowerShell), use `$env:CDSE_CLIENT_ID = "..."` and `$env:CDSE_CLIENT_SECRET = "..."`. Use `--data-dir` to write elsewhere and `--force` to re-download existing files.

This writes 15 GeoTIFFs (~11 MB) to this folder:

```
data/
├── 2018-10-30/{B02,B03,B04,B08,B11}.tif
├── 2021-11-08/...
└── 2024-10-13/...
```

## Getting CDSE credentials

1. Create a free account on the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/).
2. Open the [Sentinel Hub dashboard](https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings), go to **User settings → OAuth clients**, and create a client.
3. Export the client ID and secret as shown above. The secret is shown only once.

## Contents

| Property | Value |
| --- | --- |
| Product | Sentinel-2 Level-2A (bottom-of-atmosphere surface reflectance) |
| Acquisition dates | 2018-10-30, 2021-11-08, 2024-10-13 |
| Area of interest | Imja Tsho, Sagarmatha National Park, Nepal (~86.89–86.97° E, 27.87–27.92° N) |
| CRS / grid | WGS 84 / UTM zone 45N (EPSG:32645), 10 m pixels, 795 × 622 px |
| Bounding box (UTM, m) | 489 120, 3 082 640, 497 070, 3 088 860 |
| Bands | B02 (blue), B03 (green), B04 (red), B08 (NIR), B11 (SWIR 1.6 µm) |
| Values | uint16, surface reflectance × 10 000 |

Processing notes:

- **Harmonised values.** Products from processing baseline 04.00 onward (all acquisitions since January 2022, and any older product that has been reprocessed) carry a −1000 radiometric offset. The download requests `harmonizeValues`, which removes that offset wherever it applies, so all three dates use the same scale. Negative values are clamped to 0.
- **B11 resampling.** B11 is natively 20 m. It is upsampled to the 10 m grid with nearest-neighbour, so no values are interpolated.
- **Scene selection.** If several tiles cover the area on a date, they are mosaicked with `leastCC`: each pixel comes from the tile with the lowest cloud cover in its metadata (cloud cover of the whole ~110 km tile, not of this area).
- **No data.** Pixels without data are 0 in every band. The analysis excludes them; these three scenes have none.

## Attribution

Contains modified Copernicus Sentinel data (2018, 2021, 2024), processed by the
Copernicus Data Space Ecosystem. Use of Sentinel data is governed by the
[Copernicus Sentinel data terms](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice).
