"""Download the Sentinel-2 L2A bands used in the Imja Tsho analysis.

Data is fetched from the Copernicus Data Space Ecosystem (CDSE) Sentinel Hub
Process API as single-band GeoTIFFs containing L2A surface reflectance scaled
by 10 000 (uint16), on the native Sentinel-2 10 m UTM grid.

Requires a free CDSE account and an OAuth client, see data/README.md:

    export CDSE_CLIENT_ID=...
    export CDSE_CLIENT_SECRET=...
    uv run python scripts/download_data.py

Files are written to data/<acquisition-date>/<band>.tif and existing files are
skipped unless --force is given.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import rasterio
import requests

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Area of interest: Imja Tsho and its surroundings (Sagarmatha National Park, Nepal),
# in WGS 84 / UTM zone 45N. Bounds are aligned to the 10 m Sentinel-2 pixel grid.
CRS_EPSG = 32645
BBOX = (489_120, 3_082_640, 497_070, 3_088_860)  # (min_x, min_y, max_x, max_y) in metres
RESOLUTION_M = 10
WIDTH = round((BBOX[2] - BBOX[0]) / RESOLUTION_M)  # 795 px
HEIGHT = round((BBOX[3] - BBOX[1]) / RESOLUTION_M)  # 622 px

ACQUISITION_DATES = ("2018-10-30", "2021-11-08", "2024-10-13")
BANDS = ("B02", "B03", "B04", "B08", "B11")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

EVALSCRIPT_TEMPLATE = """//VERSION=3
function setup() {{
  return {{
    input: [{{ bands: ["{band}"], units: "DN" }}],
    output: {{ bands: 1, sampleType: "UINT16" }}
  }};
}}
function evaluatePixel(sample) {{
  return [sample.{band}];
}}
"""


def get_access_token(client_id: str, client_secret: str) -> str:
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def build_request(date: str, band: str) -> dict:
    return {
        "input": {
            "bounds": {
                "bbox": list(BBOX),
                "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{CRS_EPSG}"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date}T00:00:00Z",
                            "to": f"{date}T23:59:59Z",
                        },
                        "mosaickingOrder": "leastCC",
                    },
                    "processing": {
                        # Remove the -1000 offset introduced in processing baseline 04.00
                        # (Jan 2022) so all dates share the same DN = reflectance * 10000 scale.
                        "harmonizeValues": True,
                        # B11 is natively 20 m; nearest-neighbour keeps the original values.
                        "upsampling": "NEAREST",
                    },
                }
            ],
        },
        "output": {
            "width": WIDTH,
            "height": HEIGHT,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT_TEMPLATE.format(band=band),
    }


def download_band(session: requests.Session, date: str, band: str, out_path: Path) -> None:
    response = session.post(
        PROCESS_URL,
        json=build_request(date, band),
        headers={"Accept": "image/tiff"},
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(
            f"Request for {date} {band} failed ({response.status_code}): {response.text}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tif.part")
    tmp_path.write_bytes(response.content)
    validate(tmp_path)
    tmp_path.replace(out_path)


def validate(path: Path) -> None:
    with rasterio.open(path) as src:
        problems = []
        if (src.width, src.height) != (WIDTH, HEIGHT):
            problems.append(f"size {src.width}x{src.height}, expected {WIDTH}x{HEIGHT}")
        if src.crs is None or src.crs.to_epsg() != CRS_EPSG:
            problems.append(f"CRS {src.crs}, expected EPSG:{CRS_EPSG}")
        if src.read(1).max() == 0:
            problems.append("image is empty (no acquisition on this date?)")
    if problems:
        raise RuntimeError(f"{path.name}: " + "; ".join(problems))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"output directory (default: {DEFAULT_DATA_DIR.relative_to(REPO_ROOT)}/)",
    )
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        print(
            "error: set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET (see data/README.md)",
            file=sys.stderr,
        )
        return 1

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {get_access_token(client_id, client_secret)}"

    for date in ACQUISITION_DATES:
        for band in BANDS:
            out_path = args.data_dir / date / f"{band}.tif"
            if out_path.exists() and not args.force:
                print(f"skip      {out_path.relative_to(args.data_dir)} (exists)")
                continue
            download_band(session, date, band, out_path)
            print(f"download  {out_path.relative_to(args.data_dir)}")

    print(f"Done. Data is in {args.data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
