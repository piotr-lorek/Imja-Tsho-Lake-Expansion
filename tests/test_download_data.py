"""Checks the download request without contacting the API."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
download_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_data)


def test_default_data_dir_is_in_the_repository():
    assert SCRIPT.parents[1] / "data" == download_data.DATA_DIR


def test_output_grid_is_10_m():
    assert (download_data.WIDTH, download_data.HEIGHT) == (795, 622)


def test_build_request():
    request = download_data.build_request("2024-10-13", "B11")
    data = request["input"]["data"][0]

    assert request["input"]["bounds"]["bbox"] == list(download_data.BBOX)
    assert request["input"]["bounds"]["properties"]["crs"].endswith("/32645")
    assert data["type"] == "sentinel-2-l2a"
    assert data["dataFilter"]["timeRange"] == {
        "from": "2024-10-13T00:00:00Z",
        "to": "2024-10-13T23:59:59Z",
    }
    assert data["processing"]["harmonizeValues"] is True
    assert (request["output"]["width"], request["output"]["height"]) == (795, 622)
    assert 'bands: ["B11"]' in request["evalscript"]
    assert "sample.B11" in request["evalscript"]


def test_missing_credentials_exit_with_error(monkeypatch, capsys):
    monkeypatch.delenv("CDSE_CLIENT_ID", raising=False)
    monkeypatch.delenv("CDSE_CLIENT_SECRET", raising=False)
    monkeypatch.setattr("sys.argv", ["download_data.py"])
    assert download_data.main() == 1
    assert "CDSE_CLIENT_ID" in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["CDSE_CLIENT_ID", "CDSE_CLIENT_SECRET"])
def test_partial_credentials_exit_with_error(monkeypatch, missing):
    monkeypatch.setenv("CDSE_CLIENT_ID", "id")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "secret")
    monkeypatch.delenv(missing)
    monkeypatch.setattr("sys.argv", ["download_data.py"])
    assert download_data.main() == 1


def test_session_retries_rate_limits_and_server_errors():
    session = download_data.make_session("token")
    retry = session.get_adapter("https://sh.dataspace.copernicus.eu").max_retries
    assert session.headers["Authorization"] == "Bearer token"
    assert retry.total > 0
    assert {429, 503} <= set(retry.status_forcelist)
    assert retry.is_retry("POST", 429)
