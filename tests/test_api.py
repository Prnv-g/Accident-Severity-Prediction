from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import httpx
import joblib
import numpy as np
import pandas as pd
import pytest
from model_inference import (
    AccidentSeverityModel,
    DEFAULT_MODEL_PATH,
    RAW_FEATURES,
    make_features,
)


EXAMPLE_RECORD = {
    "Start_Time": "2022-08-15 17:30:00",
    "Street": "I-95 N",
    "Start_Lat": 40.7128,
    "Start_Lng": -74.0060,
    "Temperature(F)": 78.0,
    "Humidity(%)": 68.0,
    "Pressure(in)": 29.85,
    "Visibility(mi)": 10.0,
    "Wind_Speed(mph)": 8.0,
    "Precipitation(in)": 0.0,
    "Amenity": False,
    "Bump": False,
    "Crossing": False,
    "Give_Way": False,
    "Junction": True,
    "No_Exit": False,
    "Railway": False,
    "Roundabout": False,
    "Station": False,
    "Stop": False,
    "Traffic_Calming": False,
    "Traffic_Signal": False,
    "State": "NY",
    "Weather_Condition": "Fair",
    "Sunrise_Sunset": "Day",
    "Civil_Twilight": "Day",
}


@pytest.fixture(scope="module")
def client():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    project_dir = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=project_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Uvicorn exited before the API became ready.")
            try:
                with urlopen(f"{base_url}/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except (OSError, URLError):
                time.sleep(0.2)
        else:
            raise RuntimeError("Timed out waiting for the test API to start.")

        with httpx.Client(base_url=base_url, timeout=30) as test_client:
            yield test_client
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_model_loading_and_contract():
    model = AccidentSeverityModel()
    assert model.model_path == DEFAULT_MODEL_PATH
    assert model.raw_input_columns == RAW_FEATURES
    assert model.class_labels == [1, 2, 3, 4]
    assert model.probabilities_supported is True


def test_health_and_model_info(client: httpx.Client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_ready"] is True

    info = client.get("/model/info")
    assert info.status_code == 200
    body = info.json()
    assert body["required_inputs"] == RAW_FEATURES
    assert body["supported_severity_levels"] == [1, 2, 3, 4]
    assert body["evaluation_metrics"]["validation_metrics"]

    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "Swagger UI" in docs.text

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert {"/health", "/model/info", "/predict", "/predict/batch"}.issubset(
        openapi.json()["paths"]
    )


def test_single_prediction_response(client: httpx.Client):
    response = client.post("/predict", json=EXAMPLE_RECORD)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_severity"] in [1, 2, 3, 4]
    assert set(body["model_scores"]) == {"1", "2", "3", "4"}
    assert sum(body["model_scores"].values()) == pytest.approx(1.0)


def test_api_matches_notebook_prediction_logic(client: httpx.Client):
    bundle = joblib.load(DEFAULT_MODEL_PATH)
    features = make_features(pd.DataFrame([EXAMPLE_RECORD]))
    scores = bundle["model"].predict_proba(features)
    expected = int((scores * bundle["score_multipliers"]).argmax(axis=1)[0] + 1)

    response = client.post("/predict", json=EXAMPLE_RECORD)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_severity"] == expected
    assert np.asarray(list(body["model_scores"].values())) == pytest.approx(scores[0])


def test_batch_prediction_and_single_consistency(client: httpx.Client):
    second = deepcopy(EXAMPLE_RECORD)
    second.update(
        {
            "Start_Time": "2022-09-03 07:45:00",
            "Street": "Main Street",
            "Start_Lat": 34.0522,
            "Start_Lng": -118.2437,
            "Temperature(F)": None,
            "Precipitation(in)": None,
            "State": "CA",
            "Weather_Condition": "Light Rain",
            "Traffic_Signal": True,
        }
    )
    batch = client.post(
        "/predict/batch", json={"records": [EXAMPLE_RECORD, second]}
    )
    assert batch.status_code == 200
    body = batch.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2

    single = client.post("/predict", json=EXAMPLE_RECORD)
    assert body["predictions"][0] == single.json()


@pytest.mark.parametrize(
    "field,value",
    [
        ("Start_Time", "not-a-date"),
        ("Temperature(F)", "hot"),
        ("Junction", "sometimes"),
    ],
)
def test_invalid_requests_return_422(client: httpx.Client, field: str, value):
    invalid = deepcopy(EXAMPLE_RECORD)
    invalid[field] = value
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_missing_required_field_returns_422(client: httpx.Client):
    incomplete = deepcopy(EXAMPLE_RECORD)
    del incomplete["State"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_malformed_batch_returns_422(client: httpx.Client):
    response = client.post("/predict/batch", json={"records": []})
    assert response.status_code == 422


def test_joblib_and_pkl_artifacts_produce_identical_results():
    features = make_features(pd.DataFrame([EXAMPLE_RECORD]))
    joblib_bundle = joblib.load(DEFAULT_MODEL_PATH)
    pkl_bundle = joblib.load(DEFAULT_MODEL_PATH.with_suffix(".pkl"))
    joblib_scores = joblib_bundle["model"].predict_proba(features)
    pkl_scores = pkl_bundle["model"].predict_proba(features)
    assert np.array_equal(joblib_scores, pkl_scores)
    assert np.array_equal(
        joblib_bundle["score_multipliers"], pkl_bundle["score_multipliers"]
    )
