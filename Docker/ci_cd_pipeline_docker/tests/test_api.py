"""Teste API Flask (health, model-info, predict) fara server live."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture()
def flask_client(models_dir, monkeypatch):
    """
    Reincarca serve_model cu MODELS_DIR setat la fixture-ul temporar.
    Evitam importul la nivel de modul care ar esua fara fisiere joblib.
    """
    monkeypatch.setenv("MODELS_DIR", str(models_dir))

    import serve_model

    importlib.reload(serve_model)
    serve_model.app.config["TESTING"] = True

    with serve_model.app.test_client() as client:
        yield client # yield este un operator in Python care returneaza un generator.


def test_health_endpoint(flask_client):
    response = flask_client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert "RandomForest" in body["model"]


def test_model_info_endpoint(flask_client):
    response = flask_client.get("/model-info")
    assert response.status_code == 200
    body = response.get_json()
    assert body["model_type"] == "RandomForestRegressor"
    assert body["features_total"] == 20


def test_predict_endpoint_success(flask_client, sample_payload):
    response = flask_client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    body = response.get_json()
    assert "predicted_time_on_site_seconds" in body
    assert body["predicted_time_on_site_seconds"] >= 0
    assert "predicted_time_on_site_minutes" in body


def test_predict_missing_fields(flask_client):
    response = flask_client.post("/predict", json={"pageviews": 1})
    assert response.status_code == 400
    assert "Missing fields" in response.get_json()["error"]


def test_register_model_script(models_dir):
    """register_model.py valideaza ca artefactele joblib se incarca corect."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "register_model.py", str(models_dir)],
        cwd=PIPELINE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
