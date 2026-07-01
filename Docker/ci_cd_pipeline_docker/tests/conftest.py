"""
Fixtures pytest: artefacte joblib temporare pentru teste GitHub Actions.
"""

from __future__ import annotations # __future__ este un modul in Python care permite utilizarea de functii si clase din viitoare versiuni ale limbajului.

import json
import sys
from pathlib import Path # Path este o clasa in Python care reprezinta un drum (cale) in sistemul de fisiere.

PIPELINE_DIR = Path(__file__).resolve().parents[1] # Path(__file__).resolve().parents[1] este o cale absoluta catre directorul curent. 1 din parents inseamna un nivel mai sus in structura de directoare.
PROJECT_ROOT = PIPELINE_DIR.parent.parent # Project_ROOT este directorul root al proiectului.
METADATA_SOURCE = PROJECT_ROOT / "models" / "model_metadata_v2.json"

# Permite importul modulelor din Docker/ci_cd_pipeline_docker/ in pytest
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import joblib # joblib este o librarie in Python care permite incarcarea si salvarea modelelor in format joblib.
import pytest # pytest este o librarie in Python care permite testarea codului.
from metadata_utils import get_features_all
from sklearn.ensemble import RandomForestRegressor # RandomForestRegressor este o clasa in Python care reprezinta un model Random Forest.
from sklearn.preprocessing import StandardScaler # StandardScaler este o clasa in Python care standardizeaza datele.
git add 

@pytest.fixture(scope="session")
def metadata() -> dict:
    """Incarca metadata reala din repo (JSON versionat in Git)."""
    with open(METADATA_SOURCE, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture()
def models_dir(tmp_path: Path, metadata: dict) -> Path:
    """
    Creeaza un folder models/ temporar cu RF + scaler joblib compatibile.
    Nu depinde de fisierele .joblib mari din repo (pot lipsi pe GitHub).
    """
    import numpy as np

    features_all = get_features_all(metadata)
    n_features = len(features_all)

    rng = np.random.default_rng(42)
    x_train = rng.uniform(0, 10, size=(120, n_features))
    y_train = rng.uniform(50, 500, size=120)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)

    model = RandomForestRegressor(n_estimators=5, max_depth=4, random_state=42)
    model.fit(x_scaled, y_train)

    out = tmp_path / "models"
    out.mkdir()
    joblib.dump(model, out / "random_forest_model_v2_BEST.joblib")
    joblib.dump(scaler, out / "web_traffic_scaler_v2.joblib")
    with open(out / "model_metadata_v2.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)

    return out


@pytest.fixture()
def sample_payload() -> dict:
    """Input API reprezentativ pentru predictie."""
    return {
        "pageviews": 5,
        "visitNumber": 2,
        "hits": 12,
        "device_category": "desktop",
        "country": "United States",
    }
