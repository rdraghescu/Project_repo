"""
Pregateste artefactele joblib pentru teste locale, Docker si GitHub Actions.

1. Daca exista deja in ../../models/ -> le copiaza in Docker/ci_cd_pipeline_docker/models/
2. Altfel genereaza un model RF minimal compatibil (doar pentru CI / smoke tests)

Pentru productie: inlocuieste fisierele cu cele reale dupa antrenare.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent.parent
SOURCE_MODELS = PROJECT_ROOT / "models"
TARGET_MODELS = PIPELINE_DIR / "models"

ARTIFACTS = (
    "random_forest_model_v2_BEST.joblib",
    "web_traffic_scaler_v2.joblib",
    "model_metadata_v2.json",
)


def _copy_production_models() -> bool:
    """Copiaza artefactele reale din models/ daca toate cele 3 fisiere exista."""
    if not all((SOURCE_MODELS / name).exists() for name in ARTIFACTS):
        return False

    TARGET_MODELS.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        shutil.copy2(SOURCE_MODELS / name, TARGET_MODELS / name)
    print(f"[OK] Artefacte copiate din {SOURCE_MODELS} -> {TARGET_MODELS}")
    return True


def _create_synthetic_models() -> None:
    """
    Creeaza un RF mic + scaler pe date sintetice cu aceleasi 20 features.
    Folosit doar cand modelul real nu e in repo (ex. GitHub Actions fara Git LFS).
    """
    metadata_path = SOURCE_MODELS / "model_metadata_v2.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Lipseste metadata de referinta: {metadata_path}. "
            "Commit-uieste cel putin model_metadata_v2.json in models/."
        )

    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)

    features_all = metadata["features_all"]
    n_features = len(features_all)

    rng = np.random.default_rng(42)
    x_synthetic = rng.uniform(0, 10, size=(200, n_features)) # rng.uniform(0, 10, size=(200, n_features)) este o functie care genereaza 200 de numere aleatoare uniform distribuite intre 0 si 10, in dimensiunea (200, n_features).
    y_synthetic = rng.uniform(30, 600, size=200) # rng.uniform(30, 600, size=200) este o functie care genereaza 200 de numere aleatoare uniform distribuite intre 30 si 600.

    scaler = StandardScaler() # StandardScaler este o clasa in Python care standardizeaza datele.
    x_scaled = scaler.fit_transform(x_synthetic)

    model = RandomForestRegressor(
        n_estimators=10,
        max_depth=5,
        random_state=42,
        n_jobs=1,
    )
    model.fit(x_scaled, y_synthetic)

    TARGET_MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, TARGET_MODELS / ARTIFACTS[0], compress=3)
    joblib.dump(scaler, TARGET_MODELS / ARTIFACTS[1], compress=3)
    shutil.copy2(metadata_path, TARGET_MODELS / ARTIFACTS[2])

    print(
        "[WARN] Model real joblib lipsea. S-a generat un RF sintetic pentru CI/Docker. "
        "Inlocuieste cu artefactele din antrenarea reala inainte de productie."
    )


def main() -> int:
    if _copy_production_models():
        return 0
    _create_synthetic_models()
    return 0


if __name__ == "__main__":
    sys.exit(main())
