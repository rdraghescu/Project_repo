"""
Incarcare artefacte Random Forest ENHANCED salvate cu joblib + metadata JSON.
Caile pot fi suprascrise prin variabile de mediu (util in Docker / CI).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib

# Fisiere implicite (relative la MODELS_DIR)
DEFAULT_MODEL_FILE = "random_forest_model_v2_BEST.joblib"
DEFAULT_SCALER_FILE = "web_traffic_scaler_v2.joblib"
DEFAULT_METADATA_FILE = "model_metadata_v2.json"

# get_models_dir() este o functie care returneaza directorul unde sunt stocate modelul, scaler-ul si metadata.
# Path este o clasa in Python care reprezinta un drum (cale) in sistemul de fisiere.
def get_models_dir() -> Path: 
    """Directorul unde sunt stocate modelul, scaler-ul si metadata."""
    return Path(os.environ.get("MODELS_DIR", "models"))


# get_artifact_paths() este o functie care returneaza caile complete catre cele 3 artefacte.
# semnul -> inseamna ca functia returneaza un dictionar cu cheile "model", "scaler" si "metadata".
def get_artifact_paths(models_dir: Path | None = None) -> dict[str, Path]:
    """Rezolva caile complete catre cele 3 artefacte."""
    base = models_dir or get_models_dir()
    return {
        "model": base / os.environ.get("MODEL_FILE", DEFAULT_MODEL_FILE),
        "scaler": base / os.environ.get("SCALER_FILE", DEFAULT_SCALER_FILE),
        "metadata": base / os.environ.get("METADATA_FILE", DEFAULT_METADATA_FILE),
    }

# load_artifacts() este o functie care incarca modelul, scaler-ul si metadata.
# semnul -> inseamna ca functia returneaza un tuplu cu 3 elemente: model, scaler si metadata.
# Any este un tip generic in Python care poate fi orice tip de date.
# dict este un tip de date care reprezinta un dictionar in Python.
# tuple este un tip de date care reprezinta un tuplu in Python.
# FileNotFoundError este o exceptie care este aruncata cand un fisier nu este gasit.
# paths este un dictionar cu cheile "model", "scaler" si "metadata".
# name este o cheie din dictionar.
# path este o cale completa catre artefact.
def load_artifacts(models_dir: Path | None = None) -> tuple[Any, Any, dict]:
    """
    Incarca modelul RF, StandardScaler si metadata JSON.

    Returns:
        (model, scaler, metadata_dict)
    """
    paths = get_artifact_paths(models_dir)

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Artefact lipsa ({name}): {path}. "
                "Ruleaza antrenarea (mlflow_ml_model_web_traffic.py) sau prepare_ci_models.py."
            )

    model = joblib.load(paths["model"])
    scaler = joblib.load(paths["scaler"])

    with open(paths["metadata"], encoding="utf-8") as handle:
        metadata = json.load(handle)

    return model, scaler, metadata
