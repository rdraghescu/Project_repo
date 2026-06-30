"""
Validare artefacte joblib inainte de build Docker / deploy.

Inlocuieste vechiul flux MLflow iris_elasticnet: acum verificam fisierele
Random Forest ENHANCED salvate cu joblib in folderul models/.
"""

from __future__ import annotations

import sys
from pathlib import Path

from model_loader import get_artifact_paths, load_artifacts

PIPELINE_DIR = Path(__file__).resolve().parent


def main() -> int:
    models_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PIPELINE_DIR / "models"
    paths = get_artifact_paths(models_dir)

    print("Verificare artefacte Random Forest ENHANCED (joblib)...")
    for label, path in paths.items():
        print(f"  - {label}: {path} {'OK' if path.exists() else 'LIPSA'}")

    model, scaler, metadata = load_artifacts(models_dir)

    print("\n[OK] Model incarcat cu joblib")
    print(f"     Tip: {metadata.get('model_type')}")
    print(f"     Versiune: {metadata.get('model_version')}")
    print(f"     Features: {metadata.get('features_total', len(metadata.get('features_all', [])))}")
    print(f"     Scaler: {type(scaler).__name__}")
    print(f"     Estimator: {type(model).__name__}, n_estimators={getattr(model, 'n_estimators', 'N/A')}")

    print("\nDeploy local (fara Docker):")
    print("  cd Docker/ci_cd_pipeline_docker && MODELS_DIR=models python serve_model.py")
    print("\nDeploy cu Docker:")
    print("  docker build -f Docker/ci_cd_pipeline_docker/Dockerfile -t web-traffic-rf:latest .")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
