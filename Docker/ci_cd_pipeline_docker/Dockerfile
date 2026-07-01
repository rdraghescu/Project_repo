# Imagine Docker pentru servirea modelului Random Forest ENHANCED (joblib).
# Build din radacina proiectului:
#   docker build -f Docker/ci_cd_pipeline_docker/Dockerfile -t web-traffic-rf:latest .

FROM python:3.11-slim

WORKDIR /app

# Instalare dependinte (strat cache-abil)
COPY Docker/ci_cd_pipeline_docker/requirements_for_container.txt .
RUN pip install --no-cache-dir -r requirements_for_container.txt

# Cod servire + feature engineering
COPY Docker/ci_cd_pipeline_docker/feature_engineering.py .
COPY Docker/ci_cd_pipeline_docker/metadata_utils.py .
COPY Docker/ci_cd_pipeline_docker/model_loader.py .
COPY Docker/ci_cd_pipeline_docker/serve_model.py .

# Artefacte model (joblib + metadata JSON) - pregatite de prepare_ci_models.py
COPY Docker/ci_cd_pipeline_docker/models/ ./models/

ENV MODELS_DIR=/app/models
ENV PORT=8501

EXPOSE 8501

# Gunicorn pentru productie; Flask dev server doar la rulare directa python serve_model.py
CMD ["gunicorn", "--bind", "0.0.0.0:8501", "--workers", "1", "--threads", "2", "serve_model:app"]
