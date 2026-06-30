# Dockerfile pentru MLflow Tracking Server
FROM python:3.13-slim

# Setează directorul de lucru
WORKDIR /mlflow

# Instalează MLflow
RUN pip install --no-cache-dir mlflow==3.12.0 psycopg2-binary

# Creează director pentru artifacts
RUN mkdir -p /mlflow/artifacts

# Expune portul 5000
EXPOSE 5000

# Pornește serverul MLflow
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts"]
