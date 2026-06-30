# Dockerfile pentru Mediu Reproducibil Complet
FROM python:3.13.9-slim

# Instalează dependențe sistem pentru matplotlib și scikit-learn
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Setează directorul de lucru
WORKDIR /app

# Copiază requirements.txt
COPY ../../requirements.txt .

# Instalează dependențele Python (versiuni exacte)
RUN pip install --no-cache-dir -r requirements.txt

# Copiază întregul proiect
COPY ../.. .

# Creează directoare pentru output
RUN mkdir -p visualizations models data/predictions mlruns

# Variabile de mediu
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=./mlruns

# Comandă default: rulează training script
CMD ["python", "sources/mlflow_ml_model_web_traffic.py"]
