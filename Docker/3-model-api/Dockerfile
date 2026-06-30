# Dockerfile pentru Model API
FROM python:3.13.9-slim

# Instalează dependențe sistem
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Setează directorul de lucru
WORKDIR /app

# Copiază requirements
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copiază modelul, scaler și metadata
COPY ../../models/linear_regression_model_v2.pkl models/
COPY ../../models/web_traffic_scaler_v2.pkl models/
COPY ../../models/model_metadata_v2.json models/

# Copiază aplicația API
COPY app.py .

# Expune portul 8000
EXPOSE 8000

# Variabile de mediu
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=models/linear_regression_model_v2.pkl
ENV SCALER_PATH=models/web_traffic_scaler_v2.pkl
ENV METADATA_PATH=models/model_metadata_v2.json

# Pornește API-ul
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
