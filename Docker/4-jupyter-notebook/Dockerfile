# Dockerfile pentru Jupyter Notebook Containerizat
FROM python:3.13.9-slim

# Instalează dependențe sistem
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Setează directorul de lucru
WORKDIR /workspace

# Instalează Jupyter și dependențe ML
RUN pip install --no-cache-dir \
    jupyter==1.1.1 \
    jupyterlab==4.3.4 \
    ipykernel==7.2.0 \
    pandas==2.3.3 \
    numpy==2.4.6 \
    scikit-learn==1.8.0 \
    mlflow==3.12.0 \
    matplotlib==3.10.9 \
    seaborn==0.13.2

# Expune portul Jupyter
EXPOSE 8888

# Variabile de mediu
ENV PYTHONUNBUFFERED=1
ENV JUPYTER_ENABLE_LAB=yes

# Comandă pentru pornire Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
