# 🚀 Varianta 3: Model API - REST API pentru Predicții

## 🎯 Ce face?

Expune modelul ML ca REST API cu FastAPI:
- ✅ Endpoint `/predict` pentru predicții individuale
- ✅ Endpoint `/batch-predict` pentru predicții multiple
- ✅ Endpoint `/model-info` pentru informații model
- ✅ Documentație Swagger automată
- ✅ Validare input cu Pydantic
- ✅ Production-ready cu Uvicorn

**Model servit ca microservice!**

---

## 🚀 Cum să pornești

### **Pas 1: Asigură-te că ai modelul antrenat**
```powershell
# Verifică că există:
# - models/linear_regression_model_v2.pkl
# - models/web_traffic_scaler_v2.pkl
# - models/model_metadata_v2.json

# Dacă nu există, rulează training:
cd C:\Users\User\Desktop\repos\Project_repo
python sources/mlflow_ml_model_web_traffic.py
```

### **Pas 2: Deschide folderul API în terminal**
```powershell
cd C:\Users\User\Desktop\repos\Project_repo\docker\3-model-api
```

### **Pas 3: Build containerul**
```powershell
docker-compose build
```

### **Pas 4: Pornește API-ul**
```powershell
docker-compose up
```

**Așteptă să vezi:**
```
web-traffic-model-api | INFO: Application startup complete.
web-traffic-model-api | INFO: Uvicorn running on http://0.0.0.0:8000
```

### **Pas 5: Accesează API-ul**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/

---

## 📡 Endpoints Disponibile

### **1. GET `/` - Health Check**
```powershell
curl http://localhost:8000/
```

**Response:**
```json
{
  "status": "online",
  "api": "Web Traffic Prediction API",
  "version": "2.0",
  "model": "LinearRegression",
  "features_count": 20
}
```

### **2. POST `/predict` - Predicție Individuală**
```powershell
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"pageviews\": 5, \"visitNumber\": 3, \"hits\": 12, \"device_category\": \"desktop\", \"country\": \"United States\"}"
```

**Response:**
```json
{
  "predicted_time_on_site_seconds": 245.67,
  "predicted_time_on_site_minutes": 4.09,
  "prediction_timestamp": "2026-05-30T12:30:45",
  "model_version": "v2.0_ENHANCED",
  "features_used": ["pageviews", "visitNumber", "hits", "device_category_mobile", "device_category_tablet", "..."]
}
```

### **3. POST `/batch-predict` - Predicții Multiple**
```powershell
curl -X POST http://localhost:8000/batch-predict ^
  -H "Content-Type: application/json" ^
  -d "{\"samples\": [{\"pageviews\": 5, \"visitNumber\": 3, \"hits\": 12, \"device_category\": \"desktop\", \"country\": \"United States\"}, {\"pageviews\": 2, \"visitNumber\": 1, \"hits\": 5, \"device_category\": \"mobile\", \"country\": \"India\"}]}"
```

**Response:**
```json
{
  "predictions": [245.67, 89.34],
  "count": 2,
  "average_time": 167.50
}
```

### **4. GET `/model-info` - Informații Model**
```powershell
curl http://localhost:8000/model-info
```

**Response:**
```json
{
  "model_type": "LinearRegression",
  "version": "v2.0_ENHANCED",
  "features_count": 20,
  "performance": {
    "r2_score": 0.4651,
    "mae": 111.48,
    "rmse": 271.43
  },
  "top_features": [...]
}
```

---

## 🌐 Testare în Browser

### **Swagger UI (Interfață Interactivă)**

1. Deschide: http://localhost:8000/docs
2. Click pe `/predict` → "Try it out"
3. Editează JSON-ul cu datele tale:
```json
{
  "pageviews": 7,
  "visitNumber": 2,
  "hits": 15,
  "device_category": "mobile",
  "country": "Canada"
}
```
4. Click "Execute"
5. Vezi predicția în Response body

---

## 🧪 Testare cu Python

### **Script de test:**
```python
import requests
import json

# URL API
API_URL = "http://localhost:8000"

# Test predicție individuală
data = {
    "pageviews": 5,
    "visitNumber": 3,
    "hits": 12,
    "device_category": "desktop",
    "country": "United States"
}

response = requests.post(f"{API_URL}/predict", json=data)
print(f"Status: {response.status_code}")
print(f"Predicție: {response.json()}")

# Test batch
batch_data = {
    "samples": [
        {"pageviews": 5, "visitNumber": 3, "hits": 12, "device_category": "desktop", "country": "United States"},
        {"pageviews": 2, "visitNumber": 1, "hits": 5, "device_category": "mobile", "country": "India"}
    ]
}

response = requests.post(f"{API_URL}/batch-predict", json=batch_data)
print(f"Batch predicții: {response.json()}")
```

---

## 🔧 Comenzi Utile

### **Pornește în background**
```powershell
docker-compose up -d
```

### **Vezi logs**
```powershell
docker-compose logs -f
```

### **Oprește API-ul**
```powershell
docker-compose down
```

### **Rebuild după modificări**
```powershell
docker-compose build --no-cache
docker-compose up
```

---

## 🎓 Pentru Comisie

### **Demonstrație API:**

**1. Pornește API:** `docker-compose up -d`

**2. Arată Swagger UI:** http://localhost:8000/docs

**3. Demo predicție live:**
- Click pe `/predict` în Swagger
- "Try it out"
- Modifică valorile (pageviews, hits, etc.)
- "Execute"
- Arată predicția în Response

**4. Demo batch predicții:**
- Click pe `/batch-predict`
- Trimite 5 sample-uri simultan
- Arată că API procesează toate

**5. Info model:**
- GET `/model-info`
- Arată metrici (R²=0.4651, MAE=111.48)
- Arată top 10 features importante

### **Avantaje:**
- ✅ **Production-ready**: API REST standard
- ✅ **Scalabil**: Poate servi mii de request-uri
- ✅ **Documentat**: Swagger UI automată
- ✅ **Validare**: Pydantic verifică input-ul
- ✅ **Integrabil**: Poate fi apelat din orice limbaj/framework

---

## 🔌 Integrare cu Aplicații

### **JavaScript (Frontend)**
```javascript
const predictTime = async (data) => {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  });
  return await response.json();
};

// Folosire
const result = await predictTime({
  pageviews: 5,
  visitNumber: 3,
  hits: 12,
  device_category: "desktop",
  country: "United States"
});
console.log(`Timp prezis: ${result.predicted_time_on_site_minutes} minute`);
```

### **Mobile App (React Native)**
```javascript
import axios from 'axios';

const predict = async (features) => {
  const response = await axios.post('http://localhost:8000/predict', features);
  return response.data;
};
```

---

## ❓ Troubleshooting

### **Eroare: "port 8000 already in use"**
```powershell
# Schimbă portul în docker-compose.yml
ports:
  - "8001:8000"  # Folosește 8001 local
```

### **Eroare: "FileNotFoundError: models/..."**
- Asigură-te că ai rulat training: `python sources/mlflow_ml_model_web_traffic.py`
- Verifică că există `models/linear_regression_model_v2.pkl`

### **API nu răspunde**
```powershell
# Verifică că containerul rulează
docker ps

# Vezi logs pentru erori
docker-compose logs
```

### **Predicții ciudate**
- Verifică input-ul (pageviews >= 0, visitNumber >= 1)
- Verifică device_category (desktop/mobile/tablet)
- Verifică country (string valid)

---

## 📊 Monitoring și Logs

### **Health check automat**
Docker verifică automat dacă API-ul e sănătos:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/"]
  interval: 30s
```

### **Vezi metrici request-uri**
Logs arată:
- Număr request-uri
- Timp răspuns
- Status codes

---

## 🚀 Deploy Production

### **Pentru deploy pe server:**

**1. Schimbă host în producție:**
```python
# app.py - pentru CORS
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrânge în producție
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**2. Folosește reverse proxy (Nginx):**
```nginx
location /api {
    proxy_pass http://localhost:8000;
}
```

**3. SSL/TLS:**
Adaugă certificat HTTPS pentru securitate.

---

**✅ API production-ready gata de folosit!**
