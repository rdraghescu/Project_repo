# 📊 Varianta 1: MLflow Tracking Server

## 🎯 Ce face?

Rulează un server MLflow persistent în Docker care:
- ✅ Stochează toate experimentele ML
- ✅ Oferă UI web pentru vizualizare
- ✅ Permite comparare runs
- ✅ Păstrează datele între restarts

---

## 🚀 Cum să pornești

### **Pas 1: Deschide acest folder în terminal**
```powershell
cd C:\Users\User\Desktop\repos\Project_repo\docker\1-mlflow-tracking
```

### **Pas 2: Pornește containerul**
```powershell
docker-compose up
```

**Așteptă să vezi:**
```
mlflow-tracking-server | [INFO] Listening at: http://0.0.0.0:5000
```

### **Pas 3: Accesează MLflow UI**
Deschide browser la: **http://localhost:5000**

---

## 🔧 Comenzi Utile

### **Pornește în background (fără să blochezi terminalul)**
```powershell
docker-compose up -d
```

### **Vezi logs în timp real**
```powershell
docker-compose logs -f
```

### **Oprește serverul**
```powershell
docker-compose down
```

### **Oprește și șterge datele (ATENȚIE: șterge toate experimentele!)**
```powershell
docker-compose down -v
```

---

## 📊 Cum să loghezi experimente în acest server

### **Din scriptul Python**
```python
import mlflow

# IMPORTANT: Setează tracking URI înainte de set_experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Web_Traffic_Prediction")

with mlflow.start_run(run_name="Linear_Regression_ENHANCED"):
    mlflow.log_param("features", 20)
    mlflow.log_metric("r2_score", 0.4651)
    # ... rest of logging
```

### **Din notebook**
Adaugă la început:
```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
```

---

## 📂 Unde sunt stocate datele?

Datele sunt persistente în **Docker volume** `mlflow-data`:
- Database: `mlflow.db` (SQLite)
- Artifacts: `/mlflow/artifacts/`

**Datele rămân salvate chiar dacă oprești containerul!**

---

## 🎓 Pentru Comisie

### **Demonstrație:**
1. Pornește serverul: `docker-compose up -d`
2. Arată UI în browser: `http://localhost:5000`
3. Rulează training: `python sources/mlflow_ml_model_web_traffic.py`
4. Refresh UI → vezi noul experiment
5. Compară runs, vezi metrici, grafice

### **Avantaje:**
- ✅ **Profesional**: Tracking server centralizat
- ✅ **Persistent**: Istoricul experimentelor salvat
- ✅ **Portabil**: Rulează pe orice sistem cu Docker
- ✅ **Colaborativ**: Echipa accesează același server

---

## ❓ Troubleshooting

### **Eroare: "port 5000 already in use"**
```powershell
# Oprește procesul care folosește portul 5000
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### **Eroare: "Docker daemon not running"**
- Pornește Docker Desktop din Start Menu
- Așteaptă să vezi iconița Docker în system tray (verde)

### **Nu văd experimentele vechi**
- Verifică că folosești același volume: `docker volume ls`
- Experimentele sunt în `mlflow-data` volume

---

## 📊 Exemple de Metrici Vizibile în UI

După rularea training script, vei vedea:
- **Parametri**: features=20, test_size=0.2, random_state=42
- **Metrici**: R²=0.4651, MAE=111.48, RMSE=271.43
- **Artifacts**: model.pkl, scaler.pkl, 8 grafice PNG
- **Tags**: model=LinearRegression, version=v2.0_ENHANCED

---

**✅ Gata! Server-ul e pornit și funcțional!**
