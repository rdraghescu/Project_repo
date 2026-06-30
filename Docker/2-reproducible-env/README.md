# 🔬 Varianta 2: Mediu Reproducibil Complet

## 🎯 Ce face?

Creează un container izolat cu:
- ✅ Python 3.13.9 exact
- ✅ Toate bibliotecile în versiunile corecte (pandas, numpy, scikit-learn, mlflow, matplotlib, seaborn)
- ✅ Rulează training script complet
- ✅ Generează toate output-urile (modele, grafice, metrici)

**Garanție 100% reproducibilitate pe orice sistem!**

---

## 🚀 Cum să pornești

### **Pas 1: Deschide acest folder în terminal**
```powershell
cd C:\Users\User\Desktop\repos\Project_repo\docker\2-reproducible-env
```

### **Pas 2: Build containerul (prima dată)**
```powershell
docker-compose build
```
⏱️ Durează ~2-3 minute (instalează toate dependențele)

### **Pas 3: Rulează training-ul**
```powershell
docker-compose up
```

**Vei vedea în terminal:**
```
✓ Date încărcate: 100000 rânduri
✓ După curățare: 51009 rânduri
✓ Features create: 20 (3 numeric + 12 categorical + 5 engineered)
✓ Model antrenat
✓ R² Score: 0.4651
✓ MAE: 111.48 secunde
✓ RMSE: 271.43 secunde
✓ 8 grafice generate
✓ Model salvat
```

---

## 📥 Date Necesare (input)

Înainte de rulare, asigură-te că există:

```
Project_repo/data/
├── raw/
│   └── ga-sessions.csv              # Dataset brut (obligatoriu)
└── processed/                       # Creat automat la training
    ├── X_simple.csv                 # Features SIMPLE (3 coloane)
    └── X_enhanced.csv               # Features ENHANCED (20 coloane)
```

---

## 📂 Output-uri Generate

După rulare, vei găsi în folderul principal:

### **1. Visualizări** (`visualizations/`)
- `01_target_distribution.png`
- `02_features_distribution.png`
- `03_scatter_plots.png`
- `04_descriptive_statistics.png`
- `05_train_test_comparison.png`
- `06_models_comparison.png`
- `07_residual_plot.png`
- `08_feature_coefficients.png`

### **2. Modele** (`models/`)
- `linear_regression_model_v2.pkl` (model antrenat)
- `web_traffic_scaler_v2.pkl` (StandardScaler)
- `model_metadata_v2.json` (metadata completă cu 20 features)

### **3. Validare** (`data/predictions/`)
- `validation_results.csv` (validare pe TEST SET cu ~10,200 rânduri REALE)

### **4. MLflow Tracking** (`mlruns/`)
- Experiment logs cu toate metricile

---

## 🔧 Comenzi Utile

### **Rulează training din nou**
```powershell
docker-compose up
```

### **Rebuild container (după modificări cod)**
```powershell
docker-compose build --no-cache
docker-compose up
```

### **Rulează în background**
```powershell
docker-compose up -d
```

### **Vezi logs**
```powershell
docker-compose logs -f
```

### **Oprește container**
```powershell
docker-compose down
```

---

## 🎓 Pentru Comisie

### **Demonstrație Reproducibilitate:**

**1. Rulare pe Windows:**
```powershell
cd docker\2-reproducible-env
docker-compose up
```

**2. Rulare pe Mac/Linux:**
```bash
cd docker/2-reproducible-env
docker-compose up
```

**Rezultat: Identic pe ambele sisteme!**
- ✅ Aceleași metrici (R²=0.4651, MAE=111.48)
- ✅ Aceleași grafice (pixel-perfect)
- ✅ Același model salvat

### **Avantaje:**
- ✅ **Zero setup**: Nu trebuie instalat Python, pip, pachete
- ✅ **Izolare**: Nu afectează mediul local
- ✅ **Versioning**: Docker image conține snapshot exact al mediului
- ✅ **Portabilitate**: Rulează pe orice sistem cu Docker

---

## 🔍 Verificare Versiuni Pachete

### **Vezi ce versiuni sunt instalate în container:**
```powershell
docker-compose run training pip list
```

**Output așteptat:**
```
ipykernel                7.2.0
matplotlib               3.10.9
mlflow                   3.12.0
numpy                    2.4.6
pandas                   2.3.3
scikit-learn             1.8.0
seaborn                  0.13.2
```

---

## 📊 Modificare Script

### **Dacă vrei să rulezi alt script:**

Editează `docker-compose.yml`, linia `command`:
```yaml
command: python ml-model-web-traffic1.py
```

Apoi:
```powershell
docker-compose up
```

---

## 🧪 Testare Model

### **Rulează testele sintetice:**
```powershell
docker-compose run training python sources/test_models_synthetic.py
```

### **Predicții noi:**
```powershell
docker-compose run training python sources/predict_new_data_example.py
```

---

## ❓ Troubleshooting

### **Eroare: "No such file or directory"**
- Verifică că ești în folderul corect: `docker\2-reproducible-env`
- Verifică că `../../requirements.txt` există

### **Eroare: "Cannot connect to Docker daemon"**
- Pornește Docker Desktop
- Așteaptă să vezi iconița verde în system tray

### **Output-urile nu apar local**
- Sunt în Docker volumes
- Pentru a le vedea local, folosește volumes mapping (deja configurat)

### **Build durează mult**
- Prima dată instalează toate dependențele (~2-3 min)
- Rulările ulterioare sunt instant (cache)

---

## 📦 Distribuire Container

### **Salvează image pentru distribuire:**
```powershell
docker save ml-training-reproducible > ml-training-image.tar
```

### **Încarcă image pe alt sistem:**
```powershell
docker load < ml-training-image.tar
docker-compose up
```

**Comisia poate rula proiectul fără nicio instalare!**

---

**✅ Mediu 100% reproducibil creat!**
