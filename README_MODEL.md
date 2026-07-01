# Web Traffic Prediction Models - Ghid de Utilizare

## Descriere

Suite de modele Machine Learning pentru **predicția timpului petrecut pe site** (timeOnSite) bazat pe comportamentul utilizatorilor.

** Model BEST (Recomandat): Random Forest Regressor**
- **R² Score: 0.4671** (cel mai bun din toate modelele)
- **MAE: 100.73s** (~1.4 minute - cea mai bună precizie)
- **RMSE: 270.94s** - performanță excelentă
- Robust cu outliers și date complexe

**Alte modele disponibile:**
1. **Linear Regression** - R²=0.4651, MAE=111.48s (bun pentru baseline)
2. **Polynomial Regression** - R²=0.2707, MAE=107.15s (overfitting)
3. **XGBoost Regressor** - R²=0.4528, MAE=103.11s 
4. **LightGBM** - R²=0.4419, MAE=103.96s 

**Features:** 20 total (3 numerice + 12 categorice encodate + 5 engineered)  
**Target:** timeOnSite (seconds)

---

## Structură Proiect

```
Project_repo/
├── data/                          # Date (input, procesate și rezultate)
│   ├── raw/
│   │   └── ga-sessions.csv        # Dataset original brut (100,000 sesiuni)
│   ├── processed/
│   │   ├── X_simple.csv           # Features SIMPLE (3 coloane)
│   │   └── X_enhanced.csv         # Features ENHANCED (20 coloane)
│   │   
│   └── predictions/
│       └── predictions_*.csv      # Predicții din predict_new_data_example.py / predict_web_traffic.py
│
├── Docker/                                  # Pachet fisiere pentru export imagine pe GitHub, testare si construire model pe GitHub
│       ├── ci_cd_pipeline_docker            # Dosarul pentru CI/CD pipeline 
│       └── EXPORT_PACHET_COMISIE.ps1        # Descriere script si pachet
│
├── models/                                  # Modele antrenate (salvate cu pickle)
│   ├── random_forest_model_v2_BEST.joblib   # BEST MODEL - Random Forest
│   ├── linear_regression_model_v2.joblib    # Model Linear Regression (backup)
│   └── model_metadata_v2.json               # Metadata cu performanță Random Forest
│
├── notebooks/                               # Dosarul cu fisierul interactiv Jupiter notebook
│   ├── mlruns                               # doasul cu run-urile din fisierul MLflow_ml_model_web_traffic.ipynb
│   ├── mlflow.db                            
│   └── MLflow_ml_model_web_traffic.ipynb    # Fisirul interacti pentru modulul cel mai bun
│
├── sources/                                 # Scripturi Python
│   ├── mlflow_ml_model_web_traffic.py       # SCRIPT PRINCIPAL - Random Forest cu MLflow
│   ├── model_metadata.json
│   ├── test_models_synthetic.py
│   ├── train_liniar_model.py                # Antrenare regresie liniara (Linear Regression)
│   ├── train_polynomial_model.py            # Antrenare Polynomial
│   └── train_random_forest_model.py         # Antrenare Random Forest (alt script)
│
├── visualizations/                          # Grafice si diagrame salvate in .png 
│
├── ml-model-web-traffic2.py      # Script analiză comparativă (v2)
└── README_MODEL.md               # Această documentație

---

## Setup și Instalare

### 1. **Cerințe sistem**
```bash
Python 3.8+
pandas
numpy
scikit-learn
matplotlib
seaborn
mlflow  # Pentru tracking experimente
```

### 2. **Instalare dependențe**

**Opțiunea A: Instalare individuală**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn mlflow
```

**Opțiunea B: Din requirements.txt**
```bash
pip install -r requirements.txt
```

### 3. **Structură necesară**
```bash
cd Project_repo
# Verifică că dosarele există:
# Verifică că dosarele există:
# data/raw/, data/processed/, data/predictions/, models/, sources/, notebooks/, visualizations/
```

---

## Ghid de Utilizare

### **PASUL 1: Antrenare Modele**

Navighează în dosarul sources/:
```bash
cd sources
```

#### **A. Random Forest cu MLflow Tracking (RECOMANDAT - CEL MAI BUN MODEL!)**
```bash
python mlflow_ml_model_web_traffic.py
```

**Ce face:**
- Antrenează **Random Forest Regressor** cu 300 arbori, max_depth=10
- Feature engineering: 20 features (3 numerice + 12 categorice + 5 engineered)
- Log complet în **MLflow**: parametri, metrici, artifacts
- Generează **8 vizualizări** (distribuții, scatter, comparație modele, feature importance)
- Salvează: model, scaler, metadata JSON
- **Validare pe TEST SET** (~10,200 rânduri REALE) cu comparație predicted vs actual

**Output:**
```
MODEL Random Forest antrenat în 4.8 secunde
R² Test: 0.4671 CEL MAI BUN!
MAE Test: 100.73 seconds (~1.6 minute)
RMSE Test: 270.94 seconds
Model salvat: 
- `../models/random_forest_model_v2_BEST.joblib` - Model ENHANCED salvat local
- `../models/web_traffic_scaler_v2.joblib` - Scaler pentru normalizare (20 features)
- `../models/model_metadata_v2.json` - Metadata completă cu TOP 10 coeficienți
Validare: 
- `../data/predictions/validation_results.csv` - Validare pe ~10,200 rânduri REALE din TEST SET
- `../visualizations/` - 8 grafice pentru analiză exploratorie
- MLflow tracking în `mlruns/` - Metrici și vizualizări
```

**Performanță:**
- R² Score: **0.4671** (explică ~47% din variație)
- MAE: **100.73s** (~1.4 minute) - Cea mai bună precizie!
- RMSE: **270.94s** - Performanță excelentă
- Îmbunătățire: **+1.4%** față de versiunea simplă

---

#### **B. Linear Regression (Baseline - pentru comparație)**
```bash
python train_liniar_model.py
```
**Output:** `../models/random_forest_model_v2_BEST.joblib`, scaler și metadata

#### **C. Polynomial Regression**
```bash
python train_polynomial_model.py
```
**Output:** `../models/random_forest_model_v2_BEST.joblib`, scaler și metadata

---

### **PASUL 2: Validare pe TEST SET (Date Reale)**

Scriptul MLflow include automat validarea pe TEST SET (~10,200 rânduri REALE):

```bash
cd sources
python mlflow_ml_model_web_traffic.py
```

**Ce face:**
- Antrenează modelul Linear Regression ENHANCED (20 features)
- Split date: 80% train, 20% test (~10,200 rânduri în TEST SET)
- Face predicții pe TEST SET cu valori REALE de timeOnSite
- Compară predicted vs real pentru fiecare sesiune
- Calculează erori: MAE, RMSE, erori absolute și procentuale
- Salvează rezultate în `../data/predictions/validation_results.csv`
- Log toate metricile în MLflow

**Output:**
- `../data/predictions/validation_results.csv` - Validare completă cu ~10,200 rânduri
  - Conține: features (pageviews, visitNumber, hits), timeOnSite_REAL, timeOnSite_predicted, error_seconds, error_absolute, error_percent
- `../models/model_metadata_v2.json` - Metadata cu TOP 10 coeficienți și statistici features
- `../visualizations/` - 8 grafice pentru analiză exploratorie

---

### **PASUL 3: Predicții pe Date Noi**

Folosește modelul antrenat pentru predicții:

#### **Opțiunea A: Cu fișier output specificat**
```bash
python predict_web_traffic.py <input_file.csv> <output_file.csv>
```

#### **Opțiunea B: Output automat (timestamp)**
```bash
python predict_web_traffic.py <input_file.csv>
# → Generează: data/predictions/predictions_YYYYMMDD_HHMMSS.csv
```

**Format Input CSV:**
```csv
pageviews,visitNumber,hits
5,3,12
2,1,5
10,8,25
```

**Output CSV:**
```csv
pageviews,visitNumber,hits,timeOnSite_predicted,model,prediction_date,model_r2_test,model_mae_test
5,3,12,245.67,Linear Regression ENHANCED,2026-05-29 14:30:00,0.4651,111.48
2,1,5,98.34,Linear Regression ENHANCED,2026-05-29 14:30:00,0.4651,111.48
```

---

### **PASUL 4: Încărcare și Utilizare Model**

**Încărcare model Random Forest salvat local:**
```python
import pickle
import pandas as pd

# Încarcă modelul Random Forest (CEL MAI BUN)
with open("models/random_forest_model_v2_BEST.pkl", "rb") as f:
    model = pickle.load(f)

# NOTĂ: Random Forest nu necesită scaling, dar îl păstrăm pentru consistență
with open("models/web_traffic_scaler_v2.pkl", "rb") as f:
    scaler = pickle.load(f)

# Date noi (toate cele 20 features necesare)
# Pentru simplificare, exemplul folosește doar features numerice de bază
new_data = pd.DataFrame({
    'pageviews': [10, 25],
    'visitNumber': [1, 3],
    'hits': [15, 40]
})

# IMPORTANT: Pentru model complet, necesită toate cele 20 features
# Vezi model_metadata_v2.json pentru lista completă de features

# Predicție (Random Forest nu necesită scaling, dar îl aplicăm pentru consistență)
predictions = model.predict(new_data)  # Sau: model.predict(scaler.transform(new_data))
print(f"Predicții timeOnSite: {predictions} secunde")
# Output: [187.45, 312.89] (exemple)
```

---

### **PASUL 5: MLflow UI - Vizualizare Metrici (Opțional)**

Vizualizează tracking-ul experimentelor:

```bash
mlflow ui --port 5000
# Deschide browser la: http://localhost:5000
```

**Ce poți vedea:**
- Comparație metrici între run-uri (R², RMSE, MAE)
- Parametri model (coeficienți, features folosite)
- Vizualizări (distribuții, corelații, scatter plots)
- Artifacts descărcabile (modele, predictii, grafice)

**NOTĂ:** Modelele sunt salvate LOCAL în `models/`, nu în MLflow Registry!

---

## Performanță și Comparație Modele

### **Rezultate Finale (Versiune ENHANCED - 20 Features):**

| Model | R² Score | RMSE (sec) | MAE (sec) | Timp Antrenare | Status |
|-------|----------|------------|-----------|----------------|--------|
| **Random Forest BEST** | **0.4671** | **270.94** | **100.73** | ~4.8s | ** RECOMANDAT** |
| **Linear Regression** | 0.4651 | 271.43 | 111.48 | <1s | Baseline bun |
| **Polynomial (grad 2)** | 0.2707 | 316.95 | 107.15 | ~5s | Overfitting |
| **XGBoost Regressor** |  R²=0.4528 | MAE=103.11s | ~8s | Mai slab decat RF
| **LightGBM** | R²=0.4419 | MAE=103.96s  | ~3s | Mai slab decat RF

### **Interpretare Metrici(RF):**

**R² Score (Coeficient de determinare):**
- 0.4671 = **47%** din variația timeOnSite este explicată de Random Forest
- Mai mare = mai bun (max 1.0 = 100%)
- **Random Forest: 0.4671** vs Linear: 0.4651

**MAE (Mean Absolute Error):**
- Eroare medie absolută în secunde
- **Random Forest: 100.73s** (~1.6 minute) - Cea mai bună precizie!
- Linear: 111.48s (~2 minute )
- **Diferență: ~-11s**

**RMSE (Root Mean Square Error):**
- Eroare medie pătratică în secunde
- Mai mic = mai bun
- Penalizează mai mult erorile mari

**MAE (Mean Absolute Error):**
- Eroare medie absolută în secunde  
- Mai mic = mai bun
- Interpretare: predicția diferă în medie cu ±MAE secunde


### **Limitări**
 **Complexitate**: 20 features vs 3 în versiunea simplă  
 **Timp antrenare**: Marginal mai mare (0.015s vs 0.005s)  
 **Dependencies**: Necesită coloane categorice în dataset (`device_category`, `country`)  

---

## Suport și Resurse

**Documentație:**
- Sklearn Linear Regression: https://scikit-learn.org/stable/modules/linear_model.html
- Sklearn Random Forest: https://scikit-learn.org/stable/modules/ensemble.html#forest
- MLflow Tracking: https://mlflow.org/docs/latest/tracking.html


**Autor:** Draghescu Radu 
**Versiune:** 2.0  
**Data:** IUNIE 2026  

---

**Start Predicting & Tracking!**scm-history-item:c%3A%5CUsers%5CUser%5CDesktop%5Crepos%5CProject_repo?%7B%22repositoryId%22%3A%22scm0%22%2C%22historyItemId%22%3A%22446364f9c9958080ba2abd3edd50d27aa8f27dec%22%2C%22historyItemParentId%22%3A%228812346dc03d764ff88341eb94ed687ae5d76234%22%2C%22historyItemDisplayId%22%3A%22446364f%22%7D