# 🚀 Web Traffic Prediction Models - Ghid de Utilizare

## 📋 Descriere

Suite de modele Machine Learning pentru **predicția timpului petrecut pe site** (timeOnSite) bazat pe comportamentul utilizatorilor.

**🏆 Model BEST (Recomandat): Random Forest Regressor**
- **R² Score: 0.4928** (cel mai bun din toate modelele)
- **MAE: 96.69s** (~1.6 minute - cea mai bună precizie)
- **RMSE: 264.26s** - performanță excelentă
- Robust cu outliers și date complexe

**Alte modele disponibile:**
1. **Linear Regression** - R²=0.4651, MAE=111.48s (bun pentru baseline)
2. **Polynomial Regression** - R²=0.2707, MAE=107.15s (overfitting)

**Features:** 20 total (3 numerice + 12 categorice encodate + 5 engineered)  
**Target:** timeOnSite (seconds)

---

## 📂 Structură Proiect

```
Project_repo/
├── data/                          # 📊 Date (input, procesate și rezultate)
│   ├── raw/
│   │   └── ga-sessions.csv        # Dataset original brut (100,000 sesiuni)
│   ├── processed/
│   │   ├── X_simple.csv           # Features SIMPLE (3 coloane)
│   │   ├── X_enhanced.csv         # Features ENHANCED (20 coloane)
│   │   └── processed_training_data.csv  # Export din scripturile train_*.py (dacă rulate)
│   └── predictions/
│       ├── validation_results.csv # Validare pe TEST SET (~10,200 rânduri REALE)
│       └── predictions_*.csv      # Predicții din predict_new_data_example.py / predict_web_traffic.py
│
├── models/                        # 🤖 Modele antrenate (salvate cu pickle)
│   ├── random_forest_model_v2_BEST.pkl   # ⭐ MODEL BEST - Random Forest
│   ├── linear_regression_model_v2.pkl    # Model Linear Regression (backup)
│   ├── web_traffic_scaler_v2.pkl         # Scaler (Random Forest nu necesită, dar salvat)
│   ├── model_metadata_v2.json             # Metadata cu performanță Random Forest
│   ├── polynomial_regression_model.pkl   # Model Polynomial (dacă antrenat)
│   ├── polynomial_transformer.pkl        # Transformer polinomial
│   └── web_traffic_model_v1.pkl          # (Versiune veche - poate fi șters)
│
├── sources/                       # 💻 Scripturi Python
│   ├── mlflow_ml_model_web_traffic.py    # ⭐ SCRIPT PRINCIPAL - Random Forest cu MLflow
│   ├── train_liniar_model.py           # Antrenare regresie liniara (Linear Regression)
│   ├── train_polynomial_model.py         # Antrenare Polynomial
│   ├── train_random_forest_model.py      # Antrenare Random Forest (alt script)
│   └── predict_web_traffic.py            # Predicții pe date noi
│
├── notebooks/                     # 📓 Jupyter notebooks
│
├── ml-model-web-traffic1.py      # Script analiză comparativă (v1)
├── ml-model-web-traffic2.py      # Script analiză comparativă (v2)
└── README_MODEL.md               # Această documentație

---

## 🚀 Setup și Instalare

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

## 📚 Ghid de Utilizare

### **PASUL 1: Antrenare Modele**

Navighează în dosarul sources/:
```bash
cd sources
```

#### **A. Random Forest cu MLflow Tracking (RECOMANDAT - CEL MAI BUN MODEL! ⭐)**
```bash
python mlflow_ml_model_web_traffic.py
```

**Ce face:**
- Antrenează **Random Forest Regressor** cu 300 arbori, max_depth=30
- Feature engineering: 20 features (3 numerice + 12 categorice + 5 engineered)
- Log complet în **MLflow**: parametri, metrici, artifacts
- Generează **8 vizualizări** (distribuții, scatter, comparație modele, feature importance)
- Salvează: model, scaler, metadata JSON
- **Validare pe TEST SET** (~10,200 rânduri REALE) cu comparație predicted vs actual

**Output:**
```
MODEL Random Forest antrenat în 2.5 secunde
R² Test: 0.4928 ⭐ CEL MAI BUN!
MAE Test: 96.69 seconds (~1.6 minute)
RMSE Test: 264.26 seconds
Model salvat: models/random_forest_model_v2_BEST.pkl
Validare: data/predictions/validation_results.csv (~10,200 rânduri)
```

**Performanță:**
- R² Score: **0.4928** (explică ~49% din variație)
- MAE: **96.69s** (~1.6 minute) - 🎯 Cea mai bună precizie!
- RMSE: **264.26s** - Performanță excelentă
- Îmbunătățire: **+10.3%** față de versiunea simplă

---

#### **B. Linear Regression (Baseline - pentru comparație)**
```bash
python mlflow_ml_model_web_traffic.py
```

**Output:**
- ✅ `../models/linear_regression_model_v2.pkl` - Model ENHANCED salvat local
- ✅ `../models/web_traffic_scaler_v2.pkl` - Scaler pentru normalizare (20 features)
- ✅ `../models/model_metadata_v2.json` - Metadata completă cu TOP 10 coeficienți
- ✅ `../data/predictions/validation_results.csv` - Validare pe ~10,200 rânduri REALE din TEST SET
- ✅ `../visualizations/` - 8 grafice pentru analiză exploratorie
- ✅ MLflow tracking în `mlruns/` - Metrici și vizualizări

**Sau versiunea simplă (fără MLflow):**
```bash
python train_liniar_model.py
```
**Output:** `../models/linear_regression_model.pkl`, scaler și metadata

#### **B. Polynomial Regression**
```bash
python train_polynomial_model.py
```

**Output:**
- ✅ `../models/polynomial_regression_model.pkl`
- ✅ `../models/polynomial_transformer.pkl`
- ✅ `../models/polynomial_scaler.pkl`
- ✅ `../models/polynomial_regression_metadata.json`

#### **C. Random Forest**
```bash
python train_random_forest_model.py
```

**Output:**
- ✅ `../models/random_forest_model.pkl`
- ✅ `../models/random_forest_scaler.pkl`
- ✅ `../models/random_forest_metadata.json`

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
- ✅ Comparație metrici între run-uri (R², RMSE, MAE)
- ✅ Parametri model (coeficienți, features folosite)
- ✅ Vizualizări (distribuții, corelații, scatter plots)
- ✅ Artifacts descărcabile (modele, predictii, grafice)

**NOTĂ:** Modelele sunt salvate LOCAL în `models/`, nu în MLflow Registry!

---

## 📊 Performanță și Comparație Modele

### **Rezultate Finale (Versiune ENHANCED - 20 Features):**

| Model | R² Score | RMSE (sec) | MAE (sec) | Timp Antrenare | Status |
|-------|----------|------------|-----------|----------------|--------|
| **Random Forest ⭐ BEST** | **0.4928** | **264.26** | **96.69** | ~2.5s | **🏆 RECOMANDAT** |
| **Linear Regression** | 0.4651 | 271.43 | 111.48 | <1s | Baseline bun |
| **Polynomial (grad 2)** | 0.2707 | 316.95 | 107.15 | ~5s | ❌ Overfitting |

### **🏆 De ce Random Forest este cel mai bun:**

1. **R² Score: 0.4928** - Cel mai mare (explică ~49% din variație)
2. **MAE: 96.69s** - Cea mai bună precizie (~1.6 minute eroare medie)
3. **RMSE: 264.26s** - Cel mai mic (performanță excelentă cu outliers)
4. **Robust** - Gestionează date complexe și outliers mult mai bine
5. **Feature Importance** - Oferă interpretabilitate (ce features sunt importante)
6. **Îmbunătățire +10.3%** față de versiunea simplă (vs +2.15% Linear Regression)

### **Interpretare Metrici:**

**R² Score (Coeficient de determinare):**
- 0.4928 = **49.28%** din variația timeOnSite este explicată de Random Forest
- Mai mare = mai bun (max 1.0 = 100%)
- **Random Forest: 0.4928** vs Linear: 0.4651 → **+5.9% îmbunătățire**

**MAE (Mean Absolute Error):**
- Eroare medie absolută în secunde
- **Random Forest: 96.69s** (~1.6 minute) - Cea mai bună precizie!
- Linear: 111.48s (~1.9 minute)
- **Diferență: -14.79s** (~15 secunde mai precis)

**RMSE (Root Mean Square Error):**
- Eroare medie pătratică în secunde
- Mai mic = mai bun
- Penalizează mai mult erorile mari

**MAE (Mean Absolute Error):**
- Eroare medie absolută în secunde  
- Mai mic = mai bun
- Interpretare: predicția diferă în medie cu ±MAE secunde

### **Recomandări Utilizare:**

| Scenariu | Model Recomandat | Motiv |
|----------|------------------|-------|
| **Producție rapidă** | Linear Regression | Cel mai simplu, rapid, ușor de înțeles |
| **Performanță maximă** | Random Forest | Cel mai precis (+5% față de Linear) |
| **Compromis** | Polynomial | Echilibru performanță/complexitate |
| **Debugging/Explicabilitate** | Linear Regression | Coeficienți interpretabili direct |
| **MLflow Experiments** | Toate 3 | Comparație side-by-side |

---

## Exemple de Utilizare

### **Exemplu 1: Predicție simplă pentru o sesiune**
```python
import pandas as pd

# Creare date test
data = {
    'pageviews': [5],
    'visitNumber': [3],
    'hits': [12]
}
df_test = pd.DataFrame(data)
df_test.to_csv('test_session.csv', index=False)

# Rulare predicție din terminal
# cd sources
# python predict_web_traffic.py ../test/test_session.csv result.csv

# Rezultat: timeOnSite_predicted ≈ 245 secunde (~4 minute)
```

### **Exemplu 2: Validare completă pe TEST SET cu MLflow**
```bash
cd sources
python mlflow_ml_model_web_traffic.py

# Output:
# ✓ Model ENHANCED antrenat (20 features)
# ✓ Validare pe ~10,200 rânduri REALE din TEST SET
# ✓ Rezultate salvate in ../data/predictions/validation_results.csv
# ✓ 8 grafice generate in ../visualizations/
# ✓ MLflow tracking complet
```

**Verificare rezultate validare:**
```bash
cd ../data/predictions
# Deschide validation_results.csv pentru a vedea comparația predicted vs real
```

### **Exemplu 3: Comparație modele cu MLflow**
```bash
# Antrenează toate modelele cu MLflow tracking
cd sources

# Linear Regression ENHANCED
python mlflow_ml_model_web_traffic.py

# Poți adapta scriptul pentru celelalte modele
# Apoi vizualizează comparația:
mlflow ui --port 5000
# Deschide: http://localhost:5000
```

---

## Troubleshooting

### **Eroare: "Fisier model lipsa!"**
**Soluție:** Rulați mai întâi scriptul de antrenare:
```bash
cd sources
python train_liniar_model.py
```

### **Eroare: "Coloane lipsa din input"**
**Soluție:** Asigurați-vă că CSV-ul conține coloanele: `pageviews`, `visitNumber`, `hits`

**Format corect:**
```csv
pageviews,visitNumber,hits
5,3,12
2,1,5
```

### **Predicții negative sau prea mari**
**Soluție:** Scriptul limitează automat predicțiile la valori pozitive și rezonabile (<10,000s)

### **Performanță slabă pe date noi**
**Cauze posibile:**
1. Date noi foarte diferite de setul de antrenare (distribuție diferită)
2. Features lipsă sau incorecte
3. Model învechit (re-antrenare necesară)

**Soluție:** Re-antrenați modelul pe date recente:
```bash
# Actualizați data/raw/ga-sessions.csv cu date noi
cd sources
python train_liniar_model.py
```

### **MLflow UI nu pornește**
**Verificări:**
1. MLflow instalat: `pip install mlflow`
2. Port 5000 liber (sau folosește alt port: `mlflow ui --port 5001`)
3. Rulează din directorul corect

---
   - `bounce_rate` indicator
   
3. **Hyperparameter tuning:**
   - Random Forest cu optimizare
   - Gradient Boosting (XGBoost, LightGBM)
   
4. **Ensemble methods:**
   - Combinație Linear + Random Forest
   - Voting/Stacking ensemble

---

## Troubleshooting

### **Eroare: "Fisier model lipsa!"**
**Soluție:** Rulați mai întâi scriptul de antrenare:
```bash
python train_liniar_model.py
```

### **Eroare: "Coloane lipsa din input"**
**Soluție:** Asigurați-vă că CSV-ul conține coloanele: `pageviews`, `visitNumber`, `hits`

### **Predicții negative sau prea mari**
**Soluție:** Scriptul limitează automat predicțiile la valori pozitive și rezonabile

### **Performanță slabă pe date noi**
**Cauze posibile:**
1. Date noi foarte diferite de setul de antrenare (distribuție diferită)
2. Features lipsă sau incorecte
3. Model învechit (re-antrenare necesară)

**Soluție:** Re-antrenați modelul pe date recente:
```bash
# Actualizați data/raw/ga-sessions.csv cu date noi
python train_liniar_model.py
```

---

## Suport

**Date proiect:**
- Autor: Draghescu Radu
- Data: Mai 2026
- Versiune model: 1.0

**Pentru întrebări:**
- Consultați documentația sklearn: https://scikit-learn.org/
- Review `model_metadata.json` pentru detalii tehnice

---

## Changelog

### **v2.0 - Mai 2026**
- ✅ **ENHANCED MODEL cu 20 features:** 3 numerice + 12 categorice + 5 engineered
- ✅ **Performanță îmbunătățită:** R² = 0.4651 (vs 0.4553 SIMPLE) - creștere +2.15%
- ✅ **Validare completă pe TEST SET:** ~10,200 rânduri REALE cu comparație predicted vs actual
- ✅ **3 modele disponibile:** Linear ENHANCED, Polynomial, Random Forest
- ✅ **Structură reorganizată:** data/raw/, data/processed/, data/predictions/, models/, sources/, visualizations/
- ✅ **MLflow tracking:** Comparație experimente și parametri
- ✅ **8 grafice pentru EDA:** Vizualizări complete în visualizations/
- ✅ **Jupyter Notebook:** Interactive workflow în notebooks/
- ✅ **Docker:** 4 containerizări complete (tracking, API, Jupyter, reproducible env)
- ✅ **Documentație completă:** README, EXPLICATIE_VIZUALIZARI, QUICK_REFERENCE, GHID_COMISIE

### **v1.0 - Mai 2026**
- ✅ Model inițial Linear Regression Multiple (SIMPLE)
- ✅ Features: 3 numerice (pageviews, visitNumber, hits)
- ✅ R² = 0.4553
- ✅ Scripturi producție (train + predict)
- ✅ Documentație de bază

### **v2.0 - Mai 2026 (ENHANCED)**
- ✅ Feature Engineering: 5 features noi
- ✅ Categorical Encoding: device_category, country (12 features)
- ✅ Total: 20 features (3 numerice + 12 categorice + 5 engineered)
- ✅ R² = 0.4651 (îmbunătățire +2.15%)
- ✅ MAE = 111.48s (reducere -2.5s)
- ✅ Explicații metrici integrate în cod

---

## Status Final

** SUITE MODELE GATA PENTRU PRODUCȚIE!**

### ** Modele Disponibile (Versiune ENHANCED):**
1. **Linear Regression** - R² ~0.465 (✅ CEL MAI BUN - rapid, interpretabil, 20 features)
2. **Random Forest** - R² ~0.447 (bun, dar Linear superior)
3. **Polynomial Regression** - R² ~0.271 (❌ OVERFIT pe 230 features)

### **📁 Structură Completă:**
```
✓ data/raw/          - Dataset brut ga-sessions.csv
✓ data/processed/    - X_simple.csv, X_enhanced.csv (features procesate)
✓ data/predictions/  - validation_results.csv (~10,200 rânduri) + predictions_*.csv
✓ models/            - Model ENHANCED v2 + metadata cu TOP 10 coeficienți
✓ sources/           - mlflow_ml_model_web_traffic.py (tracking + validare pe TEST SET)
✓ visualizations/    - 8 grafice pentru EDA
✓ notebooks/         - Jupyter Notebook cu workflow complet (12 secțiuni)
✓ docker/            - 4 containerizări (tracking, API, Jupyter, reproducible env)
```

### **🚀 Următorii pași:**

**1. Antrenare modele:**
```bash
cd sources
python train_liniar_model.py           # Linear Regression
python train_polynomial_model.py         # Polynomial
python train_random_forest_model.py      # Random Forest
```

**2. Validare completă pe TEST SET:**
```bash
cd sources
python mlflow_ml_model_web_traffic.py
# → Include antrenare ENHANCED + validare pe ~10,200 rânduri REALE
# → Salvare rezultate în data/predictions/validation_results.csv
```

**3. MLflow tracking:**
```bash
python mlflow_ml_model_web_traffic.py
mlflow ui --port 5000
# → Vizualizare completă: metrici, parametri, artifacts, grafice
```

**4. Predicții producție:**
```bash
python predict_web_traffic.py <input.csv> <output.csv>
```

---

## 🚀 VERSIUNE ENHANCED - Linear Regression cu 20 Features

### **Descriere**
Versiunea îmbunătățită a modelului Linear Regression care folosește **feature engineering** și **features categorice** pentru performanță superioară.

### **Features (20 total)**

#### 1. Features Numerice Originale (3):
- `pageviews`: Număr pagini vizualizate
- `visitNumber`: Număr vizite istorice
- `hits`: Interacțiuni totale

#### 2. Features Categorice Encodate (12):
- `device_category`: mobile, tablet (desktop = baseline)
- `country`: Top 10 țări + "Other" (11 coloane one-hot encoded)

#### 3. Features Engineered (5):
- `pageviews_per_visit`: pageviews / (visitNumber + 1)
- `engagement_score`: pageviews × hits
- `hits_per_pageview`: hits / (pageviews + 1)
- `high_pageviews`: binary (1 dacă > median)
- `is_returning`: binary (1 dacă visitNumber > 1)

### **Performanță ENHANCED vs SIMPLE**

| Metric | SIMPLE (3 features) | ENHANCED (20 features) | Îmbunătățire |
|--------|---------------------|------------------------|--------------|
| **R² Score** | 0.4553 | **0.4651** | **+2.15%** ✅ |
| **MAE** | 113.98s | **111.48s** | **-2.50s** ✅ |
| **RMSE** | 273.90s | **271.43s** | **-2.47s** ✅ |

### **Rulare Model ENHANCED**
```bash
python sources/mlflow_ml_model_web_traffic.py
```

Scriptul va:
1. ✅ Încărca datele din `data/raw/ga-sessions.csv`
2. ✅ Aplica feature engineering automat (5 features noi)
3. ✅ Aplica one-hot encoding pentru categorice (12 features)
4. ✅ Salva seturile procesate în `data/processed/X_simple.csv` și `data/processed/X_enhanced.csv`
5. ✅ Antrena modelul cu toate cele 20 features
6. ✅ Genera 8 vizualizări în `visualizations/`
7. ✅ Salva modelul în `models/random_forest_model_v2_BEST.pkl`
8. ✅ Valida pe TEST SET (~10,200 rânduri REALE) → `data/predictions/validation_results.csv`

### **Avantaje ENHANCED**
✅ **Performanță superioară**: +2.15% R² față de versiunea simplă  
✅ **Insights mai profunde**: Features engineered captează patterns complexe  
✅ **Compatibil cu categorice**: Device type și locație geografică  
✅ **Interpretabil**: TOP 10 coeficienți (vizibil în graficul 08)  
✅ **Production-ready**: Metadata completă și validare pe date REALE  

### **Limitări**
⚠️ **Complexitate**: 20 features vs 3 în versiunea simplă  
⚠️ **Timp antrenare**: Marginal mai mare (0.015s vs 0.005s)  
⚠️ **Dependencies**: Necesită coloane categorice în dataset (`device_category`, `country`)  

---

## Suport și Resurse

**Documentație:**
- Sklearn Linear Regression: https://scikit-learn.org/stable/modules/linear_model.html
- Sklearn Random Forest: https://scikit-learn.org/stable/modules/ensemble.html#forest
- MLflow Tracking: https://mlflow.org/docs/latest/tracking.html

**Fișiere Importante:**
- `models/linear_regression_metadata.json` - Detalii performanță Linear
- `models/polynomial_regression_metadata.json` - Detalii performanță Polynomial
- `models/random_forest_metadata.json` - Detalii performanță Random Forest
- `test/test_summary.json` - Sumar comparație modele
- `test/model_comparison_results.csv` - Comparație predictii

**Autor:** Draghescu Radu 
**Versiune:** 2.0  
**Data:** Mai 2026  

---

**Start Predicting & Tracking!**