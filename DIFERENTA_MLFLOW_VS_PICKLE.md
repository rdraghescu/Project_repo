# 📦 STRATEGIA DE SALVARE: Pickle Local + MLflow Tracking

## 🎯 ABORDAREA ACTUALĂ (2026-05-28)

**DECIZIE: Salvare locală cu pickle în `models/`, MLflow doar pentru tracking metrici**

### De ce această alegere?

✅ **Control complet** - Modelele sunt salvate local, ușor de accesat  
✅ **Simplitate** - Încărcare directă cu `pickle.load()`  
✅ **Portabilitate** - Fișiere simple `.pkl` care pot fi copiate oriunde  
✅ **MLflow pentru tracking** - Păstrăm beneficiile tracking-ului de experimente  

---

## 📁 STRUCTURA ACTUALĂ

```
models/
├── linear_regression_model_v2.pkl    ← MODEL (salvat cu pickle)
├── web_traffic_scaler_v2.pkl         ← SCALER
├── model_metadata_v2.json            ← METADATA
└── web_traffic_model_v1.pkl          ← Versiune veche (poate fi șters)

mlruns/
└── [experiment_id]/
    └── [run_id]/
        └── artifacts/
            ├── visualizations/        ← Grafice
            ├── models/                ← Copie model (pentru referință)
            ├── preprocessors/         ← Copie scaler
            └── metadata/              ← Copie metadata
```

---

## ✅ RĂSPUNS LA ÎNTREBAREA INIȚIALĂ:

> "De ce nu e `models:/web_traffic_model_v1/latest`?"

**Răspuns:** Pentru că acum **NU mai folosim MLflow Registry**!  
- Modelele sunt salvate DOAR în `models/*.pkl`  
- MLflow e folosit doar pentru tracking metrici și vizualizări
- Încărcarea se face cu `pickle.load()`, nu `mlflow.sklearn.load_model()`

---

## � CUM ÎNCĂRCĂM MODELUL ACUM

### Metoda Curentă (SIMPLĂ):

```python
import pickle
import pandas as pd

# 1. Încarcă modelul
with open("models/linear_regression_model_v2.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Încarcă scaler-ul
with open("models/web_traffic_scaler_v2.pkl", "rb") as f:
    scaler = pickle.load(f)

# 3. Încarcă metadata (opțional)
import json
with open("models/model_metadata_v2.json", "r") as f:
    metadata = json.load(f)
    print(f"Model R²: {metadata['performance']['r2_test']:.4f}")

# 4. Folosește modelul
new_data = pd.DataFrame({
    'pageviews': [10, 25],
    'visitNumber': [1, 3],
    'hits': [15, 40]
})

new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
print(f"Predicții: {predictions}")
```

---

## 📊 COMPARAȚIE: Versiunea Veche vs. Nouă

| Aspect | Versiunea Veche (MLflow Registry) | Versiunea Nouă (Pickle Local) |
|--------|-----------------------------------|-------------------------------|
| **Salvare model** | `mlflow.sklearn.log_model()` | `pickle.dump(model, f)` |
| **Încărcare model** | `mlflow.sklearn.load_model("models:/...")` | `pickle.load(f)` |
| **Locație** | `mlruns/.../artifacts/` | `models/*.pkl` |
| **Complexitate** | Medie | Simplă |
| **Portabilitate** | Necesită MLflow | Doar pickle |
| **Tracking metrici** | ✅ Da | ✅ Da (MLflow păstrat) |
| **Vizualizări** | ✅ Da | ✅ Da (MLflow păstrat) |

---

## 🎯 DE CE AM FĂCUT ACEASTĂ SCHIMBARE?

### Avantaje:
1. **Simplitate** - Încărcare directă fără dependență MLflow
2. **Control** - Știi exact unde este modelul
3. **Portabilitate** - Poți copia doar `models/` oriunde
4. **Rapiditate** - Fără overhead MLflow Registry
5. **Debugging** - Mai ușor de debugat probleme

### Ce păstrăm din MLflow:
✅ **Tracking metrici** - R², MAE, RMSE, coeficienți  
✅ **Vizualizări** - Distribuții, corelații, scatter plots  
✅ **Comparare experimente** - În MLflow UI  
✅ **Artifacts** - Copii pentru referință în `mlruns/`

---

## 📝 COD ACTUAL ÎN `mlflow_ml_model_web_traffic.py`

```python
# ===========================
# 5. SALVARE ARTIFACTS
# ===========================
print("\n[5/5] Salvare artifacts...")

# Salvam modelul LOCAL cu pickle în folderul models/
model_path = os.path.join(MODELS_DIR, "linear_regression_model_v2.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"✓ Model salvat local: {model_path}")

# Loghăm modelul ca artifact în MLflow (pentru referință)
mlflow.log_artifact(model_path, "models")
print("✓ Model logged in MLflow")

# Salvam scaler-ul
scaler_path = os.path.join(MODELS_DIR, "web_traffic_scaler_v2.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
mlflow.log_artifact(scaler_path, "preprocessors")
print(f"✓ Scaler salvat: {scaler_path}")

# Salvam metadata
metadata = {...}  # Performanță, coeficienți, features
metadata_path = os.path.join(MODELS_DIR, "model_metadata_v2.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)
mlflow.log_artifact(metadata_path, "metadata")
print(f"✓ Metadata salvată: {metadata_path}")
```

---

## 📋 REZUMAT PENTRU PREZENTARE ÎN COMISIE

### Întrebare posibilă:
> "De ce nu folosiți MLflow Model Registry?"

### Răspuns:
> "Am optat pentru **salvare locală cu pickle** pentru simplitate și control direct:
> - ✅ **Încărcare simplă**: `pickle.load()` fără dependențe complexe
> - ✅ **Portabilitate**: Fișiere `.pkl` pot fi copiate și folosite oriunde
> - ✅ **Control complet**: Știm exact unde sunt modelele
> - ✅ **MLflow pentru tracking**: Păstrăm beneficiile tracking-ului de metrici și vizualizări
> 
> Această abordare este **ideală pentru proiecte academice și prototipuri rapide**, 
> oferind un echilibru între simplitate și profesionalism."

---

## 🎯 CONCLUZIE

**Strategia actuală (2026-05-28):**
- 📦 **Modele** → Salvate în `models/*.pkl` cu pickle
- 📊 **Tracking** → MLflow pentru metrici, parametri, vizualizări
- 🎨 **Artifacts** → Copii în `mlruns/` pentru referință
- 🔧 **Încărcare** → Directă cu `pickle.load()`, simplă și rapidă

**Beneficii:**
- ✅ Simplu de folosit
- ✅ Fără dependențe complexe
- ✅ Ușor de înțeles pentru prezentare
- ✅ Control complet asupra modelelor
- ✅ Păstrează avantajele MLflow tracking

---

**Ultima actualizare:** 2026-05-28  
**Versiune:** 2.0 (Pickle Local + MLflow Tracking)
