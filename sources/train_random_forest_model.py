"""
SCRIPT DE ANTRENARE RANDOM FOREST REGRESSOR - PRODUCTIE
=========================================================
Antreneaza modelul Random Forest pentru predictia timpului pe site.

Model: Random Forest Regressor
Features: pageviews, visitNumber, hits
Target: timeOnSite (seconds)

Autor: Proiect Web Analytics
Data: 2026-05-24
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ANTRENARE MODEL PRODUCTIE - RANDOM FOREST REGRESSOR")
print("="*80)

# ===========================
# 1. INCARCARE DATE
# ===========================
print("\n[1/6] Incarcare date...")
df = pd.read_csv('../data/raw/ga-sessions.csv')
print(f"✓ Dataset incarcat: {df.shape[0]:,} randuri, {df.shape[1]} coloane")

# ===========================
# 2. CURATARE DATE
# ===========================
print("\n[2/6] Curatare si validare date...")

# Eliminam valorile lipsa din target
df_clean = df.dropna(subset=['timeOnSite'])
print(f"✓ Eliminare valori lipsa: {len(df_clean):,} randuri ramase")

# Eliminam valorile negative sau anormale
df_clean = df_clean[df_clean['timeOnSite'] >= 0]
df_clean = df_clean[df_clean['timeOnSite'] <= 10000]
print(f"✓ Eliminare valori anormale: {len(df_clean):,} randuri ramase")

# Verificam features necesare
required_features = ['pageviews', 'visitNumber', 'hits']
for feat in required_features:
    if feat not in df_clean.columns:
        raise ValueError(f"Coloana necesara '{feat}' nu exista in dataset!")
    df_clean = df_clean.dropna(subset=[feat])

print(f"✓ Date curatate final: {len(df_clean):,} randuri")

# ===========================
# 3. PREGATIRE FEATURES
# ===========================
print("\n[3/6] Pregatire features...")

X = df_clean[required_features].copy()
y = df_clean['timeOnSite'].copy()

print(f"✓ Features selectate: {list(X.columns)}")
print(f"✓ Target variable: timeOnSite")
print(f"   - Mean: {y.mean():.2f} seconds")
print(f"   - Median: {y.median():.2f} seconds")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✓ Train/Test split: {len(X_train):,} / {len(X_test):,} (80/20)")

# Standardizare (optional pentru Random Forest, dar poate ajuta)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✓ Features standardizate")

# ===========================
# 4. ANTRENARE MODEL
# ===========================
print("\n[4/6] Antrenare model Random Forest Regressor...")

start_time = datetime.now()

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

model.fit(X_train_scaled, y_train)
training_time = (datetime.now() - start_time).total_seconds()

print(f"✓ Model antrenat in {training_time:.4f} secunde")
print(f"   Arbori: {model.n_estimators}")
print(f"   Max depth: {model.max_depth}")

# ===========================
# 5. EVALUARE MODEL
# ===========================
print("\n[5/6] Evaluare performanta...")

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"Performanta pe setul de ANTRENARE:")
print(f"   R² Score: {r2_train:.4f}")

print(f"\nPerformanta pe setul de TEST:")
print(f"   R² Score: {r2_test:.4f}")
print(f"   RMSE: {rmse_test:.2f} seconds")
print(f"   MAE: {mae_test:.2f} seconds")

# Verificare overfitting
overfitting_gap = r2_train - r2_test
print(f"\nVerificare overfitting:")
print(f"   Gap R² (Train - Test): {overfitting_gap:.4f}")
if overfitting_gap < 0.05:
    print(f"   Status: ✓ Nu exista overfitting semnificativ")
elif overfitting_gap < 0.10:
    print(f"   Status: ⚠ Overfitting minor")
else:
    print(f"   Status: ⚠ Overfitting semnificativ!")

# Feature importances
print(f"\nImportanta features:")
feature_importances = sorted(
    zip(required_features, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)
for feat, importance in feature_importances:
    print(f"   {feat:15s}: {importance:.4f} ({importance*100:.1f}%)")

# ===========================
# 6. SALVARE MODEL
# ===========================
print("\n[6/6] Salvare model si metadate...")

# Salvare date procesate
df_clean[required_features + ['timeOnSite']].to_csv('../data/processed_training_data.csv', index=False)
print("✓ Date procesate salvate: ../data/processed_training_data.csv")

# Salvare model
# model_filename = '../models/random_forest_model.pkl'
model_filename = '../models/random_forest_model.joblib'
# with open(model_filename, 'wb') as f:
#     pickle.dump(model, f)
joblib.dump(model, model_filename, compress=3)
print(f"✓ Model salvat: {model_filename}")

# Salvare scaler
# scaler_filename = '../models/random_forest_scaler.pkl'
scaler_filename = '../models/random_forest_scaler.joblib'
# with open(scaler_filename, 'wb') as f:
#     pickle.dump(scaler, f)
joblib.dump(scaler, scaler_filename, compress=3)
print(f"✓ Scaler salvat: {scaler_filename}")

# Salvare metadate
metadata = {
    'model_type': 'RandomForestRegressor',
    'model_name': 'random_forest',
    'model_version': '2.0',
    'n_estimators': int(model.n_estimators),
    'max_depth': int(model.max_depth) if model.max_depth else None,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'training_time_seconds': training_time,
    'dataset_size': len(df_clean),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'features': required_features,
    'target': 'timeOnSite',
    'performance': {
        'r2_train': float(r2_train),
        'r2_test': float(r2_test),
        'rmse_test': float(rmse_test),
        'mae_test': float(mae_test),
        'overfitting_gap': float(overfitting_gap)
    },
    'feature_importances': {
        feat: float(imp) for feat, imp in zip(required_features, model.feature_importances_)
    },
    'target_statistics': {
        'mean': float(y.mean()),
        'median': float(y.median()),
        'std': float(y.std()),
        'min': float(y.min()),
        'max': float(y.max())
    },
    'feature_statistics': {
        feat: {
            'mean': float(X[feat].mean()),
            'std': float(X[feat].std()),
            'min': float(X[feat].min()),
            'max': float(X[feat].max())
        } for feat in required_features
    }
}

metadata_filename = '../models/random_forest_metadata.json'
with open(metadata_filename, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)
print(f"✓ Metadata salvata: {metadata_filename}")

# ===========================
# FINALIZARE
# ===========================
print("\n" + "="*80)
print("✓ MODEL RANDOM FOREST ANTRENAT SI SALVAT CU SUCCES!")
print("="*80)

print(f"\nFisiere generate:")
print(f"   1. {model_filename}")
print(f"   2. {scaler_filename}")
print(f"   3. {metadata_filename}")
print(f"   4. ../data/processed_training_data.csv")

print(f"\nPerformanta finala:")
print(f"   R² Score: {r2_test:.4f} ({r2_test*100:.1f}% variatie explicata)")
print(f"   Eroare medie: {mae_test:.2f} secunde (~{mae_test/60:.1f} minute)")

print("\n" + "="*80)
