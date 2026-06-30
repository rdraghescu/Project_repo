"""
SCRIPT DE TESTARE MODELE CU DATE SINTETICE
===========================================
Genereaza date sintetice si testeaza toate modelele antrenate.

Genereaza ~1000 de date aleatorii cu valori intre min si max pentru fiecare feature.
Testeaza toate modelele disponibile si salveaza rezultatele in test/.

Autor: Proiect Web Analytics
Data: 2026-05-24
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TESTARE MODELE CU DATE SINTETICE")
print("="*80)

# ===========================
# 1. GENERARE DATE SINTETICE
# ===========================
print("\n[1/4] Generare date sintetice...")

# Citim statistici din metadata pentru a genera date realiste
try:
    with open('../models/model_metadata_v2.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    feature_stats = metadata['feature_statistics']
    required_features = metadata.get('features_all', metadata.get('features', []))
    
    print(f"✓ Metadata incarcata pentru {len(required_features)} features")
except FileNotFoundError:
    print("⚠ Metadata nu exista. Folosim valori implicite.")
    required_features = ['pageviews', 'visitNumber', 'hits']
    feature_stats = {
        'pageviews': {'min': 1, 'max': 20},
        'visitNumber': {'min': 1, 'max': 15},
        'hits': {'min': 1, 'max': 50}
    }

# Generam 1000 de seturi de date sintetice
np.random.seed(42)
n_samples = 1000

synthetic_data = {}
for feat in required_features:
    min_val = feature_stats[feat]['min']
    max_val = feature_stats[feat]['max']
    
    # Generam valori aleatorii intre min si max
    synthetic_data[feat] = np.random.uniform(min_val, max_val, n_samples)
    
    print(f"   {feat:15s}: {n_samples} valori intre {min_val:.1f} si {max_val:.1f}")

df_synthetic = pd.DataFrame(synthetic_data)
print(f"\n✓ Date sintetice generate: {len(df_synthetic)} randuri")

# Salvam datele sintetice
synthetic_filename = '../test/synthetic_test_data.csv'
df_synthetic.to_csv(synthetic_filename, index=False)
print(f"✓ Date sintetice salvate: {synthetic_filename}")

# ===========================
# 2. TESTARE MODELE
# ===========================
print("\n[2/4] Testare modele disponibile...")

models_to_test = [
    # ('Linear Regression', '../models/linear_regression_modellinear_regression_model_v2.pkl', '../models/linear_regression_scaler.pkl', '../models/linear_regression_metadata.json'),
    # ('Polynomial Regression', '../models/polynomial_model.pkl', None, '../models/polynomial_metadata.json'),
    # ('Random Forest', '../models/random_forest_model.pkl', '../models/random_forest_scaler.pkl', '../models/random_forest_metadata.json')
    ('Random Forest ENHANCED', '../models/random_forest_model_v2_BEST.joblib', '../models/web_traffic_scaler_v2.joblib', '../models/model_metadata_v2.json'),

]

results = []

for model_name, model_path, scaler_path, metadata_path in models_to_test:
    print(f"\n--- Testare {model_name} ---")
    
    # Verificam daca modelul exista
    if not os.path.exists(model_path):
        print(f"⚠ Model {model_name} nu exista. Skiping...")
        continue
    
    try:
        # Incarcam modelul
        # with open(model_path, 'rb') as f:
        #     model = pickle.load(f)
        model = joblib.load(model_path)
        print(f"✓ Model incarcat")
        
        # Incarcam scaler-ul daca exista
        if scaler_path and os.path.exists(scaler_path):
            # with open(scaler_path, 'rb') as f:
            #     scaler = pickle.load(f)
            scaler = joblib.load(scaler_path)
            X_test = scaler.transform(df_synthetic[required_features])
            print(f"✓ Scaler aplicat")
        else:
            X_test = df_synthetic[required_features].values
            print(f"✓ Date pregatite (fara scaler)")
        
        # Facem predictii
        start_time = datetime.now()
        predictions = model.predict(X_test)
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        # Asiguram ca predictiile sunt non-negative
        predictions = np.maximum(predictions, 0)
        
        print(f"✓ Predictii generate: {len(predictions)} valori")
        print(f"   Timp: {prediction_time:.4f} secunde")
        print(f"   Viteza: {len(predictions)/prediction_time:,.0f} predictii/sec")
        
        # Statistici predictii
        print(f"   Mean: {predictions.mean():.2f} secunde")
        print(f"   Median: {np.median(predictions):.2f} secunde")
        print(f"   Min: {predictions.min():.2f} secunde")
        print(f"   Max: {predictions.max():.2f} secunde")
        
        # Salvam predictiile
        df_results = df_synthetic.copy()
        df_results['timeOnSite_predicted'] = predictions
        df_results['model_name'] = model_name
        df_results['prediction_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        output_filename = f"../test/{model_name.lower().replace(' ', '_')}_predictions.csv"
        df_results.to_csv(output_filename, index=False)
        print(f"✓ Rezultate salvate: {output_filename}")
        
        # Salvam rezultatele pentru comparatie
        results.append({
            'model_name': model_name,
            'n_samples': len(predictions),
            'prediction_time_seconds': prediction_time,
            'predictions_per_second': len(predictions)/prediction_time,
            'mean_prediction': float(predictions.mean()),
            'median_prediction': float(np.median(predictions)),
            'std_prediction': float(predictions.std()),
            'min_prediction': float(predictions.min()),
            'max_prediction': float(predictions.max())
        })
        
    except Exception as e:
        print(f"✗ EROARE la testarea {model_name}: {e}")
        continue

# ===========================
# 3. COMPARATIE MODELE
# ===========================
print("\n[3/4] Comparatie performanta modele...")

if len(results) > 0:
    print("\n" + "="*80)
    print("REZULTATE COMPARATIE")
    print("="*80)
    
    df_comparison = pd.DataFrame(results)
    
    print("\nStatistici predictii:")
    print("-" * 80)
    print(f"{'Model':<25} {'Mean (sec)':<15} {'Median (sec)':<15} {'Viteza (pred/sec)':<20}")
    print("-" * 80)
    for _, row in df_comparison.iterrows():
        print(f"{row['model_name']:<25} {row['mean_prediction']:<15.2f} {row['median_prediction']:<15.2f} {row['predictions_per_second']:<20,.0f}")
    print("-" * 80)
    
    # Salvam comparatia
    comparison_filename = '../test/models_comparison.csv'
    df_comparison.to_csv(comparison_filename, index=False)
    print(f"\n✓ Comparatie salvata: {comparison_filename}")
    
    # Salvam si ca JSON pentru vizualizare
    comparison_json_filename = '../test/models_comparison.json'
    with open(comparison_json_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"✓ Comparatie JSON salvata: {comparison_json_filename}")
    
else:
    print("⚠ Niciun model nu a fost testat cu succes")

# ===========================
# 4. GENERARE RAPORT
# ===========================
print("\n[4/4] Generare raport testare...")

report_content = f"""
RAPORT TESTARE MODELE CU DATE SINTETICE
========================================
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. DATE SINTETICE
-----------------
- Numar sample-uri: {n_samples}
- Features testate: {', '.join(required_features)}

Intervale valori generate:
"""

for feat in required_features:
    min_val = df_synthetic[feat].min()
    max_val = df_synthetic[feat].max()
    mean_val = df_synthetic[feat].mean()
    report_content += f"  - {feat}: [{min_val:.2f}, {max_val:.2f}], Mean: {mean_val:.2f}\n"

report_content += f"\n2. MODELE TESTATE\n-----------------\n"

for result in results:
    report_content += f"\n{result['model_name']}:\n"
    report_content += f"  - Predictii mean: {result['mean_prediction']:.2f} secunde\n"
    report_content += f"  - Predictii median: {result['median_prediction']:.2f} secunde\n"
    report_content += f"  - Predictii std: {result['std_prediction']:.2f} secunde\n"
    report_content += f"  - Viteza: {result['predictions_per_second']:,.0f} predictii/secunda\n"

report_content += f"\n3. FISIERE GENERATE\n-------------------\n"
report_content += f"  - {synthetic_filename}\n"
report_content += f"  - {comparison_filename}\n"
report_content += f"  - {comparison_json_filename}\n"

for result in results:
    model_file = f"../test/{result['model_name'].lower().replace(' ', '_')}_predictions.csv"
    report_content += f"  - {model_file}\n"

report_content += f"\n" + "="*80 + "\n"

# Salvam raportul
report_filename = '../test/test_report.txt'
with open(report_filename, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(report_content)
print(f"✓ Raport salvat: {report_filename}")

# ===========================
# FINALIZARE
# ===========================
print("\n" + "="*80)
print("✓ TESTARE COMPLETA CU SUCCES!")
print("="*80)

print(f"\nFisiere generate in test/:")
print(f"  - {n_samples} date sintetice")
print(f"  - {len(results)} modele testate")
print(f"  - Raport complet de testare")

print("\nPentru a vizualiza rezultatele:")
print(f"  cat {report_filename}")
print(f"  cat {comparison_filename}")

print("\n" + "="*80)
