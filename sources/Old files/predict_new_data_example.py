"""
EXEMPLU PREDICȚIE PE DATE NOI - UTILIZARE MODEL ÎN PRODUCȚIE
=============================================================
Acest script demonstrează cum se folosește modelul antrenat
pentru a face predicții pe date noi în scenarii reale.

Utilizare:
    python predict_new_data_example.py

Proiect Web Analytics
Data: 2026-05-27
"""

import pandas as pd
import numpy as np
import pickle
import mlflow.sklearn
import os
from datetime import datetime

# ===========================
# CONFIGURARE
# ===========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Directorul curent al scriptului
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # Presupunem că structura este: Project_root/sources/predict_new_data_example.py
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models') # Asigură-te că există directorul pentru modele

print("="*80)
print("PREDICȚIE TIMEONSITE PE DATE NOI")
print("="*80)

# ===========================
# 1. ÎNCĂRCARE MODEL ȘI SCALER
# ===========================
print("\n[1/4] Încărcare model și preprocessor...")

# Încărcăm scaler-ul
scaler_path = os.path.join(MODELS_DIR, "web_traffic_scaler_v2.pkl")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
print(f"Scaler încărcat: {scaler_path}")

# Încărcăm modelul din MLflow
try:
    # OPȚIUNEA 1: Încarcă din MLflow Registry (recomandat în producție)
    # Sintaxa: models:/NUMEmodel/versiune
    # NUME model = registered_model_name din mlflow_ml_model_web_traffic.py (linia 190)
    model = mlflow.sklearn.load_model("models:/WebTraffic_LinearRegression/latest")
    print(" Model încărcat din MLflow Registry: WebTraffic_LinearRegression")
except Exception as e:
    print(f"  Nu s-a găsit în MLflow Registry: {e}")
    print("    Încerc să încarc din fișiere locale...")
    
    # OPȚIUNEA 2: Încarcă din fișiere pickle locale (backup)
    # Caută fișierele .pkl disponibile în folderul models/
    possible_model_files = [
        "web_traffic_model_v1.pkl",      # Fișier existent în folderul models/
        "linear_regression_model.pkl",   # Nume generic
        "model.pkl"                      # Nume alternativ
    ]
    
    model = None
    for model_filename in possible_model_files:
        model_path = os.path.join(MODELS_DIR, model_filename)
        if os.path.exists(model_path): # Verificăm dacă fișierul există
            try:
                import pickle 
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                print(f" Model încărcat din fișier local: {model_path}")
                break
            except Exception as load_error:
                print(f" Eroare la încărcare {model_filename}: {load_error}")
    
    if model is None:
        print("\n NU S-A GĂSIT NICIUN MODEL!")
        print("   Soluții:")
        print("   1. Rulează mai întâi: python mlflow_ml_model_web_traffic.py")
        print("   2. SAU salvează manual un model în models/web_traffic_model_v1.pkl")

# ===========================
# 2. PREGĂTIRE DATE NOI
# ===========================
print("\n[2/4] Pregătire date noi pentru predicție...")

# EXEMPLU 1: Date simulate (în realitate ar veni din API Google Analytics)
print("\n SCENARIU 1: Predicție pentru 5 sesiuni noi")
print("-" * 60)

new_data = pd.DataFrame({
    'pageviews': [8, 15, 25, 5, 30],
    'visitNumber': [1, 2, 3, 1, 4],
    'hits': [12, 22, 40, 8, 50]
})

print("Date de intrare:")
print(new_data.to_string(index=False))

# Definim feature-urile (TREBUIE să fie EXACT aceleași ca la antrenare!)
# Vezi mlflow_ml_model_web_traffic.py, linia 69
required_features = ['pageviews', 'visitNumber', 'hits']

# Validare: verificăm că avem toate coloanele necesare
missing_features = [f for f in required_features if f not in new_data.columns]
if missing_features:
    raise ValueError(f" Lipsesc coloane: {missing_features}")
print(f"\n Toate feature-urile necesare sunt prezente: {', '.join(required_features)}")

# ===========================
# 3. SCALARE (OBLIGATORIU!)
# ===========================
print("\n[3/4] Scalare date (cu același StandardScaler din antrenament)...")

X_new = new_data[required_features].copy()
X_new_scaled = scaler.transform(X_new)

print(" Date scalate (valorile sunt acum standardizate)")

# ===========================
# 4. PREDICȚIE
# ===========================
print("\n[4/4] Generare predicții...")

if model is not None: # Verificăm că modelul a fost încărcat cu succes
    # Predicție
    y_predictions = model.predict(X_new_scaled)
    
    # Asigurăm valori non-negative (timpul nu poate fi negativ)
    y_predictions = np.maximum(y_predictions, 0)
    
    # Adăugăm predicțiile în DataFrame
    new_data['timeOnSite_predicted_seconds'] = y_predictions
    new_data['timeOnSite_predicted_minutes'] = y_predictions / 60
    
    print("\n" + "="*80)
    print("REZULTATE PREDICȚII")
    print("="*80)
    print(new_data.to_string(index=False))
    
    # Statistici predicții
    print(f"\n STATISTICI PREDICȚII:")
    print(f"   Timp minim prezis:  {y_predictions.min():.1f} secunde ({y_predictions.min()/60:.1f} minute)")
    print(f"   Timp maxim prezis:  {y_predictions.max():.1f} secunde ({y_predictions.max()/60:.1f} minute)")
    print(f"   Timp mediu prezis:  {y_predictions.mean():.1f} secunde ({y_predictions.mean()/60:.1f} minute)")
    print(f"   Deviație standard:  {y_predictions.std():.1f} secunde")
    
    # ===========================
    # 5. INTERPRETARE ȘI INSIGHTS
    # ===========================
    print("\n" + "="*80)
    print("INTERPRETARE & INSIGHTS")
    print("="*80)
    
    # Identificăm sesiunea cu engagement cel mai mare
    best_idx = y_predictions.argmax() # .argmax() returnează indexul valorii maxime din array
    worst_idx = y_predictions.argmin() # .argmin() returnează indexul valorii minime din array
    
    print(f"\n SESIUNE CU CEL MAI MARE ENGAGEMENT (linia {best_idx + 1}):")
    print(f"   Pageviews: {new_data.iloc[best_idx]['pageviews']}")
    print(f"   VisitNumber: {new_data.iloc[best_idx]['visitNumber']}")
    print(f"   Hits: {new_data.iloc[best_idx]['hits']}")
    print(f"   ➜ Timp prezis: {y_predictions[best_idx]:.1f}s ({y_predictions[best_idx]/60:.1f} min)")
    
    print(f"\n SESIUNE CU CEL MAI MIC ENGAGEMENT (linia {worst_idx + 1}):")
    print(f"   Pageviews: {new_data.iloc[worst_idx]['pageviews']}")
    print(f"   VisitNumber: {new_data.iloc[worst_idx]['visitNumber']}")
    print(f"   Hits: {new_data.iloc[worst_idx]['hits']}")
    print(f"   ➜ Timp prezis: {y_predictions[worst_idx]:.1f}s ({y_predictions[worst_idx]/60:.1f} min)")
    
    # Clasificare engagement
    print(f"\n CLASIFICARE ENGAGEMENT:")
    high_engagement = (y_predictions > 300).sum()
    medium_engagement = ((y_predictions >= 180) & (y_predictions <= 300)).sum()
    low_engagement = (y_predictions < 180).sum()
    
    print(f" Engagement RIDICAT (>5 min):     {high_engagement} sesiuni")
    print(f" Engagement MEDIU (3-5 min):      {medium_engagement} sesiuni")
    print(f" Engagement SCĂZUT (<3 min):      {low_engagement} sesiuni")
    
    # ===========================
    # 6. SALVARE REZULTATE
    # ===========================
    print("\n[5/5] Salvare rezultate...")
    
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    new_data.to_csv(output_path, index=False)
    print(f" Predicții salvate: {output_path}")
    
    # ===========================
    # BONUS: SCENARIU WHAT-IF
    # ===========================
    print("\n" + "="*80)
    print("BONUS: ANALIZĂ 'WHAT-IF' (Simulări)")
    print("="*80)
    
    print("\n Ce se întâmplă dacă DUBLĂM numărul de pageviews?")
    
    X_what_if_data = new_data[required_features].copy()
    X_what_if_data['pageviews'] = X_what_if_data['pageviews'] * 2
    
    X_what_if_scaled = scaler.transform(X_what_if_data)
    y_what_if_predictions = model.predict(X_what_if_scaled)
    y_what_if_predictions = np.maximum(y_what_if_predictions, 0) # 0 inseamna fara timp negativ 
    
    y_improvement = y_what_if_predictions - y_predictions
    y_improvement_pct = (y_improvement / y_predictions) * 100
    
    comparison = pd.DataFrame({
        'Pageviews_Original': new_data['pageviews'],
        'Pageviews_Doubled': X_what_if_data['pageviews'],
        'TimeOnSite_Original': y_predictions,
        'TimeOnSite_WhatIf': y_what_if_predictions,
        'Improvement_Seconds': y_improvement,
        'Improvement_Percent': y_improvement_pct
    })
    
    print("\nComparație:")
    print(comparison.to_string(index=False))
    print(f"\n Impact mediu: +{y_improvement.mean():.1f} secunde (+{y_improvement_pct.mean():.1f}%)")
    
    print("\n" + "="*80)
    print(" PREDICȚIE COMPLETĂ!")
    print("="*80)
    print("\n NEXT STEPS:")
    print("   1. Integrare cu API Google Analytics pentru date real-time")
    print("   2. Seteare alertă când predicted timeOnSite < threshold")
    print("   3. Creat dashboard cu vizualizări (Power BI / Tableau)")
    print("   4. Implementare re-training lunar cu date noi")
    
else:
    print("\n Model nu a fost găsit! Rulează mai întâi 'mlflow_ml_model_web_traffic.py'")

# ===========================
# EXEMPLU COD PENTRU INTEGRARE REAL-TIME
# ===========================
print("\n" + "="*80)
print("EXEMPLU COD PENTRU INTEGRARE ÎN PRODUCȚIE")
print("="*80)

print("""
# ========================================
# INTEGRARE CU GOOGLE ANALYTICS API
# ========================================

from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest

def get_realtime_data():
    '''Extrage date real-time din Google Analytics'''
    client = BetaAnalyticsDataClient()
    
    request = RunReportRequest(
        property=f"properties/YOUR_PROPERTY_ID",
        dimensions=[{"name": "sessionId"}],
        metrics=[
            {"name": "screenPageViews"},    # Maps to 'pageviews'
            {"name": "sessionNumber"},      # Maps to 'visitNumber'
            {"name": "eventCount"}          # Maps to 'hits'
        ],
        date_ranges=[{"start_date": "today", "end_date": "today"}]
    )
    
    response = client.run_report(request)
    
    # Procesează response și creează DataFrame cu coloanele corecte
    df = pd.DataFrame({
        'pageviews': [...],      # Din screenPageViews
        'visitNumber': [...],    # Din sessionNumber
        'hits': [...]            # Din eventCount
    })
    return df

def predict_and_alert():
    '''Rulează predicție și trimite alertă dacă e necesar'''
    new_data = get_realtime_data()
    
    # Asigură-te că ai coloanele corecte: pageviews, visitNumber, hits
    X_new = new_data[['pageviews', 'visitNumber', 'hits']]
    X_new_scaled = scaler.transform(X_new)
    predictions = model.predict(X_new_scaled)
    
    if predictions.mean() < 180:  # Sub 3 minute
        send_email_alert(" Engagement scăzut detectat!")
        
    return predictions

# Rulează periodic (de ex. cu cron job la fiecare oră)
# 0 * * * * /usr/bin/python3 /path/to/predict_and_alert.py
""")

print("\n" + "="*80)
