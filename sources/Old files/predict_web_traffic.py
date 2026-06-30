"""
SCRIPT DE PREDICTIE - PRODUCTIE
================================
Foloseste modelul antrenat pentru a face predictii pe date noi.

Utilizare:
    python predict_web_traffic.py <input_file.csv>
    python predict_web_traffic.py <input_file.csv> <output_file.csv>

Input CSV trebuie sa contina coloanele: pageviews, visitNumber, hits

Autor: Proiect Web Analytics
Data: 2026-05-21
"""

import pandas as pd
import numpy as np
import pickle
import json
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ===========================
# FUNCTII HELPER
# ===========================

def load_model():
    """Incarca modelul antrenat si scaler-ul."""
    try: 
        with open('../models/web_traffic_model.pkl', 'rb') as f: #  'rb' indica faptul ca fisierul va fi deschis in modul de citire binara, ceea ce este necesar pentru a incarca obiecte complexe precum modele de machine learning folosind pickle. Acest mod asigura ca datele sunt citite corect in format binar, pastrand structura si informatiile necesare pentru a putea utiliza modelul ulterior pentru predictii fara pierderi de informatie sau corupere a datelor.
            model = pickle.load(f)
        
        with open('../models/web_traffic_scaler.pkl', 'rb') as f: #  'rb' indica faptul ca fisierul va fi deschis in modul de citire binara, ceea ce este necesar pentru a incarca obiecte complexe precum modele de machine learning folosind pickle. Acest mod asigura ca datele sunt citite corect in format binar, pastrand structura si informatiile necesare pentru a putea utiliza scaler-ul ulterior pentru predictii fara pierderi de informatie sau corupere a datelor.
            scaler = pickle.load(f)
        
        with open('../models/model_metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return model, scaler, metadata
    
    except FileNotFoundError as e:
        print(f" EROARE: Fisier model lipsa!")
        print(f" Rulati mai intai: python train_liniar_model.py")
        sys.exit(1)

def validate_input_data(df, required_features):
    """Valideaza ca input-ul contine toate coloanele necesare."""
    missing_cols = [col for col in required_features if col not in df.columns]
    
    if missing_cols:
        print(f"EROARE: Coloane lipsa din input: {missing_cols}")
        print(f"   Coloane necesare: {required_features}")
        print(f"   Coloane disponibile: {list(df.columns)}")
        sys.exit(1)
    
    # Verificare valori lipsa
    for col in required_features:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f" ATENTIE: Coloana '{col}' are {null_count} valori lipsa")
    
    return True

def make_predictions(model, scaler, df, required_features):
    """Face predictii pe datele de input."""
    # Extrage features
    X = df[required_features].copy()
    
    # Tratare valori lipsa (completare cu median)
    for col in required_features:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            print(f"   → Completat valori lipsa in '{col}' cu median: {median_val:.2f}")
    
    # Standardizare
    X_scaled = scaler.transform(X)
    
    # Predictie
    predictions = model.predict(X_scaled)
    
    # Asigurare ca predictiile sunt non-negative
    predictions = np.maximum(predictions, 0)
    
    return predictions

# ===========================
# MAIN SCRIPT
# ===========================

def main():
    print("="*80)
    print("PREDICTIE WEB TRAFFIC - TIMP PE SITE (timeOnSite)")
    print("="*80)
    
    # Verificare argumente
    if len(sys.argv) < 2: # sys.argv este o lista care contine argumentele liniei de comanda, unde sys.argv[0] este numele scriptului, iar sys.argv[1], sys.argv[2], etc. sunt argumentele suplimentare. In acest caz, verificam daca lungimea listei sys.argv este mai mica de 2, ceea ce inseamna ca nu a fost furnizat niciun argument suplimentar (cum ar fi numele fisierului de input). Daca aceasta conditie este adevarata, afisam un mesaj de eroare si instructiuni de utilizare, apoi iesim din program cu sys.exit(1) pentru a indica o eroare.
        print("\nEROARE: Lipsa fisier input!")
        print("\nUtilizare:")
        print("   python predict_web_traffic.py <input_file.csv>")
        print("   python predict_web_traffic.py <input_file.csv> <output_file.csv>")
        print("\nExemplu:")
        print("   python predict_web_traffic.py new_sessions.csv predictions.csv")
        sys.exit(1)
    
    input_file = sys.argv[1] # sys.argv[1] reprezinta primul argument suplimentar furnizat in linia de comanda, care in acest caz ar trebui sa fie numele fisierului CSV de input care contine datele pentru care dorim sa facem predictii. Acest argument este preluat si stocat in variabila input_file pentru a fi utilizat ulterior in script pentru a incarca datele din fisierul specificat.
    output_file = sys.argv[2] if len(sys.argv) > 2 else None # sys.argv[2] reprezinta al doilea argument suplimentar furnizat in linia de comanda, care in acest caz ar trebui sa fie numele fisierului CSV de output unde dorim sa salvam rezultatele predictiilor. Daca lungimea listei sys.argv este mai mare de 2, inseamna ca a fost furnizat un al doilea argument, iar acesta va fi stocat in variabila output_file. Daca nu a fost furnizat un al doilea argument, output_file va fi setat la None, ceea ce indica faptul ca nu se va specifica un nume de fisier pentru output si scriptul va genera unul automat.
    
    # Verificare existenta fisier input
    if not os.path.exists(input_file):
        print(f"EROARE: Fisierul '{input_file}' nu exista!")
        sys.exit(1)
    
    # ===========================
    # 1. INCARCARE MODEL
    # ===========================
    print("\n[1/5] Incarcare model antrenat...")
    model, scaler, metadata = load_model() 
    print(f" Model incarcat: {metadata['model_type']}")
    print(f" Versiune: {metadata['model_version']}")
    print(f" Antrenat la: {metadata['training_date']}")
    print(f" Performanta (R²): {metadata['performance']['r2_test']:.4f}")
    
    required_features = metadata['features']
    print(f" Features necesare: {required_features}")
    
    # ===========================
    # 2. INCARCARE DATE INPUT
    # ===========================
    print(f"\n[2/5] Incarcare date input din '{input_file}'...")
    try:
        df_input = pd.read_csv(input_file)
        print(f" Date incarcate: {len(df_input):,} randuri, {len(df_input.columns)} coloane")
    except Exception as e:
        print(f" EROARE la citirea fisierului: {e}")
        sys.exit(1)
    
    # ===========================
    # 3. VALIDARE DATE
    # ===========================
    print(f"\n[3/5] Validare date input...")
    validate_input_data(df_input, required_features)
    print(f" Toate coloanele necesare sunt prezente")
    
    # ===========================
    # 4. PREDICTIE
    # ===========================
    print(f"\n[4/5] Generare predictii...")
    start_time = datetime.now()
    predictions = make_predictions(model, scaler, df_input, required_features)
    prediction_time = (datetime.now() - start_time).total_seconds()
    
    print(f" Predictii generate: {len(predictions):,} valori")
    print(f" Timp predictie: {prediction_time:.4f} secunde")
    print(f" Viteza: {len(predictions)/prediction_time:,.0f} predictii/secunda")
    
    # Adauga predictiile la dataframe
    df_output = df_input.copy()
    df_output['timeOnSite_predicted'] = predictions
    df_output['prediction_confidence'] = 'moderate'  # Based on R² = 0.4553
    df_output['prediction_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # ===========================
    # 5. SALVARE / AFISARE REZULTATE
    # ===========================
    print(f"\n[5/5] Rezultate...")
    
    # Statistici predictii
    print(f"\n Statistici predictii:")
    print(f"   Mean: {predictions.mean():.2f} seconds (~{predictions.mean()/60:.1f} minute)")
    print(f"   Median: {np.median(predictions):.2f} seconds")
    print(f"   Std: {predictions.std():.2f} seconds")
    print(f"   Min: {predictions.min():.2f} seconds")
    print(f"   Max: {predictions.max():.2f} seconds")
    
    # Salvare output
    if output_file:
        
        df_output.to_csv(output_file, index=False)
        print(f"\n Rezultate salvate in: {output_file}")
    else:
        # Generare nume fisier automat in data/predictions/
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'predictions')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f"predictions_{timestamp}.csv")
        df_output.to_csv(output_file, index=False)
        print(f"\n Rezultate salvate in: {output_file}")
    
    # Afisare sample
    print(f"\ Sample predictii (primele 10 randuri):")
    print("-" * 80)
    sample_cols = required_features + ['timeOnSite_predicted']
    print(df_output[sample_cols].head(10).to_string(index=False))
    print("-" * 80)
    
    # ===========================
    # FINALIZARE
    # ===========================
    print("\n" + "="*80)
    print(" PREDICTII GENERATE CU SUCCES!")
    print("="*80)
    
    print(f"\nFisier output: {output_file}")
    print(f"Total predictii: {len(predictions):,}")
    print(f"Timp procesare: {prediction_time:.4f} secunde")
    print(f"Nota: Aceste predictii au o acuratete (R²) de {metadata['performance']['r2_test']:.1%}")
    print(f"Eroarea medie asteptata: ±{metadata['performance']['mae_test']:.0f} secunde")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
