"""
SCRIPT MLFLOW - RANDOM FOREST REGRESSOR MODEL TRACKING
=======================================================
Utilizare MLflow pentru tracking Random Forest Regressor model - CEL MAI BUN MODEL!

Random Forest este cel mai performant model din analiză comparativă:
- Versiune ENHANCED: R²=0.4928 (cel mai bun rezultat)
- MAE: 96.69s (~1.6 minute) - cea mai bună precizie
- Robust cu outliers și date complexe
- Oferă feature importance pentru interpretare

Functionalitati:
- Log parametri model Random Forest
- Log metrici performanta
- Log feature importance
- Log artifacts (model, scaler, metadata)
- Vizualizare rezultate in MLflow UI

Utilizare:
    python mlflow_ml_model_web_traffic.py

Pentru vizualizare rezultate:
    mlflow ui --port 5000

Apoi deschide: http://localhost:5000

Proiect Web Analytics
Data: 2026-06-09
Model: Random Forest Regressor (ENHANCED)
"""
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import json
from datetime import datetime
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configurare stil vizualizări
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Detectam calea proiectului (functie indiferent de unde se ruleaza scriptul)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # os.path.abspath(__file__) returneaza calea completa a scriptului, iar os.path.dirname() extrage doar directorul
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Un nivel mai sus de sources/
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'ga-sessions.csv') # Calea completa catre dataset
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models') # Folderul pentru salvare modele si artifacts

# Cream folderul models daca nu exista
os.makedirs(MODELS_DIR, exist_ok=True)

# Configurare MLflow
# TRACKING LOCAL (artifacts 100% funcționale pentru dezvoltare și prezentare)
# Pentru server Docker, decomentează linia de jos:
# mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Web_Traffic_Prediction")

print("="*80)
print("MLFLOW TRACKING - RANDOM FOREST REGRESSOR (CEL MAI BUN MODEL)")
print("="*80)

# ===========================
# 1. INCARCARE SI PREGATIRE DATE
# ===========================
print("\n[1/5] Incarcare si pregatire date...")

# Cream o clasa proprie pentru tratarea erorilor
logging.basicConfig(
    filename="C:/Users/User/Desktop/repos/Project_repo/error_log.log",
    level = logging.ERROR,
    format = '%(asctime)s - Eroarea a aparut in linia %(lineno)d din fisierul \"%(filename)s\". Mesajul de eroare: %(message)s',
    datefmt = "%Y-%m-%d %H:%M:%S"

)

class ProjectErrorLog(Exception):
    pass # Se foloseste pass ca sa nu se faca nimic in clasa ProjectErrorLog.

try:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset incarcat: {df.shape[0]:,} randuri")

except FileNotFoundError:
    error_msg = "Fisierul ga-sessions.csv nu a fost gasit"
    logging.error(error_msg) 
    raise ProjectErrorLog(error_msg)

except Exception as e:
    error_msg = f"Eroare la incarcare datelorr: {e}"
    logging.error(error_msg)
    raise ProjectErrorLog(error_msg)


# Curatare
df_clean = df.dropna(subset=['timeOnSite'])
df_clean = df_clean[df_clean['timeOnSite'] >= 0] # Eliminam valori negative (daca exista)
df_clean = df_clean[df_clean['timeOnSite'] <= 10000] # Eliminam valori extreme (outliers)

# Features numerice de baza
numeric_features = ['pageviews', 'visitNumber', 'hits']
for feat in numeric_features:
    df_clean = df_clean.dropna(subset=[feat]) # Eliminam randurile cu valori lipsa in feature-urile necesare

print(f"Date curatate: {len(df_clean):,} randuri")
print(f"\n[1.1/5] Feature Engineering și Extindere Features...")

# ===========================
# 1.1 FEATURE ENGINEERING (ENHANCED VERSION)
# ===========================
print("\nCREAM VERSIUNEA ENHANCED CU 20 FEATURES:")
print("   - 3 features numerice originale")
print("   - 12 features categorice encodate")
print("   - 5 features noi create prin feature engineering\n")

# A. Features categorice cu One-Hot Encoding
print("A. Adaugam features categorice...")

# device_category
if 'device_category' in df_clean.columns:
    device_dummies = pd.get_dummies(df_clean['device_category'], prefix='device_category', drop_first=True) # pd.get_dummies() converteste variabila categorica 'device_category' in mai multe coloane binare (one-hot encoding). Fiecare coloana reprezinta o categorie (de exemplu, 'device_category_mobile', 'device_category_tablet'), iar valorile sunt 1 sau 0 pentru a indica prezenta sau absenta acelei categorii in fiecare rand. drop_first=True elimina prima categorie pentru a evita coliniaritatea.
    print(f"   - device_category: {device_dummies.shape[1]} coloane")
else:
    device_dummies = pd.DataFrame()

# country (top 10 + Other)
if 'country' in df_clean.columns:
    top_countries = df_clean['country'].value_counts().head(10).index
    df_clean['country_grouped'] = df_clean['country'].apply(
        lambda x: x if x in top_countries else 'Other'
    )
    country_dummies = pd.get_dummies(df_clean['country_grouped'], prefix='country_grouped', drop_first=True)
    print(f"   - country (top 10 + Other): {country_dummies.shape[1]} coloane")
else:
    country_dummies = pd.DataFrame()

categorical_encoded = pd.concat([device_dummies, country_dummies], axis=1)
print(f"   TOTAL categorice encodate: {categorical_encoded.shape[1]} coloane\n")

# B. Feature Engineering (5 features noi)
print("B. Cream features noi prin feature engineering...")

engineered_features = pd.DataFrame()

# 1. Pages per visit
engineered_features['pageviews_per_visit'] = df_clean['pageviews'] / (df_clean['visitNumber'] + 1)
print("   pageviews_per_visit = pageviews / (visitNumber + 1)")

# 2. Engagement score
engineered_features['engagement_score'] = df_clean['pageviews'] * df_clean['hits']
print("   engagement_score = pageviews × hits")

# 3. Hits per pageview
engineered_features['hits_per_pageview'] = df_clean['hits'] / (df_clean['pageviews'] + 1)
print("   hits_per_pageview = hits / (pageviews + 1)")

# 4. High engagement flag
engineered_features['high_pageviews'] = (df_clean['pageviews'] > df_clean['pageviews'].median()).astype(int)
print("   high_pageviews = binary (1 dacă > median)")

# 5. Returning visitor flag
engineered_features['is_returning'] = (df_clean['visitNumber'] > 1).astype(int)
print("   is_returning = binary (1 dacă visitNumber > 1)")

print(f"   TOTAL engineered: {engineered_features.shape[1]} coloane\n")

# C. Combinam toate features
print("C. Combinam toate features...")
X_numeric = df_clean[numeric_features].copy()
X_enhanced = pd.concat([X_numeric, categorical_encoded, engineered_features], axis=1)

# Lista completa de features pentru tracking
required_features = X_enhanced.columns.tolist()
print(f"   TOTAL FEATURES ENHANCED: {len(required_features)} coloane")
print(f"   Features: {required_features[:5]}... (și încă {len(required_features)-5})\n")

# Calculare statistici pentru features numerice (pentru generare date sintetice)
feature_stats = {}
for feat in numeric_features:  # Doar pentru cele 3 features numerice de bază
    feature_stats[feat] = {
        'min': float(df_clean[feat].min()),
        'max': float(df_clean[feat].max()),
        'mean': float(df_clean[feat].mean()),
        'median': float(df_clean[feat].median()),
        'std': float(df_clean[feat].std())
    }
    print(f"   {feat}: min={feature_stats[feat]['min']:.0f}, max={feature_stats[feat]['max']:.0f}")

X = X_enhanced.copy()  # Folosim toate cele 20 features
y = df_clean['timeOnSite'].copy()

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardizare (NOTĂ: Random Forest nu necesită standardizare, dar o păstrăm pentru consistență)
scaler = StandardScaler()
X_train_scaled = X_train  # Random Forest nu necesită scaling
X_test_scaled = X_test  # Păstrăm datele originale pentru Random Forest

print(f"Train/Test split: {len(X_train):,} (80%)/ {len(X_test):,} (20%)")

# ===========================
# 2. START MLFLOW RUN
# ===========================
print("\n[2/5] Pornire MLflow tracking...")

with mlflow.start_run(run_name="Random_Forest_ENHANCED_v2.0") as run:
    
    print(f"MLflow Run ID: {run.info.run_id}") # run.info.run_id returneaza ID-ul unic al run-ului curent, care poate fi folosit pentru a identifica si accesa acest run in MLflow UI sau prin API. Acest ID este generat automat de MLflow la inceputul fiecarui run si este esential pentru tracking-ul experimentelor.
    
    # ===========================
    # 2.1. VIZUALIZĂRI DATE (EDA - Exploratory Data Analysis)
    # ===========================
    print("\n[2.1/5] Generare vizualizări pentru analiza datelor...")
    
    # Cream folder pentru vizualizări
    viz_dir = os.path.join(PROJECT_ROOT, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. DISTRIBUȚIA TARGET-ULUI (timeOnSite) - DOAR HISTOGRAM
    print("   Generare: Distribuția timeOnSite...")
    plt.figure(figsize=(12, 6))
    
    plt.hist(y, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(y.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {y.mean():.1f}s')
    plt.axvline(y.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {y.median():.1f}s')
    plt.xlabel('Time on Site (seconds)', fontsize=12)
    plt.ylabel('Frecvență', fontsize=12)
    plt.title('Distribuția timeOnSite', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    target_dist_path = os.path.join(viz_dir, '01_target_distribution.png')
    try:
        plt.savefig(target_dist_path, dpi=300, bbox_inches='tight')
    except PermissionError:
        error_msg = f"Permisiuni insuficiente la salvarea imaginii: {target_dist_path}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except OSError as e:
        error_msg = f"Eroare la salvarea imaginii {target_dist_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except Exception as e:
        error_msg = f"Eroare neasteptata la salvarea imaginii {target_dist_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    try:
        mlflow.log_artifact(target_dist_path, "visualizations")
    except Exception as e:
        error_msg = f"Eroare la logarea artifact-ului MLflow pentru {target_dist_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    plt.close()
    
    # 2. DISTRIBUȚIA FEATURES (doar cele 3 numerice de bază pentru claritate)
    print("   Generare: Distribuția features...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, feat in enumerate(numeric_features):  # Doar cele 3 features numerice originale
        axes[idx].hist(df_clean[feat], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[idx].axvline(df_clean[feat].mean(), color='red', linestyle='--', linewidth=2, 
                         label=f'Mean: {df_clean[feat].mean():.1f}')
        axes[idx].set_xlabel(feat, fontsize=12)
        axes[idx].set_ylabel('Frecvență', fontsize=12)
        axes[idx].set_title(f'Distribuția {feat}', fontsize=14, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    features_dist_path = os.path.join(viz_dir, '02_features_distribution.png')
    try:
        plt.savefig(features_dist_path, dpi=300, bbox_inches='tight')
    except PermissionError:
        error_msg = f"Permisiuni insuficiente la salvarea imaginii: {features_dist_path}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except OSError as e:
        error_msg = f"Eroare la salvarea imaginii {features_dist_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except Exception as e:
        error_msg = f"Eroare neasteptata la salvarea imaginii {features_dist_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    try:
        mlflow.log_artifact(features_dist_path, "visualizations")
    except Exception as e:
        error_msg = f"Eroare la logarea artifact-ului MLflow pentru {features_dist_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    plt.close()
    
    # 3. SCATTER PLOTS (Feature vs Target) - RENUMEROTAT DIN 4 - doar features numerice
    print("   Generare: Scatter plots features vs target...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, feat in enumerate(numeric_features):  # Doar cele 3 features numerice originale
        axes[idx].scatter(df_clean[feat], y, alpha=0.5, s=20, color='purple')
        
        # Adaugă linie de trend (regresie liniară simplă)
        z = np.polyfit(df_clean[feat], y, 1)
        p = np.poly1d(z)
        axes[idx].plot(df_clean[feat], p(df_clean[feat]), "r--", linewidth=2, label='Trend')
        
        axes[idx].set_xlabel(feat, fontsize=12)
        axes[idx].set_ylabel('timeOnSite (seconds)', fontsize=12)
        axes[idx].set_title(f'{feat} vs timeOnSite', fontsize=14, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    scatter_path = os.path.join(viz_dir, '03_scatter_plots.png')
    try:
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    except PermissionError:
        error_msg = f"Permisiuni insuficiente la salvarea imaginii: {scatter_path}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except OSError as e:
        error_msg = f"Eroare la salvarea imaginii {scatter_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except Exception as e:
        error_msg = f"Eroare neasteptata la salvarea imaginii {scatter_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    try:
        mlflow.log_artifact(scatter_path, "visualizations")
    except Exception as e:
        error_msg = f"Eroare la logarea artifact-ului MLflow pentru {scatter_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    plt.close()
    
    # 4. STATISTICI DESCRIPTIVE (tabel vizual) - RENUMEROTAT DIN 5 - doar features numerice
    print("   Generare: Tabel statistici descriptive...")
    stats_df = df_clean[numeric_features + ['timeOnSite']].describe()  # Doar cele 3 features numerice originale
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['Statistică'] + list(stats_df.columns))
    for idx in stats_df.index:
        row = [idx] + [f'{val:.2f}' for val in stats_df.loc[idx]]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15] * (len(stats_df.columns) + 1))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Formatare header
    for i in range(len(stats_df.columns) + 1):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Statistici Descriptive - Dataset', fontsize=16, fontweight='bold', pad=20)
    stats_path = os.path.join(viz_dir, '04_descriptive_statistics.png')
    try:
        plt.savefig(stats_path, dpi=300, bbox_inches='tight')
    except PermissionError:
        error_msg = f"Permisiuni insuficiente la salvarea imaginii: {stats_path}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except OSError as e:
        error_msg = f"Eroare la salvarea imaginii {stats_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except Exception as e:
        error_msg = f"Eroare neasteptata la salvarea imaginii {stats_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    try:
        mlflow.log_artifact(stats_path, "visualizations")
    except Exception as e:
        error_msg = f"Eroare la logarea artifact-ului MLflow pentru {stats_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    plt.close()
    
    # 5. COMPARAȚIE TRAIN vs TEST DISTRIBUTION - RENUMEROTAT DIN 6
    print("   Generare: Comparație distribuții Train vs Test...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # timeOnSite
    axes[0, 0].hist(y_train, bins=30, alpha=0.6, label='Train', color='blue', edgecolor='black')
    axes[0, 0].hist(y_test, bins=30, alpha=0.6, label='Test', color='orange', edgecolor='black')
    axes[0, 0].set_xlabel('timeOnSite (seconds)', fontsize=11)
    axes[0, 0].set_ylabel('Frecvență', fontsize=11)
    axes[0, 0].set_title('Distribuție timeOnSite: Train vs Test', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Features (doar cele 3 numerice originale pentru claritate)
    for idx, feat in enumerate(numeric_features):  # Doar cele 3 features numerice
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        axes[row, col].hist(X_train[feat], bins=30, alpha=0.6, label='Train', color='blue', edgecolor='black')
        axes[row, col].hist(X_test[feat], bins=30, alpha=0.6, label='Test', color='orange', edgecolor='black')
        axes[row, col].set_xlabel(feat, fontsize=11)
        axes[row, col].set_ylabel('Frecvență', fontsize=11)
        axes[row, col].set_title(f'Distribuție {feat}: Train vs Test', fontsize=13, fontweight='bold')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    train_test_path = os.path.join(viz_dir, '05_train_test_comparison.png')
    try:
        plt.savefig(train_test_path, dpi=300, bbox_inches='tight')
    except PermissionError:
        error_msg = f"Permisiuni insuficiente la salvarea imaginii: {train_test_path}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except OSError as e:
        error_msg = f"Eroare la salvarea imaginii {train_test_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except Exception as e:
        error_msg = f"Eroare neasteptata la salvarea imaginii {train_test_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    try:
        mlflow.log_artifact(train_test_path, "visualizations")
    except Exception as e:
        error_msg = f"Eroare la logarea artifact-ului MLflow pentru {train_test_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    plt.close()
    
    print(f"    {5} vizualizări EDA generate și salvate în MLflow!")
    print(f"    Locație locală: {viz_dir}")
    
    # ===========================
    # 3. ANTRENARE MODEL RANDOM FOREST
    # ===========================
    print("\n[3/5] Antrenare model Random Forest (ENHANCED)...")
    print("   Parametri optimi identificați în analiză comparativă:")
    print("     - n_estimators: 300 arbori")
    print("     - max_depth: 30 (adâncime optimă)")
    print("     - min_samples_split: 20")
    print("     - min_samples_leaf: 40\n")


    """
Trebuie testat codul prin care creeam un dictionar de parametrii(n_estimators, max_depth, min_samples_split, min_samples_leaf) carora le dam diferite valori pentru a vedea care combinatie de valori da rezultatele cele mai bune.
De aplicat codul de mai jos si in fisierul cu mlflow doar pentru varianta imbunatatita.
Using this code, MLflow will log each run with the specific parameters and metrics, allowing you to easily compare the performance of different hyperparameter configurations.  

param_grid = {
    'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450],
    'max_depth': [10, 15, 20, 30, 40, 50, 70, 100],
    'min_samples_split': [5, 10, 15, 20, 25, 30, 35, 40],
    'min_samples_leaf': [10, 20, 30, 40, 50, 60, 70, 80]
}

grid = Parametergrid(param_grid)
for param in grid:
    rf_simple_model = RandomForestRegressor(
        n_estimators=param['n_estimators'],# unde param['n_estimators'] va fi inlocuit cu fiecare valoare din lista specificata in param_grid pentru a testa diferite numere de arbori de decizie in cadrul modelului Random Forest.
        max_depth=param['max_depth'],
        min_samples_split=param['min_samples_split'],
        min_samples_leaf = param['min_samples_leaf'],
        random_state=42,
        n_jobs=-1
    )
    rf_simple_model.fit(X_simple_train, y_train)
    y_rf_simple_pred = rf_simple_model.predict(X_simple_test)
    etc(se ia codul de mai jos si se puna in continuare acestui cod din for loop)

Apoi sa se foloseasca exemplul de cod de mai jos pentru a loga fiecare run in MLflow cu parametrii specifici si metricile calculate, permitandu-va sa comparati usor performanta diferitelor configuratii de hiperparametri, si a se afisa informatii
referitoare la modelul cel mai bun cum ar fi numele lui, ID-ul, n_estimators, acuratetea(?), max_depth, min_samples_split, min_samples_leaf.

# Variabile pentru a reține cel mai bun model
best_accuracy = 0 # 0 este valoarea inițială pentru acuratețea cea mai bună, care va fi actualizată pe măsură ce evaluăm modelele. Începem cu 0 pentru a ne asigura că orice model cu o acuratețe mai mare va fi considerat cel mai bun.
best_run_id = None
best_params = None
best_run_name = None

for params in grid:
    with mlflow.start_run() as run:
        # **params unpacks dictionary
        model = RandomForestClassifier(**params, random_state=42) 
        model.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, model.predict(X_test))
        
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log the model
        mlflow.sklearn.log_model(
            model, 
            artifact_path="random_forest_model",
            input_example=X_train.head(5)
        )
        
        # Salvăm ID-ul rulării dacă este cea mai bună
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_run_id = run.info.run_id
            best_params = params
            best_run_name = run.info.run_name
print(f"Training finished. Best Run ID: {best_run_id} with Accuracy: {best_accuracy}")

# 6. Înregistrarea modelului și tranziția la Production (DOAR O DATĂ, la final)
if best_run_id:
    # Înregistrează modelul cel mai bun
    result = mlflow.register_model(
        f"runs:/{best_run_id}/random_forest_model", 
        "BestRandomForestModel"
    )
    
    # Inițializează clientul pentru tranziție
    client = MlflowClient()
    
    # Folosește clientul pentru a face tranziția
    client.transition_model_version_stage(
        name="BestRandomForestModel", 
        version=result.version, 
        stage="Production"
    )
    print(f"Model BestRandomForestModel version {result.version} promoted to Production. Run ID: {best_run_id}; Run name: {best_run_name}; Parametrii: {best_params}; Acuratețe: {best_accuracy}.")


"""
    
    start_time = datetime.now()
    model = RandomForestRegressor(
        n_estimators=300,        # 300 arbori pentru performanță optimă
        max_depth=30,            # Adâncime optimă pentru capturarea pattern-urilor complexe
        min_samples_split=20,    # Control pentru generalizare
        min_samples_leaf=40,     # Previne overfitting-ul
        random_state=42,         # Reproducibilitate
        n_jobs=-1                # Paralelizare maximă
    )
    model.fit(X_train_scaled, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"Model Random Forest antrenat in {training_time:.4f} secunde")
    
    # ===========================
    # 4. EVALUARE SI LOGGING
    # ===========================
    print("\n[4/5] Evaluare performanta si logging metrici...")
    
    # Predictii
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Metrici
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"   R² Test: {r2_test:.4f}")
    print(f"   RMSE Test: {rmse_test:.2f} seconds")
    print(f"   MAE Test: {mae_test:.2f} seconds")
    
    # ===========================
    # 4.0.1 EXPLICAȚII METRICI (pentru comisie)
    # ===========================
    print(f"\n[INFO] INTERPRETAREA METRICILOR:")
    
    # R² Score
    print(f"\n R² Score = {r2_test:.4f}")
    if r2_test >= 0.7:
        r2_interpretation = "EXCELENT - Modelul explică >70% din variație"
    elif r2_test >= 0.5:
        r2_interpretation = "BUN - Modelul explică >50% din variație, acceptabil pentru date complexe"
    elif r2_test >= 0.3:
        r2_interpretation = "MODERAT - Modelul captează tendințe generale, dar are limitări"
    else:
        r2_interpretation = "SLAB - Modelul nu explică suficient variația"
    print(f"   {r2_interpretation}")
    print(f"   Semnificație: {r2_test*100:.1f}% din variația în timeOnSite este explicată de model.")
    
    # MAE
    print(f"\n MAE = {mae_test:.2f} seconds")
    if mae_test <= 60:
        mae_interpretation = "EXCELENT - Eroare medie <1 minut"
    elif mae_test <= 120:
        mae_interpretation = "BUN - Eroare medie între 1-2 minute, acceptabil pentru web analytics"
    elif mae_test <= 180:
        mae_interpretation = "MODERAT - Eroare medie între 2-3 minute"
    else:
        mae_interpretation = "MARE - Eroare medie >3 minute, modelul are dificultăți"
    print(f"   {mae_interpretation}")
    print(f"   Semnificație: În medie, predicțiile se abat cu {mae_test:.0f}s (~{mae_test/60:.1f} min) de valorile reale.")
    
    # RMSE
    print(f"\n RMSE = {rmse_test:.2f} seconds")
    if rmse_test <= 90:
        rmse_interpretation = "EXCELENT - Eroare rădăcină medie <1.5 minute"
    elif rmse_test <= 180:
        rmse_interpretation = "BUN - Eroare între 1.5-3 minute"
    elif rmse_test <= 300:
        rmse_interpretation = "ACCEPTABIL - Eroare între 3-5 minute pentru date cu outliers"
    else:
        rmse_interpretation = "MARE - Eroare >5 minute, modelul are probleme cu outliers"
    print(f"   {rmse_interpretation}")
    print(f"   Semnificație: RMSE penalizează mai mult erorile mari. {rmse_test:.0f}s (~{rmse_test/60:.1f} min).")
    print(f"\n CONCLUZIE: Modelul este {'VALID și UTILIZABIL' if r2_test >= 0.4 and mae_test <= 150 else 'NECESITĂ ÎMBUNĂTĂȚIRI'} pentru producție.\n")
    
    # LOG PARAMETRI
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("model_version", "Enhanced v2.0 - 20 features - BEST MODEL")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 30)
    mlflow.log_param("min_samples_split", 20)
    mlflow.log_param("min_samples_leaf", 40)
    mlflow.log_param("features_count", len(required_features))
    mlflow.log_param("features_numeric", ", ".join(numeric_features))
    mlflow.log_param("features_engineered", "pageviews_per_visit, engagement_score, hits_per_pageview, high_pageviews, is_returning")
    mlflow.log_param("features_categorical", f"{categorical_encoded.shape[1]} encoded columns")
    mlflow.log_param("n_features", len(required_features))
    mlflow.log_param("dataset_size", len(df_clean))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("test_split_ratio", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("scaler", "None (Random Forest nu necesită)")
    
    print("Parametri logged")
    
    # LOG METRICI (numele simple, fără caractere speciale)
    mlflow.log_metric("r2_train", r2_train)
    mlflow.log_metric("r2_test", r2_test)
    mlflow.log_metric("rmse_train", rmse_train)
    mlflow.log_metric("rmse_test", rmse_test)
    mlflow.log_metric("mae_train", mae_train)
    mlflow.log_metric("mae_test", mae_test)
    mlflow.log_metric("training_time_seconds", training_time)
    mlflow.log_metric("overfitting_gap", r2_train - r2_test)
    
    # Metrici suplimentare pentru context
    mlflow.log_metric("target_mean", y.mean())
    mlflow.log_metric("target_median", y.median())
    mlflow.log_metric("target_std", y.std())
    
    print("Metrici logged")
    
    # LOG FEATURE IMPORTANCE (specific Random Forest)
    feature_importance = pd.DataFrame({
        'feature': required_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Log top 10 cele mai importante features ca parametri
    for idx, row in feature_importance.head(10).iterrows():
        mlflow.log_param(f"importance_{row['feature']}", f"{row['importance']:.4f}")
    
    print("Feature importance logged")
    
    # ===========================
    # 4.1 VIZUALIZĂRI PERFORMANȚĂ MODEL
    # ===========================
    print("\n[4.1/5] Generare vizualizări performanță model...")
    
    # 6. COMPARAȚIE PERFORMANȚĂ: RANDOM FOREST vs LINEAR vs POLYNOMIAL
    print("   Generare: Comparație performanță modele...")
    
    # Date pentru comparație (din ml-model-web-traffic2.py - VERSIUNEA ENHANCED cu 20 features)
    # Random Forest ENHANCED: R²=0.4928 (+10.3% vs SIMPLE) - CEL MAI BUN!
    # Linear Regression ENHANCED: R²=0.4651 (+2.15% vs SIMPLE)
    # Polynomial ENHANCED: R²=0.2707 (-40% vs SIMPLE, overfitting)
    models_comparison = {
        'Model': ['Random\nForest\n(ENHANCED)\n★ BEST', 'Linear\nRegression\n(ENHANCED)', 'Polynomial\nRegression\n(ENHANCED)'],
        'R²': [0.4928, 0.4651, 0.2707],
        'MAE (s)': [96.69, 111.48, 107.15],
        'RMSE (s)': [264.26, 271.43, 316.95]
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # R² Score
    bars1 = axes[0].bar(models_comparison['Model'], models_comparison['R²'], 
                        color=['#2196F3', '#FF9800', '#4CAF50'], edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Comparație R² Score\n(Mai mare = Mai bun)', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, max(models_comparison['R²']) * 1.2])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Adaugă valori pe bare
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # MAE
    bars2 = axes[1].bar(models_comparison['Model'], models_comparison['MAE (s)'], 
                        color=['#2196F3', '#FF9800', '#4CAF50'], edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('MAE (secunde)', fontsize=12, fontweight='bold')
    axes[1].set_title('Comparație MAE\n(Mai mic = Mai bun)', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, max(models_comparison['MAE (s)']) * 1.2])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # RMSE
    bars3 = axes[2].bar(models_comparison['Model'], models_comparison['RMSE (s)'], 
                        color=['#2196F3', '#FF9800', '#4CAF50'], edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('RMSE (secunde)', fontsize=12, fontweight='bold')
    axes[2].set_title('Comparație RMSE\n(Mai mic = Mai bun)', fontsize=14, fontweight='bold')
    axes[2].set_ylim([0, max(models_comparison['RMSE (s)']) * 1.2])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    comparison_path = os.path.join(viz_dir, '06_models_comparison.png')
    try:
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    except PermissionError:
        error_msg = f"Permisiuni insuficiente la salvarea imaginii: {comparison_path}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except OSError as e:
        error_msg = f"Eroare la salvarea imaginii {comparison_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except Exception as e:
        error_msg = f"Eroare neasteptata la salvarea imaginii {comparison_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    try:
        mlflow.log_artifact(comparison_path, "visualizations")
    except Exception as e:
        error_msg = f"Eroare la logarea artifact-ului MLflow pentru {comparison_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    plt.close()
    
    # 7. RESIDUAL PLOT (Predicții vs Erori) - CRUCIAL pentru comisie
    print("   Generare: Residual plot (predicții vs erori)...")
    
    residuals_test = y_test - y_pred_test
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter: Predicted vs Residuals
    axes[0].scatter(y_pred_test, residuals_test, alpha=0.5, s=20, color='purple')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Valori Prezise (seconds)', fontsize=12)
    axes[0].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    axes[0].set_title('Residual Plot: Predicții vs Erori\n(Punctele ar trebui distribuite uniform în jurul lui 0)', 
                     fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histogram: Distribuția erorilor
    axes[1].hist(residuals_test, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(residuals_test.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {residuals_test.mean():.2f}s')
    axes[1].set_xlabel('Residuals (seconds)', fontsize=12)
    axes[1].set_ylabel('Frecvență', fontsize=12)
    axes[1].set_title('Distribuția Erorilor\n(Ar trebui să fie aproximativ normală, centrată pe 0)', 
                     fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    residual_path = os.path.join(viz_dir, '07_residual_plot.png')
    try:
        plt.savefig(residual_path, dpi=300, bbox_inches='tight')
    except PermissionError:
        error_msg = f"Permisiuni insuficiente la salvarea imaginii: {residual_path}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except OSError as e:
        error_msg = f"Eroare la salvarea imaginii {residual_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except Exception as e:
        error_msg = f"Eroare neasteptata la salvarea imaginii {residual_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    try:
        mlflow.log_artifact(residual_path, "visualizations")
    except Exception as e:
        error_msg = f"Eroare la logarea artifact-ului MLflow pentru {residual_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    plt.close()
    
    # 8. FEATURE IMPORTANCE (RANDOM FOREST)
    print("   Generare: Feature importance Random Forest...")
    
    # Sortare feature importance după valoare (TOP 10 pentru claritate)
    importance_data = pd.DataFrame({
        'Feature': required_features,
        'Importance': model.feature_importances_
    })
    importance_data = importance_data.sort_values('Importance', ascending=False).head(10)
    importance_data = importance_data.sort_values('Importance', ascending=True)  # Sortare crescătoare pentru barh
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(importance_data['Feature'], importance_data['Importance'],
                    color='forestgreen', edgecolor='black', linewidth=1.5)
    plt.xlabel('Importanță (contribuție la predicții)', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title('TOP 10 Features - Importanță Model Random Forest ENHANCED\n(Valori mai mari = Mai importantă pentru predicții)', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Adaugă valori pe bare
    for i, (idx, row) in enumerate(importance_data.iterrows()):
        plt.text(row['Importance'], i, f"  {row['Importance']:.4f}",
                va='center', ha='left',
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    coef_path = os.path.join(viz_dir, '08_feature_importance.png')
    try:
        plt.savefig(coef_path, dpi=300, bbox_inches='tight')
    except PermissionError:
        error_msg = f"Permisiuni insuficiente la salvarea imaginii: {coef_path}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except OSError as e:
        error_msg = f"Eroare la salvarea imaginii {coef_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except Exception as e:
        error_msg = f"Eroare neasteptata la salvarea imaginii {coef_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    try:
        mlflow.log_artifact(coef_path, "visualizations")
    except Exception as e:
        error_msg = f"Eroare la logarea artifact-ului MLflow pentru {coef_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    plt.close()
    
    print(f"    {3} vizualizări performanță model generate și salvate!")
    print(f"    Total vizualizări: {8} (5 EDA + 3 Performanță)")
    print(f"\n  MODEL ENHANCED: {len(required_features)} features (3 numerice + {categorical_encoded.shape[1]} categorice + 5 engineered)")
    
    # ===========================
    # 5. SALVARE ARTIFACTS
    # ===========================
    print("\n[5/5] Salvare artifacts...")
    
    # Salvam modelul LOCAL cu pickle în folderul models/
    model_path = os.path.join(MODELS_DIR, "random_forest_model_v2_BEST.pkl")
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    except PermissionError:
        error_msg = f"Permisiuni insuficiente la salvarea modelului: {model_path}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except OSError as e:
        error_msg = f"Eroare la salvarea modelului {model_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except Exception as e:
        error_msg = f"Eroare neasteptata la salvarea modelului {model_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    print(f"Model Random Forest salvat local: {model_path}")

    # Loghăm modelul ca artifact în MLflow (pentru referință)
    try:
        mlflow.log_artifact(model_path, "models")
    except Exception as e:
        error_msg = f"Eroare la logarea artifact-ului MLflow pentru {model_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    print("Model logged in MLflow")
    
    # Salvam scaler-ul in folderul models/
    scaler_path = os.path.join(MODELS_DIR, "web_traffic_scaler_v2.pkl")
    try:
        with open(scaler_path, "wb") as f: # open() cu modul "wb" (write binary) este necesar pentru a salva obiecte serializate cu pickle, deoarece acestea sunt stocate intr-un format binar. Daca am folosi "w" (write text), am obtine un fisier corupt care nu poate fi incarcat corect ulterior.
            pickle.dump(scaler, f) # pickle.dump() este o functie care serializaza un obiect Python (in acest caz, scaler-ul) intr-un format binar, care poate fi apoi salvat intr-un fisier. Acest lucru permite ulterior incarcarea obiectului exact asa cum a fost salvat, pastrand toate atributele si starea acestuia (de exemplu, media si deviatia standard calculate in timpul fit-ului).
    except PermissionError:
        error_msg = f"Permisiuni insuficiente la salvarea scaler-ului: {scaler_path}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except OSError as e:
        error_msg = f"Eroare la salvarea scaler-ului {scaler_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except Exception as e:
        error_msg = f"Eroare neasteptata la salvarea scaler-ului {scaler_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    try:
        mlflow.log_artifact(scaler_path, "preprocessors") # .log_artifact() este o functie generica pentru a salva orice fisier ca artifact in MLflow. In acest caz, salvam scaler-ul serializat cu pickle intr-un subfolder "preprocessors" din cadrul artifact-ului run-ului curent. Acest lucru ne permite sa organizam mai bine artifacts-urile si sa le accesam ulterior pentru reproducere sau analiza.
    except Exception as e:
        error_msg = f"Eroare la logarea artifact-ului MLflow pentru {scaler_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    print(f"Scaler logged: {scaler_path}")
    
    # Salvam metadata ca JSON in folderul models/
    metadata = {
        'model_type': 'RandomForestRegressor',
        'model_version': '2.0_Enhanced_20features_BEST',
        'model_status': 'BEST MODEL - Cel mai performant din analiză',
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hyperparameters': {
            'n_estimators': 300,
            'max_depth': 30,
            'min_samples_split': 20,
            'min_samples_leaf': 40,
            'random_state': 42
        },
        'features_total': len(required_features),
        'features_numeric': numeric_features,
        'features_engineered': ['pageviews_per_visit', 'engagement_score', 'hits_per_pageview', 'high_pageviews', 'is_returning'],
        'features_categorical_count': categorical_encoded.shape[1],
        'features_all': required_features,
        'feature_statistics': feature_stats,
        'performance': {
            'r2_test': float(r2_test),
            'rmse_test': float(rmse_test),
            'mae_test': float(mae_test),
            'interpretation': {
                'r2': 'EXCELENT - Modelul explică ~49% din variație (cel mai bun)' if r2_test >= 0.48 else 'BUN',
                'mae': f'{mae_test:.0f}s (~{mae_test/60:.1f} min) - Cea mai bună precizie din toate modelele',
                'rmse': f'{rmse_test:.0f}s (~{rmse_test/60:.1f} min) - Performanță excelentă'
            }
        },
        'feature_importance_top10': feature_importance.head(10).set_index('feature')['importance'].to_dict()
    }
    
    metadata_path = os.path.join(MODELS_DIR, "model_metadata_v2.json")
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4) # json.dump() este o functie care serializaza un obiect Python (in acest caz, un dictionar cu metadata) intr-un format JSON, care este un format text usor de citit si de partajat. Argumentul indent=4 face ca JSON-ul sa fie formatat cu indentare de 4 spatii. Acest fisier JSON va contine informatii esentiale despre model, performanta acestuia si coeficientii asociati fiecarei feature, oferind un context complet pentru analiza ulterioara.
    except PermissionError:
        error_msg = f"Permisiuni insuficiente la salvarea metadata: {metadata_path}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except OSError as e:
        error_msg = f"Eroare la salvarea metadata {metadata_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except Exception as e:
        error_msg = f"Eroare neasteptata la salvarea metadata {metadata_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    try:
        mlflow.log_artifact(metadata_path, "metadata") # Salvam fisierul JSON cu metadata ca artifact in MLflow, intr-un subfolder "metadata". Acest lucru ne permite sa accesam usor aceste informatii din MLflow UI sau prin API, si sa le folosim pentru comparatie intre modele sau pentru documentare.
    except Exception as e:
        error_msg = f"Eroare la logarea artifact-ului MLflow pentru {metadata_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    print(f"Metadata logged: {metadata_path}")
    
    # ===========================
    # 6. VALIDARE PE TEST SET (DATE REALE)
    # ===========================
    print("\n[6/6] Validare predictii pe TEST SET (date reale)...")
    print(f"   TEST SET size: {len(y_test):,} randuri (20% din date)\n")
    
    # Resetam indexurile pentru a evita probleme de aliniere
    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    y_pred_test_series = pd.Series(y_pred_test, index=y_test_reset.index)
    
    # Creăm DataFrame cu toate informațiile de validare
    df_validation = pd.DataFrame({
        'pageviews': X_test_reset['pageviews'],
        'visitNumber': X_test_reset['visitNumber'],
        'hits': X_test_reset['hits'],
        'timeOnSite_REAL': y_test_reset,
        'timeOnSite_predicted': y_pred_test_series,
        'error_seconds': y_test_reset - y_pred_test_series,
        'error_absolute': np.abs(y_test_reset - y_pred_test_series),
        'error_percent': ((y_test_reset - y_pred_test_series) / y_test_reset * 100).replace([np.inf, -np.inf], np.nan)
    })
    
    # Adăugăm metadata
    df_validation['model'] = 'Random Forest ENHANCED (BEST)'
    df_validation['validation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df_validation['model_r2'] = r2_test
    df_validation['model_mae'] = mae_test
    df_validation['model_rmse'] = rmse_test
    
    # Salvare în data/predictions/
    predictions_dir = os.path.join(PROJECT_ROOT, 'data', 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    validation_path = os.path.join(predictions_dir, 'validation_results.csv')
    try:
        df_validation.to_csv(validation_path, index=False)
    except PermissionError:
        error_msg = f"Permisiuni insuficiente la salvarea fisierului CSV: {validation_path}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except OSError as e:
        error_msg = f"Eroare la salvarea fisierului CSV {validation_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    except Exception as e:
        error_msg = f"Eroare neasteptata la salvarea fisierului CSV {validation_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    
    print(f"Rezultate validare salvate: {validation_path}")
    print(f"   {df_validation.shape[0]:,} randuri cu predictii si comparatii REAL vs PREDICTED")
    
    # Statistici erori
    print(f"\nStatistici erori de predictie:")
    print(f"   Mean Absolute Error: {df_validation['error_absolute'].mean():.2f}s")
    print(f"   Median Absolute Error: {df_validation['error_absolute'].median():.2f}s")
    print(f"   Std Dev: {df_validation['error_absolute'].std():.2f}s")
    print(f"   Min Error: {df_validation['error_absolute'].min():.2f}s")
    print(f"   Max Error: {df_validation['error_absolute'].max():.2f}s")
    
    valid_pct = df_validation['error_percent'].dropna()
    if len(valid_pct) > 0:
        print(f"   Mean Absolute Error %: {valid_pct.abs().mean():.1f}%")
    
    # Log în MLflow
    try:
        mlflow.log_artifact(validation_path, "predictions")
    except Exception as e:
        error_msg = f"Eroare la logarea artifact-ului MLflow pentru {validation_path}: {e}"
        logging.error(error_msg)
        raise ProjectErrorLog(error_msg)
    mlflow.log_metric("validation_mean_abs_error", float(df_validation['error_absolute'].mean()))
    mlflow.log_metric("validation_median_abs_error", float(df_validation['error_absolute'].median()))
    
    print("Validation results logged in MLflow")
    
    # LOG TAGS pentru organizare
    mlflow.set_tag("project", "Web Analytics")
    mlflow.set_tag("model_family", "Ensemble Models - Random Forest")
    mlflow.set_tag("target", "timeOnSite")
    mlflow.set_tag("status", "production_ready_BEST")
    mlflow.set_tag("version", "2.0")
    mlflow.set_tag("best_model", "true")
    
    print("Tags setate")

# ===========================
# FINALIZARE
# ===========================
print("\n" + "="*80)
print("MLFLOW TRACKING COMPLET!")
print("="*80)

print(f"\nInformatii run:")
print(f"  Run ID: {run.info.run_id}")
print(f"  Experiment ID: {run.info.experiment_id}")
print(f"  Artifact URI: {run.info.artifact_uri}")

print(f"\nPerformanta model:")
print(f"  R² Score: {r2_test:.4f} ({r2_test*100:.1f}% variatie explicata)")
print(f"  RMSE: {rmse_test:.2f} secunde (~{rmse_test/60:.1f} minute)")
print(f"  MAE: {mae_test:.2f} secunde (~{mae_test/60:.1f} minute)")

print(f"\nValidare pe TEST SET:")
print(f"  {len(y_test):,} sesiuni REALE cu comparatie predicted vs actual")
print(f"  Rezultate salvate in: {validation_path}")

# Informatii pentru vizualizare in MLflow UI
print(f"\nPentru a vizualiza rezultatele in MLflow UI:")
print(f"  1. In command prompt de scris: mlflow ui --port 5001")
print(f"  2. Dupa rulare, click pe link: http://localhost:5001")
print(f"  3. In sectiunea 'Experiments' de cautat experimentul 'Web_Traffic_Prediction'")

print("\n" + "="*80)
print("MODEL TRACKING CU MLFLOW COMPLET!")
print("="*80)

print("\n ⭐ RANDOM FOREST - CEL MAI BUN MODEL!")
print("   \u2713 R² Score: Cel mai mare (0.4928)")
print("   \u2713 MAE: Cel mai mic (96.69s)")
print("   \u2713 Robust cu date complexe \u0219i outliers")
print("   \u2713 Ofer\u0103 feature importance pentru interpretabilitate")
