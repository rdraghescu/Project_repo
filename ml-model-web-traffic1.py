"""
SCENARIUL 2: PREDICTIA DE ENGAGEMENT (Time on Site)
====================================================
Acest model utilizeaza date Google Analytics pentru a prezice timpul petrecut pe site (timeOnSite).
Vom testa si compara 3 algoritmi de regresie:
1. Random Forest Regressor
2. Polynomial Regression (grad 2)
3. Linear Regression Multiple

Dataset: ga-sessions.csv (100,000 inregistrari, 19 coloane)
Variabila tinta: timeOnSite (seconds)
Features: pageviews, visitNumber, hits

Scopul este sa identificam cel mai bun model pentru predictia comportamentului utilizatorului.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Importuri pentru modele de regresie
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Importuri pentru procesare si evaluare
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time

print("="*80)
print("SCENARIUL 2: PREDICTIA DE ENGAGEMENT (TIME ON SITE) - ANALIZA COMPARATIVA")
print("="*80)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
DATA_PATH = os.path.join(RAW_DATA_DIR, 'ga-sessions.csv')
X_ENHANCED_PATH = os.path.join(PROCESSED_DATA_DIR, 'X_enhanced.csv')
X_SIMPLE_PATH = os.path.join(PROCESSED_DATA_DIR, 'X_simple.csv')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ===========================
# 1. INCARCAREA DATELOR
# ===========================
print("\n[1/7] Incarcarea datelor...")
data = pd.read_csv(DATA_PATH)
print(f"✓ Date incarcate: {data.shape[0]:,} randuri, {data.shape[1]} coloane")

# Afisam numele coloanelor
print(data.columns.tolist()) 

# Afisam primele randuri
print("\nPrimele 5 randuri din dataset:")
print(data.head())

# Afisam ultimele randuri
print(f"Ultimele randuri din tabela:\n{data.tail()}") 

# De tinut minte urmatoarele coloane pentru predictie:
# visitNumber si date;
# visitNumber si traffic_source;
# visitNumber si browser;
# visitNumber si operating_system;
# visitNumber si device_category;
# visitNumber si country;


# ===========================
# 2. EXPLORAREA DATELOR
# ===========================
print("\n[2/7] Explorarea structurii datelor...")
print(f"\nColoane disponibile ({len(data.columns)}):")
print(data.columns.tolist())

print(f"\nStatistici descriptive pentru variabilele relevante:")
numeric_cols = ['visitNumber', 'pageviews', 'timeOnSite', 'hits'] # Selectam doar coloanele relevante pentru predictie
print(data[numeric_cols].describe()) # Afisam informatii pentru coloanele selectate cum ar fi media, abaterea standard, valorile minime si maxime, etc.

# ===========================
# 3. CURATAREA DATELOR
# ===========================
print("\n[3/7] Curatarea si pregatirea datelor...")

# Verificam valorile lipsa
print(f"\nValori lipsa per coloana:")
missing = data[numeric_cols].isnull().sum() # .isnull() returneaza un DataFrame cu True pentru valorile lipsa si False pentru cele prezente, iar .sum() numara cate valori lipsa sunt in fiecare coloana. 
if missing.sum() > 0:
    print(f"\nColoana si numarul de valori lipsa detectate:\n{missing[missing > 0]}") # missing[] returneaza doar coloanele care au valori lipsa, iar missing > 0 este o conditie care filtreaza doar acele coloane care au un numar de valori lipsa mai mare decat zero.
else:
    print("Nicio valoare lipsa!")

# Completam valorile lipsa pentru features cu mediana
for col in numeric_cols:
    if data[col].isnull().sum() > 0:
        median_val = data[col].median() # calculam mediana pentru coloana curenta
        data[col].fillna(median_val, inplace=True) # .fillna() inlocuieste valorile lipsa cu mediana calculata, iar inplace=True face ca modificarea sa se aplice direct pe DataFrame-ul original, fara a returna o copie.
        print(f"   - Completat '{col}' cu mediana: {median_val:.2f}")

# Analiza distributiei timeOnSite
print(f"\nDistributia timeOnSite:")
print(f"   - Sesiuni cu timeOnSite > 0: {(data['timeOnSite'] > 0).sum():,}")
print(f"   - Sesiuni cu timeOnSite = 0: {(data['timeOnSite'] == 0).sum():,}")
print(f"   - Timp mediu pe site: {data['timeOnSite'].mean():.2f} secunde ({data['timeOnSite'].mean()/60:.2f} minute)")
print(f"   - Mediana timp pe site: {data['timeOnSite'].median():.2f} secunde")
print(f"   - Max timp pe site: {data['timeOnSite'].max():.2f} secunde")

# Analiza corelatiei
print(f"\nCorelatia cu timeOnSite:")
correlation = data[numeric_cols].corr()['timeOnSite'].sort_values(ascending=False) # .corr() calculeaza matricea de corelatie pentru coloanele numerice selectate, iar ['timeOnSite'] extrage coloana de corelatie pentru variabila tinta timeOnSite. sort_values() sorteaza valorile in ordine descrescatoare pentru a vedea care features au cea mai puternica corelatie cu timeOnSite.
print(correlation)

# timeOnSite     1.000000
# pageviews      0.668764
# hits           0.647287
# visitNumber    0.049792

# ===========================
# 4. PREGATIREA FEATURES (VERSIUNEA IMBUNATATITA)
# ===========================
print("\n[4/7] Definirea features si target...")
print("\nFAZA 2: IMBUNATATIRI - Adaugam features categorice si feature engineering")

# ===========================
# 4A. FEATURES NUMERICE ORIGINALE
# ===========================
numeric_features = ['pageviews', 'visitNumber', 'hits'] # prin [pageviews, visitNumber, hits] se scrie o lista care contine numele coloanelor care vor fi folosite ca features numerice originale pentru modelul de regresie. 
print(f"\nFeatures numerice originale: {numeric_features}")

# ===========================
# 4B. FEATURES CATEGORICE (cu One-Hot Encoding)
# ===========================
print(f"\nAdaugam features categorice...")

# Verificam coloanele categorice disponibile
categorical_features = [] # Creeam o lista goala pentru a stoca numele coloanelor categorice care vor fi adaugate ca features in modelul imbunatatit.
if 'channelGrouping' in data.columns:
    categorical_features.append('channelGrouping') # Daca coloana 'channelGrouping' exista in DataFrame-ul data, atunci adaugam numele acestei coloane in lista categorical_features folosind metoda append(). Aceasta inseamna ca vom include aceasta coloana ca feature categoric in modelul imbunatatit.
    print(f"   - channelGrouping: {data['channelGrouping'].nunique()} categorii") # .nunique() returneaza numarul de categorii unice din coloana 'channelGrouping', iar print() afiseaza acest numar impreuna cu numele coloanei.
if 'device_category' in data.columns:
    categorical_features.append('device_category') # Daca coloana 'device_category' exista in DataFrame-ul data, atunci adaugam numele acestei coloane in lista categorical_features folosind metoda append(). Aceasta inseamna ca vom include aceasta coloana ca feature categoric in modelul imbunatatit.
    print(f"   - device_category: {data['device_category'].nunique()} categorii") # .nunique() returneaza numarul de categorii unice din coloana 'device_category', iar print() afiseaza acest numar impreuna cu numele coloanei.
if 'country' in data.columns:
    # Limitam la top 10 tari pentru a evita prea multe features
    top_countries = data['country'].value_counts().head(10).index.tolist() # .value_counts() numara frecventa fiecarei tari din coloana 'country', head(10) selecteaza primele 10 tari cu cele mai multe aparitii, iar index.tolist() extrage numele acestor tari intr-o lista Python.
    data['country_grouped'] = data['country'].apply(lambda x: x if x in top_countries else 'Other') # .apply() aplica o functie lambda pe fiecare element din coloana 'country'. Functia lambda verifica daca tara (x) se afla in lista top_countries; daca da, pastreaza numele tarii, altfel inlocuieste cu 'Other'. Aceasta operatie grupeaza toate tarile care nu sunt in top 10 sub o singura categorie 'Other', reducand astfel numarul de categorii unice si evitand problemele legate de prea multe features dupa one-hot encoding.
    categorical_features.append('country_grouped')
    print(f"   - country (top 10 + Other): {data['country_grouped'].nunique()} categorii") # .nunique() returneaza numarul de categorii unice din coloana 'country_grouped', iar print() afiseaza acest numar impreuna cu numele coloanei.

# One-Hot Encoding pentru features categorice, adica transformam fiecare categorie unica din coloanele categorice in cate o coloana binara (0 sau 1) care indica prezenta sau absenta acelei categorii in fiecare rand.
if len(categorical_features) > 0:
    categorical_encoded = pd.get_dummies(data[categorical_features], prefix=categorical_features, drop_first=True)
    # Unde:
    # pd.get_dummies() este o functie din biblioteca pandas care realizeaza one-hot encoding pentru coloanele specificate in categorical_features. Aceasta functie creeaza cate o coloana binara pentru fiecare categorie unica din aceste coloane.
    # prefix=categorical_features adauga un prefix coloanelor create, corespunzator numelui coloanei originale( ex: channelGrouping_Social, device_category_mobile, etc.), pentru a pastra claritatea si a evita confuziile intre coloane.
    # drop_first=True elimina prima coloana binara pentru fiecare feature, evitand astfel coliniaritatea.
    print(f"Caracteristici(features) categorice encodate: {categorical_encoded.shape[1]} coloane.\n")
    print(f"Coloanele categorice encodate: {categorical_encoded.columns.tolist()}")
else:
    categorical_encoded = pd.DataFrame()
    print(f"Nicio caracteristica(feature) categorica disponibila")

# ===========================
# 4C. FEATURE ENGINEERING (Features Noi)
# ===========================
print(f"\nCream features noi prin feature engineering...")

engineered_features = pd.DataFrame() # Un DataFrame gol care va fi folosit pentru a stoca noile features create prin feature engineering. Acest DataFrame va avea aceeasi dimensiune (numar de randuri) ca si DataFrame-ul original, dar va contine doar coloanele corespunzatoare noilor features create. Pe masura ce adaugam noi features, acestea vor fi stocate in acest DataFrame, iar la final vom putea combina aceste features noi cu cele originale si cele categorice encodate pentru a forma setul final de date pentru antrenarea modelelor de regresie.

# 1. Pages per visit
if 'visitNumber' in data.columns and 'pageviews' in data.columns:
    engineered_features['pageviews_per_visit'] = data['pageviews'] / (data['visitNumber'] + 1)  # +1 pentru a evita diviziunea la 0
    print(f" Primele 5 inregistrari in coloana 'pageviews_per_visit' din lista nou creata 'engineered_features':\n {engineered_features['pageviews_per_visit'].head().tolist()} (pageviews / visitNumber)") # tolist() converteste seria pandas intr-o lista Python, iar print() afiseaza primele 5 valori din coloana 'pageviews_per_visit' pentru a verifica corectitudinea calculului. Aceasta noua caracteristica reprezinta numarul mediu de pagini vizitate per sesiune, oferind o masura a nivelului de engagement al utilizatorilor.

# 2. Engagement score
if 'pageviews' in data.columns and 'hits' in data.columns:
    engineered_features['engagement_score'] = data['pageviews'] * data['hits']
    print(f" Primele 5 inregistrari in coloana 'engagement_score' din lista nou creata 'engineered_features':\n {engineered_features['engagement_score'].head().tolist()} (pageviews × hits)")

# 3. Hits per pageview
if 'hits' in data.columns and 'pageviews' in data.columns:
    engineered_features['hits_per_pageview'] = data['hits'] / (data['pageviews'] + 1)  # +1 pentru a evita diviziunea la 0
    print(f" Primele 5 inregistrari in coloana 'hits_per_pageview' din lista nou creata 'engineered_features':\n {engineered_features['hits_per_pageview'].head().tolist()} (hits / pageviews)")

# 4. High engagement flag (pageviews > median)
if 'pageviews' in data.columns:
    engineered_features['high_pageviews'] = (data['pageviews'] > data['pageviews'].median()).astype(int) # .astype(int) converteste valorile booleene (True/False) in valori binare (1/0), unde 1 indica ca numarul de pageviews este mai mare decat mediana, iar 0 indica ca este egal sau mai mic. Aceasta noua caracteristica binara poate ajuta modelele de regresie sa identifice mai usor sesiunile cu un nivel ridicat de engagement, care ar putea avea un impact semnificativ asupra timpului petrecut pe site.
    print(f" Primele 5 inregistrari in coloana 'high_pageviews' din lista nou creata 'engineered_features':\n {engineered_features['high_pageviews'].head().tolist()} (binary flag)")

# 5. Returning visitor flag
if 'visitNumber' in data.columns:
    engineered_features['is_returning'] = (data['visitNumber'] > 1).astype(int)
    print(f" Primele 5 inregistrari in coloana 'is_returning' din lista nou creata 'engineered_features':\n {engineered_features['is_returning'].head().tolist()} (binary flag)")

print(f"Caracteristici(features) create: {engineered_features.shape[1]} coloane noi")
engineered_features.columns.tolist() # Afisam numele coloanelor noilor features create pentru a verifica corectitudinea si pentru a avea o evidenta clara a acestor noi caracteristici care vor fi adaugate la setul de date final pentru antrenarea modelelor de regresie.

# ===========================
# 4D. COMBINAM TOATE FEATURES
# ===========================
print(f"\nCombinam toate caracteristicile(features)...")

# Features numerice originale
X_numeric = data[numeric_features].copy() # Creaza un nou DataFrame X_numeric care contine doar coloanele specificate in lista numeric_features din DataFrame-ul original data. Acest DataFrame va fi folosit pentru a stoca doar features-urile numerice originale (pageviews, visitNumber, hits) care vor fi utilizate in modelul de regresie.

# Combinam toate features
X_enhanced = pd.concat([
    X_numeric,
    categorical_encoded,
    engineered_features
], axis=1) 
# Unde:
# pd.concat() este o functie din biblioteca pandas care concateneaza (combina) mai multe DataFrame-uri de-a lungul unui anumit ax (axis=1 indica concatenarea pe coloane). In acest caz, combinam features-urile numerice originale (X_numeric), features-urile categorice encodate (categorical_encoded) si features-urile noi create prin feature engineering (engineered_features) intr-un singur DataFrame numit X_enhanced, care va contine toate caracteristicile ce vor fi folosite pentru antrenarea modelelor de regresie.
# axis=1 specifica faptul ca dorim sa concatenam pe coloane, adica sa adaugam noile features ca noi coloane in DataFrame-ul final X_enhanced. Daca am fi folosit axis=0, am fi concatenat pe randuri, ceea ce nu ar fi fost corect in acest context deoarece vrem sa pastram aceeasi structura de randuri (sesiuni) si sa adaugam doar noi coloane (features) pentru fiecare sesiune.

# Salvam X_enhanced intr-un fisier CSV pentru a putea verifica structura finala a setului de date care va fi folosit pentru antrenarea modelelor de regresie imbunatatite, si pentru a avea o evidenta clara a tuturor features-urilor incluse in acest set de date final.
X_enhanced.to_csv(X_ENHANCED_PATH, index=False)

# Variabila tinta
y = data['timeOnSite'].copy()

print(f"\nFeatures totale:")
print(f"   - Numerice originale: {len(numeric_features)}")
print(f"   - Categorice encodate: {categorical_encoded.shape[1]}")
print(f"   - Features engineered: {engineered_features.shape[1]}")
print(f"   - TOTAL: {X_enhanced.shape[1]} features")
print(f"\nNumele coloanelor din X_enhanced:\n{X_enhanced.columns.tolist()}")
print(f"\nTarget (y): timeOnSite")
print(f"\nShape final: X={X_enhanced.shape}, y={y.shape}")

# ===========================
# 4E. SPLIT TRAIN/TEST
# ===========================
# Pentru comparatie, cream 2 versiuni:
# 1. Versiunea SIMPLA (doar features numerice originale)
# 2. Versiunea IMBUNATATITA (toate features)

print(f"\nCream 2 versiuni pentru comparatie:")

# VERSIUNEA SIMPLA
X_simple = data[numeric_features].copy() # Creaza un nou DataFrame X_simple care contine doar coloanele specificate in lista numeric_features din DataFrame-ul original data. Acest DataFrame va fi folosit pentru a stoca doar features-urile numerice originale (pageviews, visitNumber, hits) care vor fi utilizate in modelul de regresie simplu pentru comparatie cu versiunea imbunatatita care contine toate features-urile (numerice, categorice encodate si engineered).

# Salvam X_simple intr-un fisier CSV pentru a putea verifica structura setului de date care va fi folosit pentru antrenarea modelului de regresie simplu, si pentru a avea o evidenta clara a features-urilor numerice originale care vor fi incluse in acest model simplu.
X_simple.to_csv(X_SIMPLE_PATH, index=False)

X_simple_train, X_simple_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)
# Unde:
# train_test_split() este o functie din biblioteca scikit-learn care imparte setul de date in doua subseturi: unul pentru antrenare (train) si unul pentru testare (test). 
# Aceasta functie ia ca argumente features-urile (X_simple) si variabila tinta (y), precum si dimensiunea setului de 
# test (test_size=0.2 indica ca 20% din date vor fi folosite pentru testare) si 
# random_state=42 asigura reproducibilitatea impartirii datelor.

# ELIMINARE RANDURI CU NaN pentru versiunea simpla
print(f"\n[Curatare date] Verificare NaN-uri pentru versiunea SIMPLA...")
nan_mask_train = X_simple_train.isnull().any(axis=1) | y_train.isnull() # unde .any(axis=1) verifica daca exista cel putin un NaN pe fiecare rand (axis=1) in X_simple_train, iar | y_train.isnull() adauga o conditie suplimentara pentru a verifica daca exista NaN-uri in variabila tinta y_train. Rezultatul este o masca booleana (nan_mask_train) care indica cu True randurile care contin NaN-uri in oricare dintre features-urile numerice originale sau in variabila tinta, si cu False randurile care nu contin NaN-uri.
nan_mask_test = X_simple_test.isnull().any(axis=1) | y_test.isnull()

if nan_mask_train.sum() > 0:
    print(f"Eliminare {nan_mask_train.sum()} randuri cu NaN din train set (simpla)")
    X_simple_train = X_simple_train[~nan_mask_train] # ~nan_mask_train inverseaza masca booleana, astfel incat sa selecteze doar randurile care nu contin NaN-uri in X_simple_train. Aceasta operatie elimina efectiv randurile cu NaN-uri din setul de antrenare pentru versiunea simpla a modelului de regresie. De asemenea, y_train este filtrat in acelasi mod pentru a pastra consistenta intre features si variabila tinta, eliminand aceleasi randuri care contin NaN-uri in y_train.
    y_train = y_train[~nan_mask_train]

if nan_mask_test.sum() > 0:
    print(f"Eliminare {nan_mask_test.sum()} randuri cu NaN din test set (simpla)")
    X_simple_test = X_simple_test[~nan_mask_test]
    y_test = y_test[~nan_mask_test]

print(f"Date curate - Train: {len(X_simple_train)}, Test: {len(X_simple_test)}")

scaler_simple = StandardScaler() # Creaza un obiect StandardScaler care va fi folosit pentru a standardiza (normaliza) features-urile numerice 
# din versiunea simpla a modelului de regresie. StandardScaler standardizeaza datele prin eliminarea mediei si 
# scalarea la unitate de varianta, ceea ce poate imbunatati performanta modelelor de regresie, in special pentru 
# cele care sunt sensibile la scara features-urilor (ex: Linear Regression, Polynomial Regression). 
# Acest scaler va fi aplicat doar pe features-urile numerice originale (pageviews, visitNumber, hits) din 
# X_simple pentru a asigura ca acestea au o distributie standardizata inainte de antrenarea modelelor.
X_simple_train_scaled = scaler_simple.fit_transform(X_simple_train) # .fit_transform() se aplica pe setul de antrenare (X_simple_train) pentru a calcula media si abaterea standard a features-urilor numerice originale, iar apoi transforma aceste features prin standardizare. Rezultatul este un nou array numpy (X_simple_train_scaled) care contine valorile standardizate ale features-urilor numerice originale pentru setul de antrenare.
X_simple_test_scaled = scaler_simple.transform(X_simple_test) # .transform() se aplica pe setul de testare (X_simple_test) folosind media si abaterea standard calculate anterior pe setul de antrenare pentru a transforma features-urile numerice originale din setul de testare in aceeasi scara standardizata. Rezultatul este un nou array numpy (X_simple_test_scaled) care contine valorile standardizate ale features-urilor numerice originale pentru setul de testare, asigurand astfel ca atat setul de antrenare, cat si cel de testare au aceeasi distributie standardizata a features-urilor numerice originale.

# Verificare finala dupa standardizare
if np.isnan(X_simple_train_scaled).any():
    print(f"ATENTIE: NaN-uri detectate DUPA standardizare in X_simple_train_scaled!")
    print(f"   NaN count: {np.isnan(X_simple_train_scaled).sum()}")
    # Eliminam randurile cu NaN dupa standardizare
    nan_mask_scaled = np.isnan(X_simple_train_scaled).any(axis=1)
    X_simple_train_scaled = X_simple_train_scaled[~nan_mask_scaled]
    y_train = y_train.iloc[~nan_mask_scaled] if hasattr(y_train, 'iloc') else y_train[~nan_mask_scaled]
    print(f"   Eliminate {nan_mask_scaled.sum()} randuri. Raman: {len(X_simple_train_scaled)}")

print(f"   1. Versiunea SIMPLA: {X_simple.shape[1]} features")

# VERSIUNEA IMBUNATATITA
X_enhanced_train, X_enhanced_test, y_train_enh, y_test_enh = train_test_split(X_enhanced, y, test_size=0.2, random_state=42)

# ELIMINARE RANDURI CU NaN pentru versiunea imbunatatita
print(f"\n[Curatare date] Verificare NaN-uri pentru versiunea IMBUNATATITA...")
nan_mask_train_enh = X_enhanced_train.isnull().any(axis=1) | y_train_enh.isnull()
nan_mask_test_enh = X_enhanced_test.isnull().any(axis=1) | y_test_enh.isnull()

if nan_mask_train_enh.sum() > 0:
    print(f"Eliminare {nan_mask_train_enh.sum()} randuri cu NaN din train set (imbunatatita)")
    X_enhanced_train = X_enhanced_train[~nan_mask_train_enh]
    y_train_enh = y_train_enh[~nan_mask_train_enh]

if nan_mask_test_enh.sum() > 0:
    print(f"Eliminare {nan_mask_test_enh.sum()} randuri cu NaN din test set (imbunatatita)")
    X_enhanced_test = X_enhanced_test[~nan_mask_test_enh]
    y_test_enh = y_test_enh[~nan_mask_test_enh]

print(f"Date curate - Train: {len(X_enhanced_train)}, Test: {len(X_enhanced_test)}")

scaler_enhanced = StandardScaler() # Creaza un obiect StandardScaler care va fi folosit pentru a standardiza (normaliza) toate features-urile din versiunea imbunatatita a modelului de regresie, care include atat features-urile numerice originale, cat si cele categorice encodate si cele create prin feature engineering. StandardScaler va standardiza toate aceste features prin eliminarea mediei si scalarea la unitate de varianta, ceea ce poate imbunatati performanta modelelor de regresie, in special pentru cele care sunt sensibile la scara features-urilor (ex: Linear Regression, Polynomial Regression). Acest scaler va fi aplicat pe intregul set de features din X_enhanced pentru a asigura ca toate caracteristicile au o distributie standardizata inainte de antrenarea modelelor.
X_enhanced_train_scaled = scaler_enhanced.fit_transform(X_enhanced_train) # .fit_transform() se aplica pe setul de antrenare (X_enhanced_train) pentru a calcula media si abaterea standard a tuturor features-urilor din versiunea imbunatatita, iar apoi transforma aceste features prin standardizare. Rezultatul este un nou array numpy (X_enhanced_train_scaled) care contine valorile standardizate ale tuturor features-urilor pentru setul de antrenare.
X_enhanced_test_scaled = scaler_enhanced.transform(X_enhanced_test) # .transform() se aplica pe setul de testare (X_enhanced_test) folosind media si abaterea standard calculate anterior pe setul de antrenare pentru a transforma toate features-urile din setul de testare in aceeasi scara standardizata. Rezultatul este un nou array numpy (X_enhanced_test_scaled) care contine valorile standardizate ale tuturor features-urilor pentru setul de testare, asigurand astfel ca atat setul de antrenare, cat si cel de testare au aceeasi distributie standardizata a features-urilor.

# Verificare finala dupa standardizare
if np.isnan(X_enhanced_train_scaled).any():
    print(f"ATENTIE: NaN-uri detectate DUPA standardizare in X_enhanced_train_scaled!")
    print(f"   NaN count: {np.isnan(X_enhanced_train_scaled).sum()}")
    # Eliminam randurile cu NaN dupa standardizare
    nan_mask_scaled_enh = np.isnan(X_enhanced_train_scaled).any(axis=1)
    X_enhanced_train_scaled = X_enhanced_train_scaled[~nan_mask_scaled_enh]
    y_train_enh = y_train_enh.iloc[~nan_mask_scaled_enh] if hasattr(y_train_enh, 'iloc') else y_train_enh[~nan_mask_scaled_enh]
    print(f"   Eliminate {nan_mask_scaled_enh.sum()} randuri. Raman: {len(X_enhanced_train_scaled)}")

print(f"   2. Versiunea IMBUNATATITA: {X_enhanced.shape[1]} features") # Afiseaza numarul total de features din versiunea imbunatatita a modelului de regresie, care include atat features-urile numerice originale, cat si cele categorice encodate si cele create prin feature engineering. Acest numar reprezinta dimensiunea setului de date final care va fi folosit pentru antrenarea modelelor de regresie imbunatatite, oferind o baza mai bogata de informatii pentru predictia timpului petrecut pe site (timeOnSite).

print(f"Train set: {X_enhanced_train.shape[0]:,} samples")
print(f"Test set: {X_enhanced_test.shape[0]:,} samples")

# ===========================
# 5. ANTRENAREA MODELELOR (COMPARATIE: SIMPLA vs IMBUNATATITA)
# ===========================
print("\n[5/7] Antrenarea modelelor de regresie...")
print("="*80)
print("\nVom antrena fiecare model in 2 VERSIUNI pentru comparatie:")
print("   - Versiune SIMPLA: doar 3 features numerice")
print("   - Versiune IMBUNATATITA: features categorice + engineered\n")

# Dictionare pentru stocarea rezultatelor
results_simple = {}
results_enhanced = {}

# ===========================
# MODEL 1: Linear Regression Multiple # este un model de regresie liniara care utilizeaza mai multe features pentru a prezice variabila tinta (timeOnSite). Acest model va fi antrenat in doua versiuni: una simpla, care va folosi doar features-urile numerice originale (pageviews, visitNumber, hits), si una imbunatatita, care va include atat aceste features numerice originale, cat si features-urile categorice encodate si cele create prin feature engineering. Scopul este de a compara performanta acestor doua versiuni ale modelului de regresie liniara pentru a evalua impactul adaugarii de noi features asupra acuratetii predictiilor timpului petrecut pe site.
# ===========================
print("\n" + "="*80)
print("MODEL 1: Linear Regression Multiple")
print("="*80)

# VERSIUNEA SIMPLA
print("\n   - Versiune SIMPLA (3 features):")
start_time = time.time() # time.time() returneaza timpul curent in secunde de la epoca (1 ianuarie 1970), 
# iar atribuirea acestui timp la variabila start_time ne permite sa masuram durata antrenarii modelului de 
# regresie simplu prin calcularea diferentei dintre timpul curent dupa antrenare si timpul initial stocat in start_time. Aceasta metoda este utila pentru a evalua eficienta si performanta modelelor de regresie in termeni de timp de antrenare, oferind o perspectiva asupra resurselor necesare pentru a obtine predictii precise.
lr_simple_model = LinearRegression() # Creaza un obiect LinearRegression care va fi folosit pentru a antrena modelul de regresie liniara simplu folosind features-urile numerice originale (pageviews, visitNumber, hits) din versiunea simpla a setului de date. Acest model va invata relatia liniara dintre aceste features si variabila tinta (timeOnSite) pentru a putea face predictii asupra timpului petrecut pe site in functie de valorile acestor features.
# Antrenam modelul de regresie liniara simplu folosind metoda fit() pe setul de antrenare standardizat (X_simple_train_scaled) si variabila tinta (y_train). Aceasta operatie va ajusta coeficientii modelului pentru a minimiza eroarea dintre predictiile modelului si valorile reale ale variabilei tinta in setul de antrenare, permitand astfel modelului sa invete relatia dintre features-urile numerice originale si timpul petrecut pe site (timeOnSite).
lr_simple_model.fit(X_simple_train_scaled, y_train)

# Facem predictii folosind modelul antrenat pe setul de testare standardizat (X_simple_test_scaled) pentru a evalua performanta modelului de regresie liniara simplu in predictia timpului petrecut pe site. Aceste predictii vor fi comparate cu valorile reale ale variabilei tinta (y_test) pentru a calcula metrici de evaluare precum R², RMSE si MAE, oferind o perspectiva asupra acuratetii si eficientei modelului in versiunea sa simpla.
y_lr_simple_pred = lr_simple_model.predict(X_simple_test_scaled)

# Calculam timpul de antrenare al modelului de regresie liniara simplu prin scaderea timpului initial stocat in 
# start_time din timpul curent dupa ce predictiile au fost generate. Acest timp de antrenare ofera o masura a 
# resurselor necesare pentru a obtine predictii folosind acest model, fiind util pentru compararea eficientei intre 
# versiunea simpla si cea imbunatatita a modelului de regresie.
lr_simple_time = time.time() - start_time 

# Calculam metrici de evaluare pentru modelul de regresie liniara simplu, comparand predictiile generate (y_lr_simple_pred) cu valorile reale ale variabilei tinta (y_test). R² Score masoara proportia variatiei in variabila tinta care poate fi explicata de features-urile din model, RMSE (Root Mean Squared Error) ofera o masura a erorii medii a predictiilor in aceeasi unitate ca si variabila tinta (secunde), iar MAE (Mean Absolute Error) ofera o masura a erorii absolute medii a predictiilor. Acesti metrici ne vor ajuta sa evaluam acuratetea si performanta modelului de regresie liniara simplu in predictia timpului petrecut pe site.
# Se va folosi y_test pentru a avea o comparatie corecta intre predictiile modelului si valorile reale ale variabilei tinta in setul de testare, oferind astfel o evaluare precisa a performantei modelului in versiunea sa simpla.
lr_simple_r2 = r2_score(y_test, y_lr_simple_pred) # Coeficientul de determinare R² masoara proportia variatiei in variabila tinta (timeOnSite) care poate fi explicata de features-urile numerice originale (pageviews, visitNumber, hits) din versiunea simpla a modelului de regresie liniara. Un R² mai mare indica o potrivire mai buna a modelului la datele de testare, sugerand ca modelul poate explica o parte semnificativa a variatiei in timpul petrecut pe site pe baza acestor features.
lr_simple_rmse = np.sqrt(mean_squared_error(y_test, y_lr_simple_pred))
lr_simple_mae = mean_absolute_error(y_test, y_lr_simple_pred)

results_simple['Linear Regression'] = { # Un dictionar care stocheaza rezultatele pentru modelul de regresie liniara simplu, incluzand modelul antrenat (lr_simple_model), predictiile generate (y_lr_simple_pred), metricii de evaluare (R², RMSE, MAE) si timpul de antrenare (lr_simple_time). Acest dictionar va fi folosit pentru a compara performanta modelului simplu cu cea a modelului imbunatatit, oferind o perspectiva clara asupra impactului adaugarii features-urilor categorice si celor create prin feature engineering asupra acuratetii predictiilor.
    'model': lr_simple_model,
    'predictions': y_lr_simple_pred,
    'r2': lr_simple_r2,
    'rmse': lr_simple_rmse,
    'mae': lr_simple_mae,
    'time': lr_simple_time
}

print(f"      R² Score: {lr_simple_r2:.4f}")
print(f"      RMSE: {lr_simple_rmse:,.2f} seconds")
print(f"      MAE: {lr_simple_mae:,.2f} seconds")
print(f"      Training time: {lr_simple_time:.4f}s")

# VERSIUNEA IMBUNATATITA
print(f"\n  Versiune IMBUNATATITA ({X_enhanced.shape[1]} features):")
start_time = time.time()
lr_enhanced_model = LinearRegression()
lr_enhanced_model.fit(X_enhanced_train_scaled, y_train_enh)
y_lr_enhanced_pred = lr_enhanced_model.predict(X_enhanced_test_scaled)
lr_enhanced_time = time.time() - start_time

lr_enhanced_r2 = r2_score(y_test_enh, y_lr_enhanced_pred)
lr_enhanced_rmse = np.sqrt(mean_squared_error(y_test_enh, y_lr_enhanced_pred)) # np.sqrt() calculeaza radacina patrata a erorii medii patratice (MSE) dintre valorile reale (y_test_enh) si predictiile modelului imbunatatit (y_lr_enhanced_pred). RMSE ofera o masura a erorii medii a predictiilor in aceeasi unitate ca si variabila tinta (secunde), fiind util pentru a evalua acuratetea modelului de regresie liniara imbunatatit in predictia timpului petrecut pe site.
lr_enhanced_mae = mean_absolute_error(y_test_enh, y_lr_enhanced_pred)

results_enhanced['Linear Regression'] = { # Un dictionar care stocheaza rezultatele pentru modelul de regresie liniara imbunatatit, incluzand modelul antrenat (lr_enhanced_model), predictiile generate (y_lr_enhanced_pred), metricii de evaluare (R², RMSE, MAE) si timpul de antrenare (lr_enhanced_time). Acest dictionar va fi folosit pentru a compara performanta modelului imbunatatit cu cea a modelului simplu, oferind o perspectiva clara asupra imbunatatirilor aduse prin adaugarea features-urilor categorice si celor create prin feature engineering.
    'model': lr_enhanced_model,
    'predictions': y_lr_enhanced_pred,
    'r2': lr_enhanced_r2,
    'rmse': lr_enhanced_rmse,
    'mae': lr_enhanced_mae,
    'time': lr_enhanced_time
}

print(f"      R² Score: {lr_enhanced_r2:.4f}")
print(f"      RMSE: {lr_enhanced_rmse:,.2f} seconds")
print(f"      MAE: {lr_enhanced_mae:,.2f} seconds")
print(f"      Training time: {lr_enhanced_time:.4f}s")

# Comparatie
improvement_r2 = lr_enhanced_r2 - lr_simple_r2
improvement_pct = (improvement_r2 / lr_simple_r2) * 100
print(f"\n IMBUNATATIRE: R² {improvement_r2:+.4f} ({improvement_pct:+.1f}%)")

# ===========================
# MODEL 2: Polynomial Regression (grad 2) # este un model de regresie care include termeni polinomiali de grad 2 pentru a captura relatii non-liniare intre features si variabila tinta (timeOnSite). Acest model va fi antrenat in doua versiuni: una simpla care foloseste doar features-urile numerice originale, si una imbunatatita care foloseste toate features-urile (numerice, categorice encodate si engineered) pentru a evalua impactul acestor caracteristici suplimentare asupra performantei modelului de regresie polinomial.
# ===========================
print("\n" + "="*80)
print("MODEL 2: Polynomial Regression (grad 2)")
print("="*80)

# VERSIUNEA SIMPLA
print("\n  Versiune SIMPLA (3 features):")
start_time = time.time()
poly_features_simple = PolynomialFeatures(degree=2, include_bias=False)
# Unde:
# PolynomialFeatures este o clasa din biblioteca scikit-learn care genereaza noi features polinomiale pe baza features-urilor originale.
# degree=2 specifica faptul ca dorim sa generam features polinomiale de grad 2, ceea ce inseamna ca vom include atat features-urile originale (pageviews, visitNumber, hits), cat si combinatii polinomiale de grad 2 (ex: pageviews^2, visitNumber^2, hits^2, pageviews*visitNumber, pageviews*hits, visitNumber*hits) pentru a captura relatii non-liniare intre aceste features si variabila tinta (timeOnSite).
# include_bias=False indica faptul ca nu dorim sa includem o coloana de bias (intercept) in setul de features generate, deoarece modelul de regresie liniara va include deja un termen de intercept implicit.   

X_simple_train_poly = poly_features_simple.fit_transform(X_simple_train) # .fit_transform() se aplica pe setul de antrenare (X_simple_train) pentru a calcula toate combinatiile polinomiale de grad 2 ale features-urilor numerice originale (pageviews, visitNumber, hits) si pentru a transforma aceste features intr-un nou set de features polinomiale. Rezultatul este un nou array numpy (X_simple_train_poly) care contine atat features-urile originale, cat si cele polinomiale generate, oferind astfel o baza mai bogata de informatii pentru modelul de regresie polinomial in versiunea sa simpla.
X_simple_test_poly = poly_features_simple.transform(X_simple_test) # .transform() se aplica pe setul de testare (X_simple_test) folosind aceleasi combinatii polinomiale calculate anterior pe setul de antrenare pentru a transforma features-urile numerice originale din setul de testare in aceleasi features polinomiale. Rezultatul este un nou array numpy (X_simple_test_poly) care contine atat features-urile originale, cat si cele polinomiale generate pentru setul de testare, asigurand astfel ca atat setul de antrenare, cat si cel de testare au aceleasi features polinomiale pentru a putea evalua corect performanta modelului de regresie polinomial in versiunea sa simpla.

# Cream modelul de regresie liniara care va fi folosit pentru a antrena modelul de regresie polinomial in versiunea sa simpla, folosind features-urile polinomiale generate anterior. Acest model va invata relatia dintre aceste features polinomiale si variabila tinta (timeOnSite) pentru a putea face predictii asupra timpului petrecut pe site in functie de valorile acestor features polinomiale.
poly_simple_model = LinearRegression()

# Antrename modelul de regresie polinomial in versiunea sa simpla folosind metoda fit() pe setul de antrenare cu features polinomiale (X_simple_train_poly) si variabila tinta (y_train). Aceasta operatie va ajusta coeficientii modelului pentru a minimiza eroarea dintre predictiile modelului si valorile reale ale variabilei tinta in setul de antrenare, permitand astfel modelului sa invete relatia dintre features-urile polinomiale si timpul petrecut pe site (timeOnSite) in versiunea sa simpla. 
poly_simple_model.fit(X_simple_train_poly, y_train)

# Testam modelul de regresie polinomial in versiunea sa simpla pe setul de testare cu features polinomiale (X_simple_test_poly) pentru a evalua performanta acestui model in predictia timpului petrecut pe site. Aceste predictii vor fi comparate cu valorile reale ale variabilei tinta (y_test) pentru a calcula metrici de evaluare precum R², RMSE si MAE, oferind o perspectiva asupra acuratetii si eficientei modelului de regresie polinomial in versiunea sa simpla. Acest model va ajuta la capturarea relatiilor non-liniare dintre features-urile numerice originale si timpul petrecut pe site, oferind astfel o baza pentru comparatie cu versiunea imbunatatita care include toate features-urile (numerice, categorice encodate si engineered).
y_poly_simple_pred = poly_simple_model.predict(X_simple_test_poly)

# Calculam timpul de antrenare al modelului de regresie polinomial in versiunea sa simpla prin scaderea timpului initial stocat in start_time din timpul curent dupa ce predictiile au fost generate. Acest timp de antrenare ofera o masura a resurselor necesare pentru a obtine predictii folosind acest model, fiind util pentru compararea eficientei intre versiunea simpla si cea imbunatatita a modelului de regresie polinomial.
poly_simple_time = time.time() - start_time

# Calculam coeficientul de determinare R², RMSE si MAE pentru modelul de regresie polinomial in versiunea sa simpla, comparand predictiile generate (y_poly_simple_pred) cu valorile reale ale variabilei tinta (y_test). Acesti metrici ne vor ajuta sa evaluam acuratetea si performanta modelului de regresie polinomial in versiunea sa simpla in predictia timpului petrecut pe site, oferind o perspectiva asupra imbunatatirilor aduse prin adaugarea features-urilor categorice si celor create prin feature engineering in versiunea imbunatatita a modelului de regresie polinomial.
poly_simple_r2 = r2_score(y_test, y_poly_simple_pred)
poly_simple_rmse = np.sqrt(mean_squared_error(y_test, y_poly_simple_pred))
poly_simple_mae = mean_absolute_error(y_test, y_poly_simple_pred)

# Cream un dictionar pentru a stoca rezultatele modelului de regresie polinomial in versiunea sa simpla, incluzand modelul antrenat (poly_simple_model), predictiile generate (y_poly_simple_pred), metricii de evaluare (R², RMSE, MAE) si timpul de antrenare (poly_simple_time). Acest dictionar va fi folosit pentru a compara performanta modelului simplu cu cea a modelului imbunatatit, oferind o perspectiva clara asupra impactului adaugarii features-urilor categorice si celor create prin feature engineering asupra acuratetii predictiilor in cazul modelului de regresie polinomial.
results_simple['Polynomial Regression'] = {
    'model': poly_simple_model,
    'predictions': y_poly_simple_pred,
    'r2': poly_simple_r2,
    'rmse': poly_simple_rmse,
    'mae': poly_simple_mae,
    'time': poly_simple_time
}

print(f"      R² Score: {poly_simple_r2:.4f}")
print(f"      RMSE: {poly_simple_rmse:,.2f} seconds")
print(f"      MAE: {poly_simple_mae:,.2f} seconds")
print(f"      Training time: {poly_simple_time:.4f}s")
print(f"      Features polinomiale: {X_simple_train_poly.shape[1]}")

# VERSIUNEA IMBUNATATITA
print(f"\n  Versiune IMBUNATATITA ({X_enhanced.shape[1]} features):")
start_time = time.time()
poly_features_enhanced = PolynomialFeatures(degree=2, include_bias=False)
X_enhanced_train_poly = poly_features_enhanced.fit_transform(X_enhanced_train)
X_enhanced_test_poly = poly_features_enhanced.transform(X_enhanced_test)

# Cream modelul de regresie liniara care va fi folosit pentru a antrena modelul de regresie polinomial in versiunea sa imbunatatita, folosind features-urile polinomiale generate anterior. Acest model va invata relatia dintre aceste features polinomiale si variabila tinta (timeOnSite) pentru a putea face predictii asupra timpului petrecut pe site in functie de valorile acestor features polinomiale in versiunea imbunatatita care include toate features-urile (numerice, categorice encodate si engineered).
poly_enhanced_model = LinearRegression()

# Antrenam modelul de regresie polinomial in versiunea sa imbunatatita folosind metoda fit() pe setul de antrenare cu features polinomiale (X_enhanced_train_poly) si variabila tinta (y_train_enh). Aceasta operatie va ajusta coeficientii modelului pentru a minimiza eroarea dintre predictiile modelului si valorile reale ale variabilei tinta in setul de antrenare, permitand astfel modelului sa invete relatia dintre features-urile polinomiale generate din toate caracteristicile (numerice, categorice encodate si engineered) si timpul petrecut pe site (timeOnSite) in versiunea sa imbunatatita. Acest model va ajuta la capturarea relatiilor non-liniare complexe dintre toate features-urile disponibile si timpul petrecut pe site, oferind astfel o baza pentru comparatie cu versiunea simpla care include doar features-urile numerice originale.
poly_enhanced_model.fit(X_enhanced_train_poly, y_train_enh)

# Testam modelul de regresie polinomial in versiunea sa imbunatatita pe setul de testare cu features polinomiale (X_enhanced_test_poly) pentru a evalua performanta acestui model in predictia timpului petrecut pe site. Aceste predictii vor fi comparate cu valorile reale ale variabilei tinta (y_test_enh) pentru a calcula metrici de evaluare precum R², RMSE si MAE, oferind o perspectiva asupra acuratetei si eficientei modelului de regresie polinomial in versiunea sa imbunatatita. Acest model va ajuta la capturarea relatiilor non-liniare complexe dintre toate features-urile disponibile (numerice, categorice encodate si engineered) si timpul petrecut pe site, oferind astfel o baza pentru comparatie cu versiunea simpla care include doar features-urile numerice originale.
# Adica generam predictii folosind modelul de regresie polinomial imbunatatit pe setul de testare cu features polinomiale generate din toate caracteristicile (X_enhanced_test_poly) si le comparam cu valorile reale ale variabilei tinta (y_test_enh) pentru a evalua performanta acestui model in versiunea sa imbunatatita.
y_poly_enhanced_pred = poly_enhanced_model.predict(X_enhanced_test_poly)

# Calculam timpul de antrenare al modelului de regresie polinomial in versiunea sa imbunatatita prin scaderea timpului initial stocat in start_time din timpul curent dupa ce predictiile au fost generate. Acest timp de antrenare ofera o masura a resurselor necesare pentru a obtine predictii folosind acest model, fiind util pentru compararea eficientei intre versiunea simpla si cea imbunatatita a modelului de regresie polinomial.
poly_enhanced_time = time.time() - start_time

# Calculam coeficientul de determinare R², RMSE si MAE pentru modelul de regresie polinomial in versiunea sa imbunatatita, comparand predictiile generate (y_poly_enhanced_pred) cu valorile reale ale variabilei tinta (y_test_enh). Acesti metrici ne vor ajuta sa evaluam acuratetea si performanta modelului de regresie polinomial in versiunea sa imbunatatita in predictia timpului petrecut pe site, oferind o perspectiva asupra imbunatatirilor aduse prin adaugarea features-urilor categorice si celor create prin feature engineering in aceasta versiune a modelului de regresie polinomial.
poly_enhanced_r2 = r2_score(y_test_enh, y_poly_enhanced_pred)
poly_enhanced_rmse = np.sqrt(mean_squared_error(y_test_enh, y_poly_enhanced_pred))
poly_enhanced_mae = mean_absolute_error(y_test_enh, y_poly_enhanced_pred)

# Cream un dictionar pentru a stoca rezultatele modelului de regresie polinomial in versiunea sa imbunatatita, incluzand modelul antrenat (poly_enhanced_model), predictiile generate (y_poly_enhanced_pred), metricii de evaluare (R², RMSE, MAE) si timpul de antrenare (poly_enhanced_time). Acest dictionar va fi folosit pentru a compara performanta modelului imbunatatit cu cea a modelului simplu, oferind o perspectiva clara asupra imbunatatirilor aduse prin adaugarea features-urilor categorice si celor create prin feature engineering in versiunea imbunatatita a modelului de regresie polinomial.
results_enhanced['Polynomial Regression'] = {
    'model': poly_enhanced_model,
    'predictions': y_poly_enhanced_pred,
    'r2': poly_enhanced_r2,
    'rmse': poly_enhanced_rmse,
    'mae': poly_enhanced_mae,
    'time': poly_enhanced_time
}

print(f"      R² Score: {poly_enhanced_r2:.4f}")
print(f"      RMSE: {poly_enhanced_rmse:,.2f} seconds")
print(f"      MAE: {poly_enhanced_mae:,.2f} seconds")
print(f"      Training time: {poly_enhanced_time:.4f}s")
print(f"      Features polinomiale: {X_enhanced_train_poly.shape[1]}")

# Comparatie
improvement_r2 = poly_enhanced_r2 - poly_simple_r2
improvement_pct = (improvement_r2 / poly_simple_r2) * 100
print(f"\n IMBUNATATIRE in cazul modelului de regresie polinomial: R² {improvement_r2:+.4f} ({improvement_pct:+.1f}%)") # +.4f formateaza imbunatatirea R² cu 4 zecimale, iar + indica faptul ca va afisa semnul plus sau minus in functie de valoarea imbunatatirii. Acest mesaj va arata cat de mult a crescut R² datorita adaugarii features-urilor categorice si celor create prin feature engineering in versiunea imbunatatita a modelului de regresie polinomial, oferind o perspectiva clara asupra impactului acestor adaugiri asupra performantei modelului in predictia timpului petrecut pe site (timeOnSite).

# ===========================
# MODEL 3: Random Forest Regressor # este un model de regresie bazat pe ansamblul de arbori de decizie, care utilizeaza tehnica de bagging pentru a imbunatati acuratetea si a reduce suprainvatarea (overfitting). Acest model va invata relatia dintre features-urile numerice originale, cele categorice encodate si cele create prin feature engineering si variabila tinta (timeOnSite) prin construirea unui ansamblu de arbori de decizie, oferind astfel o abordare non-liniara pentru predictia timpului petrecut pe site. Random Forest Regressor este cunoscut pentru capacitatea sa de a gestiona datele cu multe features si pentru a oferi predictii robuste, fiind adesea folosit in problemele de regresie cu date complexe.
# ===========================
print("\n" + "="*80)
print("MODEL 3: Random Forest Regressor")
print("="*80)

# VERSIUNEA SIMPLA
print("\n  Versiune SIMPLA (3 features):")
start_time = time.time()

# Cream modelul de regresie Random Forest pentru versiunea simpla, folosind doar features-urile numerice originale (pageviews, visitNumber, hits) din setul de date. Acest model va invata relatia dintre aceste features si variabila tinta (timeOnSite) prin construirea unui ansamblu de arbori de decizie, oferind astfel o abordare non-liniara pentru predictia timpului petrecut pe site in versiunea sa simpla.
rf_simple_model = RandomForestRegressor(
    # n_estimators=100, # numarul de arbori de decizie care vor fi construiți în cadrul modelului Random Forest. Un numar mai mare de arbori poate imbunatati acuratetea modelului, dar poate creste si timpul de antrenare. In acest caz, am ales 100 de arbori pentru a oferi un echilibru intre performanta si eficienta.
    n_estimators=1500, # numarul de arbori de decizie care vor fi construiți în cadrul modelului Random Forest. Un numar mai mare de arbori poate imbunatati acuratetea modelului, dar poate creste si timpul de antrenare. In acest caz, am ales 1000 de arbori pentru a oferi un echilibru intre performanta si eficienta.
    
    # max_depth=15, # adancimea maxima a fiecarui arbore de decizie din cadrul modelului Random Forest. Limitarea adancimii arborilor poate preveni suprainvatarea (overfitting) si poate imbunatati generalizarea modelului, dar poate reduce si capacitatea modelului de a captura relatii complexe in date. In acest caz, am ales o adancime maxima de 15 pentru a oferi un echilibru intre complexitate si generalizare.
    max_depth=50, # adancimea maxima a fiecarui arbore de decizie din cadrul modelului Random Forest. Limitarea adancimii arborilor poate preveni suprainvatarea (overfitting) si poate imbunatati generalizarea modelului, dar poate reduce si capacitatea modelului de a captura relatii complexe in date. In acest caz, am ales o adancime maxima de 100 pentru a oferi un echilibru intre complexitate si generalizare.
    
    # min_samples_split=10, # numarul minim de esantioane necesare pentru a imparti un nod intern. Cresterea acestei valori poate preveni suprainvatarea, dar poate reduce si capacitatea modelului de a captura relatii complexe.
    min_samples_split=20, # numarul minim de esantioane necesare pentru a imparti un nod intern. Cresterea acestei valori poate preveni suprainvatarea, dar poate reduce si capacitatea modelului de a captura relatii complexe.
    
    # min_samples_leaf=5, # numarul minim de esantioane necesare pentru a fi intr-un nod frunza. Cresterea acestei valori poate preveni suprainvatarea, dar poate reduce si capacitatea modelului de a captura relatii complexe.
    min_samples_leaf=30, # numarul minim de esantioane necesare pentru a fi intr-un nod frunza. Cresterea acestei valori poate preveni suprainvatarea, dar poate reduce si capacitatea modelului de a captura relatii complexe.
    
    random_state=42, # asigura reproducibilitatea rezultatelor prin fixarea seed-ului generatorului de numere aleatoare.
    n_jobs=-1 # utilizeaza toate nucleele disponibile ale procesorului pentru a accelera antrenarea modelului.
)

# Antrenam modelul de regresie Random Forest folosind metoda fit() pe setul de antrenare (X_simple_train) si variabila tinta (y_train). Aceasta operatie va construi un ansamblu de arbori de decizie bazati pe features-urile numerice originale pentru a invata relatia dintre aceste features si timpul petrecut pe site (timeOnSite) in versiunea sa simpla.
rf_simple_model.fit(X_simple_train, y_train)

# Facem predictii folosind modelul de regresie Random Forest antrenat pe setul de testare (X_simple_test) pentru a evalua performanta acestui model in predictia timpului petrecut pe site. Aceste predictii vor fi comparate cu valorile reale ale variabilei tinta (y_test) pentru a calcula metrici de evaluare precum R², RMSE si MAE, oferind o perspectiva asupra acuratetei si eficientei modelului de regresie Random Forest in versiunea sa simpla.
y_rf_simple_pred = rf_simple_model.predict(X_simple_test)

# Calculam timpul de antrenare al modelului de regresie Random Forest in versiunea sa simpla prin scaderea timpului initial stocat in start_time din timpul curent dupa ce predictiile au fost generate. Acest timp de antrenare ofera o masura a resurselor necesare pentru a obtine predictii folosind acest model, fiind util pentru compararea eficientei intre versiunea simpla si cea imbunatatita a modelului de regresie Random Forest.
rf_simple_time = time.time() - start_time

# Calculam coeficientul de determinare R², RMSE si MAE pentru modelul de regresie Random Forest in versiunea sa simpla, comparand predictiile generate (y_rf_simple_pred) cu valorile reale ale variabilei tinta (y_test). Acesti metrici ne vor ajuta sa evaluam acuratetea si performanta modelului de regresie Random Forest in versiunea sa simpla in predictia timpului petrecut pe site, oferind o perspectiva asupra imbunatatirilor aduse prin adaugarea features-urilor categorice si celor create prin feature engineering in versiunea imbunatatita a modelului de regresie Random Forest.
rf_simple_r2 = r2_score(y_test, y_rf_simple_pred)
rf_simple_rmse = np.sqrt(mean_squared_error(y_test, y_rf_simple_pred))
rf_simple_mae = mean_absolute_error(y_test, y_rf_simple_pred)

# Cream un dictionar pentru a stoca rezultatele modelului de regresie Random Forest in versiunea sa simpla, incluzand modelul antrenat (rf_simple_model), predictiile generate (y_rf_simple_pred), metricii de evaluare (R², RMSE, MAE) si timpul de antrenare (rf_simple_time). Acest dictionar va fi folosit pentru a compara performanta modelului simplu cu cea a modelului imbunatatit, oferind o perspectiva clara asupra impactului adaugarii features-urilor categorice si celor create prin feature engineering asupra acuratetii predictiilor in cazul modelului de regresie Random Forest.
results_simple['Random Forest'] = {
    'model': rf_simple_model,
    'predictions': y_rf_simple_pred,
    'r2': rf_simple_r2,
    'rmse': rf_simple_rmse,
    'mae': rf_simple_mae,
    'time': rf_simple_time
}

print(f"      R² Score: {rf_simple_r2:.4f}")
print(f"      RMSE: {rf_simple_rmse:,.2f} seconds")
print(f"      MAE: {rf_simple_mae:,.2f} seconds")
print(f"      Training time: {rf_simple_time:.4f}s")

print(f"\n  Feature Importance (SIMPLA):")

# In coduld de mai jos, se itereaza prin lista de features numerice (numeric_features: 'pageviews', 'visitNumber', 'hits') si 
# valorile de importanta a acestor features (rf_simple_model.feature_importances_) 
# pentru modelul de regresie Random Forest in versiunea sa simpla. 
# .feature_importances_ este un atribut al modelului Random Forest care contine valorile de importanta a fiecarei features in predictia variabilei tinta (timeOnSite).
# Valorile de importanta sunt formatate cu 4 zecimale pentru a oferi o perspectiva clara asupra contributiei relative a fiecarei features numerice in cadrul modelului de regresie Random Forest in versiunea sa simpla. Acest lucru ajuta la intelegerea care dintre features-urile numerice originale are cel mai mare impact asupra predictiei timpului petrecut pe site (timeOnSite) in aceasta versiune a modelului.
# Pentru fiecare feature si importanta sa, se afiseaza numele feature-ului si valoarea importantei formatata cu 4 zecimale. Aceasta informatie ofera o perspectiva asupra contributiei fiecarui feature numeric la predictia timpului petrecut pe site (timeOnSite) in versiunea simpla a modelului de regresie Random Forest, ajutand astfel la intelegerea importantei relative a acestor features in cadrul modelului.
# zip() este o functie built-in in Python care combina doua sau mai multe iterabile (in acest caz, lista de features numerice si lista de valori de importanta) intr-un singur iterabil de tupluri, unde fiecare tuplu contine cate un element din fiecare dintre iterabilele originale. 
# In acest caz, zip(numeric_features, rf_simple_model.feature_importances_) va returna un iterabil de tupluri, 
# unde fiecare tuplu va contine numele unei features numerice si valoarea de importanta corespunzatoare acesteia in 
# modelul de regresie Random Forest in versiunea sa simpla.
for feat, imp in zip(numeric_features, rf_simple_model.feature_importances_):
    print(f"         {feat}: {imp:.4f}")
    # feat reprezinta numele unei features numerice (ex: 'pageviews', 'visitNumber', 'hits'), iar imp reprezinta valoarea de importanta a acelei features in modelul de regresie Random Forest in versiunea sa simpla. 
    # Acest loop va afisa numele fiecarei features numerice si valoarea de importanta formatata cu 4 zecimale, 
    # oferind astfel o perspectiva clara asupra contributiei relative a fiecarei features numerice in cadrul modelului de regresie Random Forest in versiunea sa simpla.


# VERSIUNEA IMBUNATATITA
print(f"\n  Versiune IMBUNATATITA ({X_enhanced.shape[1]} features):")
start_time = time.time()

# Cream modelul de regresie Random Forest pentru versiunea imbunatatita, folosind toate features-urile disponibile (numerice, categorice encodate si engineered) din setul de date. Acest model va invata relatia dintre aceste features si variabila tinta (timeOnSite) prin construirea unui ansamblu de arbori de decizie, oferind astfel o abordare non-liniara pentru predictia timpului petrecut pe site in versiunea sa imbunatatita care include toate features-urile.
rf_enhanced_model = RandomForestRegressor(
    # n_estimators=100,
    n_estimators=300,
    # max_depth=15,
    max_depth=30,
    # min_samples_split=10,    
    min_samples_split=20,
    # min_samples_leaf=5,
    min_samples_leaf=40,
    random_state=42,
    n_jobs=-1
)

#Antrenam modelul
rf_enhanced_model.fit(X_enhanced_train, y_train_enh)

# Facem predictii folosind modelul de regresie Random Forest antrenat pe setul de testare (X_enhanced_test) pentru a evalua performanta acestui model in predictia timpului petrecut pe site. Aceste predictii vor fi comparate cu valorile reale ale variabilei tinta (y_test_enh) pentru a calcula metrici de evaluare precum R², RMSE si MAE, oferind o perspectiva asupra acuratetei si eficientei modelului de regresie Random Forest in versiunea sa imbunatatita care include toate features-urile (numerice, categorice encodate si engineered).
y_rf_enhanced_pred = rf_enhanced_model.predict(X_enhanced_test)

# Calculam timpul de antrenare
rf_enhanced_time = time.time() - start_time

# Calculam coeficientul de determinare R², RMSE si MAE pentru modelul de regresie Random Forest in versiunea sa imbunatatita, comparand predictiile generate (y_rf_enhanced_pred) cu valorile reale ale variabilei tinta (y_test_enh). Acesti metrici ne vor ajuta sa evaluam acuratetea si performanta modelului de regresie Random Forest in versiunea sa imbunatatita in predictia timpului petrecut pe site, oferind o perspectiva asupra imbunatatirilor aduse prin adaugarea features-urilor categorice si celor create prin feature engineering in aceasta versiune a modelului de regresie Random Forest.
rf_enhanced_r2 = r2_score(y_test_enh, y_rf_enhanced_pred)
rf_enhanced_rmse = np.sqrt(mean_squared_error(y_test_enh, y_rf_enhanced_pred))
rf_enhanced_mae = mean_absolute_error(y_test_enh, y_rf_enhanced_pred)


# Cream un dictionar pentru a stoca rezultatele modelului de regresie Random Forest in versiunea sa imbunatatita, incluzand modelul antrenat (rf_enhanced_model), predictiile generate (y_rf_enhanced_pred), metricii de evaluare (R², RMSE, MAE) si timpul de antrenare (rf_enhanced_time). Acest dictionar va fi folosit pentru a compara performanta modelului imbunatatit cu cea a modelului simplu, oferind o perspectiva clara asupra imbunatatirilor aduse prin adaugarea features-urilor categorice si celor create prin feature engineering in versiunea imbunatatita a modelului de regresie Random Forest.
results_enhanced['Random Forest'] = {
    'model': rf_enhanced_model,
    'predictions': y_rf_enhanced_pred,
    'r2': rf_enhanced_r2,
    'rmse': rf_enhanced_rmse,
    'mae': rf_enhanced_mae,
    'time': rf_enhanced_time
}

print(f"      R² Score: {rf_enhanced_r2:.4f}")
print(f"      RMSE: {rf_enhanced_rmse:,.2f} seconds")
print(f"      MAE: {rf_enhanced_mae:,.2f} seconds")
print(f"      Training time: {rf_enhanced_time:.4f}s")

print(f"\n Feature Importance (TOP 10 IMBUNATATITA):")
feature_names_all = list(X_enhanced.columns)
feature_importances = list(zip(feature_names_all, rf_enhanced_model.feature_importances_)) # rf_enhanced_model.feature_importances_ este un atribut al modelului de regresie Random Forest care contine valorile de importanta a fiecarei features in predictia variabilei tinta (timeOnSite) pentru versiunea imbunatatita a modelului. Aceste valori de importanta sunt asociate cu numele corespunzatoarelor features din lista feature_names_all, care contine toate numele features-urilor disponibile in setul de date imbunatatit (numerice, categorice encodate si engineered).
feature_importances_sorted = sorted(feature_importances, key=lambda x: x[1], reverse=True)
for feat, imp in feature_importances_sorted[:10]:
    print(f"         {feat}: {imp:.4f}")

# Comparatie
improvement_r2 = rf_enhanced_r2 - rf_simple_r2
improvement_pct = (improvement_r2 / rf_simple_r2) * 100
print(f"\n IMBUNATATIRE in cazul modelului Random Forest: R² {improvement_r2:+.4f} ({improvement_pct:+.1f}%)")

# ===========================
# 6. COMPARAREA REZULTATELOR (SIMPLA vs IMBUNATATITA)
# ===========================
print("\n" + "="*80)
print("[6/7] COMPARAREA PERFORMANTEI MODELELOR")
print("="*80)

# Cream DataFrames pentru comparatie
comparison_simple = pd.DataFrame({
    'Model': list(results_simple.keys()),
    'R² Score': [results_simple[m]['r2'] for m in results_simple.keys()], # results_simple[m]['r2'] accesa valoarea R² pentru fiecare model stocata in dictionarul results_simple, unde m reprezinta numele modelului (ex: 'Polynomial Regression', 'Random Forest'). Acest loop va extrage valorile R² pentru toate modelele din versiunea simpla si le va include in coloana 'R² Score' a DataFrame-ului comparison_simple, oferind astfel o perspectiva clara asupra performantei fiecarui model in versiunea sa simpla.
    'RMSE (s)': [results_simple[m]['rmse'] for m in results_simple.keys()],
    'MAE (s)': [results_simple[m]['mae'] for m in results_simple.keys()],
    'Time (s)': [results_simple[m]['time'] for m in results_simple.keys()]
})

comparison_enhanced = pd.DataFrame({
    'Model': list(results_enhanced.keys()),
    'R² Score': [results_enhanced[m]['r2'] for m in results_enhanced.keys()],
    'RMSE (s)': [results_enhanced[m]['rmse'] for m in results_enhanced.keys()],
    'MAE (s)': [results_enhanced[m]['mae'] for m in results_enhanced.keys()],
    'Time (s)': [results_enhanced[m]['time'] for m in results_enhanced.keys()]
})

# Sortam dupa R² Score
comparison_simple = comparison_simple.sort_values('R² Score', ascending=False).reset_index(drop=True)
comparison_enhanced = comparison_enhanced.sort_values('R² Score', ascending=False).reset_index(drop=True)

print("\nVERSIUNEA SIMPLA (3 features numerice):")
print(comparison_simple.to_string(index=False)) # comparison_simple.to_string(index=False) va afisa intregul DataFrame comparison_simple intr-un format tabular, fara a include indexul (index=False) pentru a oferi o perspectiva clara asupra performantei fiecarui model in versiunea sa simpla, concentrandu-se doar pe numele modelului, R² Score, RMSE, MAE si timpul de antrenare pentru fiecare model in parte. Acest lucru ajuta la intelegerea rapida a rezultatelor si la compararea performantei intre modelele din versiunea simpla.

print("\nVERSIUNEA IMBUNATATITA (features categorice + engineered):")
print(comparison_enhanced.to_string(index=False)) # comparison_enhanced.to_string(index=False) va afisa intregul DataFrame comparison_enhanced intr-un format tabular, fara a include indexul (index=False) pentru a oferi o perspectiva clara asupra performantei fiecarui model in versiunea sa imbunatatita, concentrandu-se doar pe numele modelului, R² Score, RMSE, MAE si timpul de antrenare pentru fiecare model in parte. Acest lucru ajuta la intelegerea rapida a rezultatelor si la compararea performantei intre modelele din versiunea imbunatatita.

# Identificam cel mai bun model pentru fiecare versiune
best_simple_name = comparison_simple.iloc[0]['Model'] # comparison_simple.iloc[0]['Model'] acceseaza numele modelului care are cel mai mare R² Score in versiunea simpla, deoarece DataFrame-ul comparison_simple a fost sortat descrescator dupa coloana 'R² Score'. Acest lucru ne permite sa identificam rapid care model a avut cea mai buna performanta in versiunea sa simpla, oferind astfel o perspectiva clara asupra celui mai eficient model in predictia timpului petrecut pe site (timeOnSite) folosind doar features-urile numerice originale.
best_simple_r2 = comparison_simple.iloc[0]['R² Score']

best_enhanced_name = comparison_enhanced.iloc[0]['Model'] # comparison_enhanced.iloc[0]['Model'] acceseaza numele modelului care are cel mai mare R² Score in versiunea imbunatatita, deoarece DataFrame-ul comparison_enhanced a fost sortat descrescator dupa coloana 'R² Score'. Acest lucru ne permite sa identificam rapid care model a avut cea mai buna performanta in versiunea sa imbunatatita, oferind astfel o perspectiva clara asupra celui mai eficient model in predictia timpului petrecut pe site (timeOnSite) folosind features-urile originale si cele noi.
best_enhanced_r2 = comparison_enhanced.iloc[0]['R² Score']

print(f"\n" + "+"*3)
print(f"CEL MAI BUN MODEL - VERSIUNEA SIMPLA:")
print(f"   Model: {best_simple_name}")
print(f"   R² Score: {best_simple_r2:.4f}")
print(f"   RMSE: {comparison_simple.iloc[0]['RMSE (s)']:,.2f} seconds")
print(f"   MAE: {comparison_simple.iloc[0]['MAE (s)']:,.2f} seconds")
print(f"\n" + "+"*3)
print(f"CEL MAI BUN MODEL - VERSIUNEA IMBUNATATITA:")
print(f"   Model: {best_enhanced_name}")
print(f"   R² Score: {best_enhanced_r2:.4f}")
print(f"   RMSE: {comparison_enhanced.iloc[0]['RMSE (s)']:,.2f} seconds")
print(f"   MAE: {comparison_enhanced.iloc[0]['MAE (s)']:,.2f} seconds")
print(f"+"*3)

# Calculam imbunatatirea
overall_improvement = best_enhanced_r2 - best_simple_r2
overall_improvement_pct = (overall_improvement / best_simple_r2) * 100

print(f"\n IMBUNATATIRE TOTALA:")
print(f"   Δ R² Score: {overall_improvement:+.4f} puncte")
print(f"   Δ Procentual: {overall_improvement_pct:+.2f}%")

if overall_improvement > 0.05:
    print(f"\n   IMBUNATATIRE SEMNIFICATIVA! Features-urile noi aduc valoare reala!")
elif overall_improvement > 0.02:
    print(f"\n   Imbunatatire moderata. Features-urile noi ajuta.")
else:
    print(f"\n   Imbunatatire minima. Features-urile noi nu aduc prea multa valoare.")

# Tabel comparativ detaliat pentru fiecare model
print(f"\nCOMPARATIE DETALIATA PE FIECARE MODEL:")
print(f"\n{'Model':<25} {'R² Simpla':<12} {'R² Imbun.':<12} {'Δ R²':<10} {'Δ %':<10}")
print("="*70)

for model_name in results_simple.keys():
    r2_simple = results_simple[model_name]['r2']
    r2_enhanced = results_enhanced[model_name]['r2']
    delta_r2 = r2_enhanced - r2_simple
    delta_pct = (delta_r2 / r2_simple) * 100
    print(f"{model_name:<25} {r2_simple:<12.4f} {r2_enhanced:<12.4f} {delta_r2:+<10.4f} {delta_pct:+<10.2f}%")

# ===========================
# 7. VIZUALIZARI (COMPARATIE SIMPLA vs IMBUNATATITA)
# ===========================
print("\n[7/7] Generarea vizualizarilor...")

fig = plt.figure(figsize=(12, 8)) 

# Subplot 1: Comparatie R² - SIMPLA vs IMBUNATATITA
ax1 = plt.subplot(2, 4, 1) # ax1 reprezinta primul subplot dintr-o figura cu 2 randuri si 4 coloane, unde vom afisa comparatia R² intre versiunea simpla si cea imbunatatita a modelelor. Acest subplot va contine un grafic de tip bar plot orizontal (barh) care va arata valorile R² pentru fiecare model in ambele versiuni, oferind o perspectiva vizuala clara asupra imbunatatirilor aduse prin adaugarea features-urilor categorice si celor create prin feature engineering in versiunea imbunatatita a modelelor.
model_names = list(results_simple.keys())
r2_simple_values = [results_simple[m]['r2'] for m in model_names]
r2_enhanced_values = [results_enhanced[m]['r2'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax1.barh(x - width/2, r2_simple_values, width, label='Simpla', color='#3498db', edgecolor='black') # ax1.barh(x - width/2, r2_simple_values, width, label='Simpla', color='#3498db', edgecolor='black') va crea un grafic de tip bar plot orizontal (barh) in ax1, unde fiecare bara va reprezenta valoarea R² pentru fiecare model in versiunea simpla. x - width/2 pozitioneaza barele pentru versiunea simpla putin mai jos pe axa y pentru a le diferentia de cele ale versiunii imbunatatite. width specifica latimea fiecarei bare, iar label='Simpla' adauga o eticheta pentru legenda. color='#3498db' seteaza culoarea barelor pentru versiunea simpla, iar edgecolor='black' adauga o margine neagra la fiecare bara pentru a le face mai vizibile.
bars2 = ax1.barh(x + width/2, r2_enhanced_values, width, label='Imbunatatita', color='#2ecc71', edgecolor='black')

ax1.set_xlabel('R² Score', fontweight='bold', fontsize=7)
ax1.set_ylabel('Model', fontweight='bold', fontsize=7)
ax1.set_title('Comparatie R² Score\nSIMPLA vs IMBUNATATITA', fontweight='bold', fontsize=7)
ax1.set_yticks(x)
ax1.set_yticklabels(model_names, fontsize=7)
ax1.legend(fontsize=7)
ax1.grid(axis='x', alpha=0.3)
ax1.locator_params(axis='x', nbins=6)  # Marim numarul de gradatii pe axa orizontala
ax1.tick_params(axis='x', labelsize=7)  # Marimea fontului pentru valorile de pe axa X

# Adaugam valorile pe bare
for i, (v1, v2) in enumerate(zip(r2_simple_values, r2_enhanced_values)):
    ax1.text(v1 + 0.01, i - width/2, f'{v1:.3f}', va='center', fontsize=7)
    ax1.text(v2 + 0.01, i + width/2, f'{v2:.3f}', va='center', fontsize=7, fontweight='bold')

# Subplot 2: Improvement per model
ax2 = plt.subplot(2, 4, 2)
improvements = [results_enhanced[m]['r2'] - results_simple[m]['r2'] for m in model_names]
colors_imp = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
bars = ax2.barh(model_names, improvements, color=colors_imp, edgecolor='black', linewidth=1.5)
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('Δ R² Score', fontweight='bold', fontsize=7)
ax2.set_title('Imbunatatire per Model\n(Imbunatatita - Simpla)', fontweight='bold', fontsize=7)
ax2.grid(axis='x', alpha=0.3)
ax2.locator_params(axis='x', nbins=6)  # Marim numarul de gradatii pe axa orizontala
ax2.tick_params(axis='y', labelsize=7)  # Marimea fontului pentru labelurile de pe axa Y (model names)
ax2.tick_params(axis='x', labelsize=7)  # Marimea fontului pentru valorile de pe axa X
for i, (bar, val) in enumerate(zip(bars, improvements)):
    ax2.text(val + 0.005 if val > 0 else val - 0.005, i, f'{val:+.4f}', va='center', fontsize=7, fontweight='bold')

# Subplot 3: MAE Comparison
ax3 = plt.subplot(2, 4, 3)
mae_simple_values = [results_simple[m]['mae'] for m in model_names]
mae_enhanced_values = [results_enhanced[m]['mae'] for m in model_names]
bars1 = ax3.barh(x - width/2, mae_simple_values, width, label='Simpla', color='#e67e22', edgecolor='black')
bars2 = ax3.barh(x + width/2, mae_enhanced_values, width, label='Imbunatatita', color='#9b59b6', edgecolor='black')
ax3.set_xlabel('MAE (seconds)', fontweight='bold', fontsize=7)
ax3.set_title('Comparatie MAE\n(mai mic = mai bun)', fontweight='bold', fontsize=7)
ax3.set_yticks(x)
ax3.set_yticklabels(model_names, fontsize=7)
ax3.legend(fontsize=7)
ax3.grid(axis='x', alpha=0.3)
ax3.locator_params(axis='x', nbins=6)  # Marim numarul de gradatii pe axa orizontala
ax3.tick_params(axis='x', labelsize=7)  # Marimea fontului pentru valorile de pe axa X

# Subplot 4: Cel mai bun model - Reale vs Prezise (SIMPLA)
ax4 = plt.subplot(2, 4, 4)
best_simple_pred = results_simple[best_simple_name]['predictions']
sample_size = min(2000, len(y_test))
indices = np.random.choice(len(y_test), sample_size, replace=False)
ax4.scatter(y_test.iloc[indices], best_simple_pred[indices], 
           alpha=0.4, s=15, c='steelblue', edgecolor='none')
max_val = max(y_test.max(), best_simple_pred.max())
ax4.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect')
ax4.set_xlabel('Reale (s)', fontweight='bold', fontsize=7)
ax4.set_ylabel('Prezise (s)', fontweight='bold', fontsize=7)
ax4.set_title(f'{best_simple_name} - SIMPLA\nR²={best_simple_r2:.4f}', fontweight='bold', fontsize=7)
ax4.legend(fontsize=7)
ax4.grid(True, alpha=0.3)
ax4.locator_params(axis='x', nbins=6)  # Marim numarul de gradatii pe axa orizontala
ax4.locator_params(axis='y', nbins=6)  # Marim numarul de gradatii pe axa verticala
ax4.tick_params(axis='x', labelsize=7)  # Marimea fontului pentru valorile de pe axa X
ax4.tick_params(axis='y', labelsize=7)  # Marimea fontului pentru valorile de pe axa Y

# Subplot 5: Cel mai bun model - Reale vs Prezise (IMBUNATATITA)
ax5 = plt.subplot(2, 4, 5)
best_enhanced_pred = results_enhanced[best_enhanced_name]['predictions']
indices_enh = np.random.choice(len(y_test_enh), sample_size, replace=False)
ax5.scatter(y_test_enh.iloc[indices_enh], best_enhanced_pred[indices_enh], 
           alpha=0.4, s=15, c='#2ecc71', edgecolor='none')
max_val_enh = max(y_test_enh.max(), best_enhanced_pred.max())
ax5.plot([0, max_val_enh], [0, max_val_enh], 'r--', lw=2, label='Perfect')
ax5.set_xlabel('Reale (s)', fontweight='bold', fontsize=7)
ax5.set_ylabel('Prezise (s)', fontweight='bold', fontsize=7)
ax5.set_title(f'{best_enhanced_name} - IMBUNATATITA\nR²={best_enhanced_r2:.4f}', fontweight='bold', fontsize=7)
ax5.legend(fontsize=7)
ax5.grid(True, alpha=0.3)
ax5.locator_params(axis='x', nbins=6)  # Marim numarul de gradatii pe axa orizontala
ax5.locator_params(axis='y', nbins=6)  # Marim numarul de gradatii pe axa verticala
ax5.tick_params(axis='x', labelsize=7)  # Marimea fontului pentru valorile de pe axa X
ax5.tick_params(axis='y', labelsize=7)  # Marimea fontului pentru valorile de pe axa Y
# Marcam cu border verde
for spine in ax5.spines.values():
    spine.set_color('#2ecc71')
    spine.set_linewidth(3)

# Subplot 6: Feature Importance - Random Forest Enhanced (TOP 10)
ax6 = plt.subplot(2, 4, 6)
if 'Random Forest' in results_enhanced:
    feature_names_all = list(X_enhanced.columns)
    importances_all = rf_enhanced_model.feature_importances_
    top_10_idx = np.argsort(importances_all)[::-1][:10]
    top_10_features = [feature_names_all[i] for i in top_10_idx]
    top_10_values = [importances_all[i] for i in top_10_idx]
    
    colors_feat = ['#2ecc71' if v > 0.1 else '#3498db' for v in top_10_values]
    bars = ax6.barh(range(len(top_10_features)), top_10_values, color=colors_feat, edgecolor='black')
    ax6.set_yticks(range(len(top_10_features)))
    ax6.set_yticklabels(top_10_features, fontsize=7)
    ax6.set_xlabel('Importance', fontweight='bold', fontsize=7)
    ax6.set_title('Random Forest (Enhanced)\nTOP 10 Features', fontweight='bold', fontsize=7)
    ax6.grid(axis='x', alpha=0.3)
    ax6.locator_params(axis='x', nbins=6)  # Marim numarul de gradatii pe axa orizontala
    ax6.tick_params(axis='x', labelsize=7)  # Marimea fontului pentru valorile de pe axa X
    for i, (bar, val) in enumerate(zip(bars, top_10_values)):
        ax6.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=7)

# Subplot 7: Distributia erorilor - Simpla (eroarea reprezinta diferenta dintre valorile reale si cele prezise, adica y_test - y_pred)
# Reprezentarea grafica este o distributie a erorilor in jurul liniei punctate rosii care marcheaza eroarea zero, oferind o perspectiva vizuala asupra acuratetii predictiilor modelului simplu. O distributie stransa in jurul liniei zero indica predictii mai precise, in timp ce o distributie larga sau asimetrica poate sugera prezenta unor erori semnificative sau a unor bias-uri in predictii.
ax7 = plt.subplot(2, 4, 7)
errors_simple = y_test.values - best_simple_pred
ax7.hist(errors_simple, bins=40, color='#3498db', edgecolor='black', alpha=0.7)
ax7.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax7.set_xlabel('Eroare (s)', fontweight='bold', fontsize=7)
ax7.set_ylabel('Frecventa', fontweight='bold', fontsize=7)
ax7.set_title(f'Distributia Erorilor - SIMPLA\nStd: {np.std(errors_simple):.2f}s', fontweight='bold', fontsize=7)
ax7.grid(True, alpha=0.3)
ax7.locator_params(axis='x', nbins=6)  # Marim numarul de gradatii pe axa orizontala
ax7.tick_params(axis='x', labelsize=7)  # Marimea fontului pentru valorile de pe axa X

# Subplot 8: Distributia erorilor - Imbunatatita
ax8 = plt.subplot(2, 4, 8)
errors_enhanced = y_test_enh.values - best_enhanced_pred
ax8.hist(errors_enhanced, bins=40, color='#2ecc71', edgecolor='black', alpha=0.7)
ax8.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax8.set_xlabel('Eroare (s)', fontweight='bold', fontsize=7)
ax8.set_ylabel('Frecventa', fontweight='bold', fontsize=7)
ax8.set_title(f'Distributia Erorilor - IMBUNATATITA\nStd: {np.std(errors_enhanced):.2f}s', fontweight='bold', fontsize=7)
ax8.grid(True, alpha=0.3)
ax8.locator_params(axis='x', nbins=6)  # Marim numarul de gradatii pe axa orizontala
ax8.tick_params(axis='x', labelsize=7)  # Marimea fontului pentru valorile de pe axa X
# Marcam cu border verde
for spine in ax8.spines.values():
    spine.set_color('#2ecc71')
    spine.set_linewidth(3)

plt.suptitle('SCENARIUL 2: PREDICTIA ENGAGEMENT - COMPARATIE SIMPLA vs IMBUNATATITA', 
            fontweight='bold', fontsize=10, y=0.995)
plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=1.0)  # Reducere spatiu dintre grafice: pad=padding general, h_pad=padding orizontal, w_pad=padding vertical
plt.savefig('C:/Users/User/Desktop/repos/Project_repo/visualizations/scenario2_enhanced_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Grafice salvate: visualizations/scenario2_enhanced_comparison.png")
plt.show()

# ===========================
# CONCLUZII FINALE
# ===========================
print("\n" + "="*80)
print("CONCLUZII FINALE - SCENARIUL 2 IMBUNATATIT")
print("="*80)

print(f"\nAnaliza completa finalizata cu succes!")
print(f"Testate: {len(results_simple)} modele × 2 versiuni = {len(results_simple)*2} configuratii")

print(f"\n CEL MAI BUN MODEL:")
print(f"   Model: {best_enhanced_name}")
print(f"   R² Score: {best_enhanced_r2:.4f}")
print(f"   RMSE: {comparison_enhanced.iloc[0]['RMSE (s)']:,.2f} seconds")
print(f"   MAE: {comparison_enhanced.iloc[0]['MAE (s)']:,.2f} seconds")

print(f"\n IMBUNATATIRE FATA DE VERSIUNEA SIMPLA:")
print(f"   Δ R² Score: {overall_improvement:+.4f} puncte ({overall_improvement_pct:+.2f}%)")
print(f"   Versiune Simpla: R² = {best_simple_r2:.4f}")
print(f"   Versiune Imbunatatita: R² = {best_enhanced_r2:.4f}")

if overall_improvement > 0.05:
    print(f"\n  IMBUNATATIRE SEMNIFICATIVA!")
    print(f"   => Features categorice si engineered aduc valoare substantiala!")
elif overall_improvement > 0.02:
    print(f"\n  Imbunatatire moderata.")
    print(f"   => Features noi ajuta, dar impact limitat.")
else:
    print(f"\n  Imbunatatire minima.")
    print(f"   => Features noi nu aduc prea multa valoare pentru acest dataset.")

# Evaluare calitativa
print(f"\nEVALUARE CALITATIVA:")
if best_enhanced_r2 >= 0.7:
    print(f"Modelul ofera predictii foarte bune daca si numai daca R² ≥ 0.7 => rezultate excelente.")
elif best_enhanced_r2 >= 0.5:
    print(f"Modelul ofera predictii acceptabile daca si numai daca R² ≥ 0.5")
elif best_enhanced_r2 >= 0.3:
    print(f"Modelul ofera predictii limitate daca si numai daca R² ≥ 0.3")
else:
    print(f"Modelul nu ofera predictii foarte utile daca R² < 0.3")

print(f"\nCOMPARATIE CU SCENARIUL 1:")
print(f"   Scenariul 1 (Predictia Venituri): R² ≈ 0.029 (3%)")
print(f"   Scenariul 2 (Predictia Engagement): R² = {best_enhanced_r2:.4f} ({best_enhanced_r2*100:.1f}%)")
print(f"   → Scenariul 2 este {best_enhanced_r2/0.029:.1f}x MAI BUN!")

print(f"\nCONCLUZII:")
print(f"   1. De folosit modelul {best_enhanced_name} cu features imbunatatite.")
print(f"   2. De monitorizat performanta pe date noi (R² > {best_enhanced_r2:.2f}).")
print(f"   3. De continuat feature engineering pe viitor  astfel:")
print(f"      - Adaugare mai multe features categorice daca sunt disponibile;")
print(f"      - Testare interactiuni intre features (pageviews × channelGrouping, etc.);")
print(f"      - Explorare agregari temporale (zi saptamana, ora zilei);")
print(f"   4. Testare diferite valori pe urmatorii hyperparameters pentru Random Forest:")
print(f"      - n_estimators: 1500-3000 pentru varianta simpla, 300-500 pentru varianta imbunatatita")
print(f"      - max_depth: 20-30")
print(f"      - min_samples_split: 20 si chiar 50 pentru a reduce overfitting-ul")
print(f"      - min_samples_leaf: 40-50 pentru a reduce overfitting-ul")
print(f"   5. Considerare modele ensemble (combinatie Linear + Random Forest), respectiv modele de tip Gradient Boosting (XGBoost, LightGBM) pentru imbunatatirea performantei dar numai dupa ce a fost crescut numarul de features relevante(pe actualul numar de caracteristici 'numerice' XGBoost si LightGBM au fost testate si dau rezultate mai mici decat RandomForest).")

print(f"\nINSIGHTS IMPORTANTE:")
if 'Random Forest' in results_enhanced:
    print(f"   - Feature-ul cel mai important: {top_10_features[0]} ({top_10_values[0]:.3f})")
    print(f"   - Features engineered contribuie la performanta modelului")
    print(f"   - Diversitatea de features ajuta la capturarea pattern-urilor complexe")

print("\n" + "="*80)
print("ANALIZA FINALIZATA CU SUCCES!")
print("="*80)
print(f"\n Fisiere generate:")
print(f"   - visualizations/scenario2_enhanced_comparison.png")
print("="*80)
