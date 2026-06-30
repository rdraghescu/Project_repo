"""Script rapid pentru a afișa doar rezultatele cheie"""
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ANALIZA RAPIDA - REZULTATE CHEIE")
print("="*80)

# Incarcarea datelor
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'ga-sessions.csv')
data = pd.read_csv(DATA_PATH)
print(f"\n✓ Date incarcate: {data.shape[0]:,} randuri")

# Features numerice
numeric_features = ['pageviews', 'visitNumber', 'hits']

# Tratarea valorilor lipsa
for col in numeric_features + ['timeOnSite']:
    if data[col].isnull().sum() > 0:
        median_val = data[col].median()
        data[col].fillna(median_val, inplace=True)
        
y = data['timeOnSite'].copy()

# Corelatia
print(f"\nCORELATIA cu timeOnSite:")
correlation = data[numeric_features + ['timeOnSite']].corr()['timeOnSite'].sort_values(ascending=False)
for feat, corr in correlation.items():
    if feat != 'timeOnSite':
        print(f"   {feat:20s}: {corr:.4f}")

# Versiunea SIMPLA
X_simple = data[numeric_features].copy()
X_simple_train, X_simple_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)

scaler_simple = StandardScaler()
X_simple_train_scaled = scaler_simple.fit_transform(X_simple_train)
X_simple_test_scaled = scaler_simple.transform(X_simple_test)

results_simple = {}

# 1. Linear Regression - SIMPLA
lr_simple = LinearRegression()
lr_simple.fit(X_simple_train_scaled, y_train)
y_lr_simple_pred = lr_simple.predict(X_simple_test_scaled)
lr_simple_r2 = r2_score(y_test, y_lr_simple_pred)
results_simple['Linear Regression'] = {'r2': lr_simple_r2}

# 2. Polynomial - SIMPLA
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_simple_train_poly = poly_features.fit_transform(X_simple_train)
X_simple_test_poly = poly_features.transform(X_simple_test)
poly_simple = LinearRegression()
poly_simple.fit(X_simple_train_poly, y_train)
y_poly_simple_pred = poly_simple.predict(X_simple_test_poly)
poly_simple_r2 = r2_score(y_test, y_poly_simple_pred)
results_simple['Polynomial Regression'] = {'r2': poly_simple_r2}

# 3. Random Forest - SIMPLA
rf_simple = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_simple.fit(X_simple_train, y_train)
y_rf_simple_pred = rf_simple.predict(X_simple_test)
rf_simple_r2 = r2_score(y_test, y_rf_simple_pred)
results_simple['Random Forest'] = {'r2': rf_simple_r2}

print(f"\n{'='*80}")
print("REZULTATE - VERSIUNEA SIMPLA (3 features)")
print(f"{'='*80}")
for model_name, results in sorted(results_simple.items(), key=lambda x: x[1]['r2'], reverse=True):
    print(f"{model_name:30s} R² = {results['r2']:.4f}")

# Versiunea IMBUNATATITA (cu features categorice si engineered)
categorical_features = []
if 'channelGrouping' in data.columns:
    categorical_features.append('channelGrouping')
if 'device_category' in data.columns:
    categorical_features.append('device_category')
if 'country' in data.columns:
    top_countries = data['country'].value_counts().head(10).index.tolist()
    data['country_grouped'] = data['country'].apply(lambda x: x if x in top_countries else 'Other')
    categorical_features.append('country_grouped')

if len(categorical_features) > 0:
    categorical_encoded = pd.get_dummies(data[categorical_features], prefix=categorical_features, drop_first=True)
else:
    categorical_encoded = pd.DataFrame()

# Feature engineering
engineered_features = pd.DataFrame()
if 'visitNumber' in data.columns and 'pageviews' in data.columns:
    engineered_features['pageviews_per_visit'] = data['pageviews'] / (data['visitNumber'] + 1)
if 'pageviews' in data.columns and 'hits' in data.columns:
    engineered_features['engagement_score'] = data['pageviews'] * data['hits']
if 'hits' in data.columns and 'pageviews' in data.columns:
    engineered_features['hits_per_pageview'] = data['hits'] / (data['pageviews'] + 1)
if 'pageviews' in data.columns:
    engineered_features['high_pageviews'] = (data['pageviews'] > data['pageviews'].median()).astype(int)
if 'visitNumber' in data.columns:
    engineered_features['is_returning'] = (data['visitNumber'] > 1).astype(int)

X_numeric = data[numeric_features].copy()
X_enhanced = pd.concat([X_numeric, categorical_encoded, engineered_features], axis=1)

X_enhanced_train, X_enhanced_test, y_train_enh, y_test_enh = train_test_split(X_enhanced, y, test_size=0.2, random_state=42)

scaler_enhanced = StandardScaler()
X_enhanced_train_scaled = scaler_enhanced.fit_transform(X_enhanced_train)
X_enhanced_test_scaled = scaler_enhanced.transform(X_enhanced_test)

results_enhanced = {}

# 1. Linear Regression - IMBUNATATITA
lr_enhanced = LinearRegression()
lr_enhanced.fit(X_enhanced_train_scaled, y_train_enh)
y_lr_enhanced_pred = lr_enhanced.predict(X_enhanced_test_scaled)
lr_enhanced_r2 = r2_score(y_test_enh, y_lr_enhanced_pred)
results_enhanced['Linear Regression'] = {'r2': lr_enhanced_r2}

# 2. Polynomial - IMBUNATATITA
poly_features_enh = PolynomialFeatures(degree=2, include_bias=False)
X_enhanced_train_poly = poly_features_enh.fit_transform(X_enhanced_train)
X_enhanced_test_poly = poly_features_enh.transform(X_enhanced_test)
poly_enhanced = LinearRegression()
poly_enhanced.fit(X_enhanced_train_poly, y_train_enh)
y_poly_enhanced_pred = poly_enhanced.predict(X_enhanced_test_poly)
poly_enhanced_r2 = r2_score(y_test_enh, y_poly_enhanced_pred)
results_enhanced['Polynomial Regression'] = {'r2': poly_enhanced_r2}

# 3. Random Forest - IMBUNATATITA
rf_enhanced = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_enhanced.fit(X_enhanced_train, y_train_enh)
y_rf_enhanced_pred = rf_enhanced.predict(X_enhanced_test)
rf_enhanced_r2 = r2_score(y_test_enh, y_rf_enhanced_pred)
results_enhanced['Random Forest'] = {'r2': rf_enhanced_r2}

print(f"\n{'='*80}")
print(f"REZULTATE - VERSIUNEA IMBUNATATITA ({X_enhanced.shape[1]} features)")
print(f"{'='*80}")
for model_name, results in sorted(results_enhanced.items(), key=lambda x: x[1]['r2'], reverse=True):
    print(f"{model_name:30s} R² = {results['r2']:.4f}")

# Feature Importance pentru Random Forest Enhanced
print(f"\n{'='*80}")
print("FEATURE IMPORTANCE - Random Forest (Enhanced) - TOP 10")
print(f"{'='*80}")
feature_names_all = list(X_enhanced.columns)
importances = rf_enhanced.feature_importances_
feature_importance_pairs = list(zip(feature_names_all, importances))
feature_importance_sorted = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)

for i, (feat, imp) in enumerate(feature_importance_sorted[:10], 1):
    print(f"{i:2d}. {feat:35s} {imp:.4f}")

# Comparatie
print(f"\n{'='*80}")
print("COMPARATIE: SIMPLA vs IMBUNATATITA")
print(f"{'='*80}")
print(f"\n{'Model':<30s} {'R² Simpla':<12} {'R² Imbun.':<12} {'Δ R²':<10}")
print("-"*70)
for model_name in results_simple.keys():
    r2_s = results_simple[model_name]['r2']
    r2_e = results_enhanced[model_name]['r2']
    delta = r2_e - r2_s
    print(f"{model_name:<30s} {r2_s:<12.4f} {r2_e:<12.4f} {delta:+<10.4f}")

# CEL MAI BUN MODEL
best_simple_name = max(results_simple, key=lambda x: results_simple[x]['r2'])
best_simple_r2 = results_simple[best_simple_name]['r2']

best_enhanced_name = max(results_enhanced, key=lambda x: results_enhanced[x]['r2'])
best_enhanced_r2 = results_enhanced[best_enhanced_name]['r2']

overall_best_name = best_enhanced_name if best_enhanced_r2 > best_simple_r2 else best_simple_name
overall_best_r2 = max(best_simple_r2, best_enhanced_r2)
overall_best_version = "IMBUNATATITA" if best_enhanced_r2 > best_simple_r2 else "SIMPLA"

print(f"\n{'='*80}")
print("CONCLUZIE FINALA")
print(f"{'='*80}")
print(f"\n✓ CEL MAI BUN MODEL: {overall_best_name} ({overall_best_version})")
print(f"✓ R² Score: {overall_best_r2:.4f}")

if overall_best_r2 >= 0.7:
    print(f"✓✓ EXCELENT: Model foarte bun pentru predictii!")
elif overall_best_r2 >= 0.5:
    print(f"✓ BUN: Model acceptabil pentru predictii.")
elif overall_best_r2 >= 0.3:
    print(f"⚠ MODERAT: Model poate fi imbunatatit.")
else:
    print(f"⚠ SLAB: Model necesita imbunatatiri semnificative.")

print(f"\n{'='*80}")
