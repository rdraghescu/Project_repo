"""
Aceasta pagina contine un model de invatare automata pentru a prezice traficul web. 
Modelul utilizeaza date istorice despre traficul web pentru a face predictii despre numarul de vizitatori in viitor. 
Acest model poate fi folosit pentru a ajuta la planificarea resurselor si la optimizarea performantei site-ului web.
Va genera grafice pentru regresie liniara, regresie logistica, regresie polinomiala, si SVM adica Support Vector Machine, 
pentru a compara performanta acestor modele in prezicerea traficului web. 
De asemenea, va afisa valorile reale si prezise pentru a evalua acuratetea modelului.
Fisierul de date utilizat pentru antrenarea modelului este "website_wata.csv",  
care contine informatii despre traficul web in diferite perioade de timp. 
Modelul va fi antrenat pe aceste date si va face predictii pentru perioadele viitoare.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn #
from sklearn.linear_model import LinearRegression
from IPython.display import display, Latex
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from scipy.stats import alpha
from sklearn.svm import SVR


# Incarcam datele
data = pd.read_csv('website_wata.csv') 
# Afisam primele 5 randuri din date
print(data.head())

"""
   Page Views  Session Duration  Bounce Rate Traffic Source  Time on Page  Previous Visits  Conversion Rate
0           5         11.051381     0.230652        Organic      3.890460                3              1.0
1           4          3.429316     0.391001         Social      8.478174                0              1.0
2           4          1.621052     0.397986        Organic      9.636170                2              1.0
3           5          3.629279     0.180458        Organic      2.071925                3              1.0
4           5          4.235843     0.291541           Paid      1.960654                5              1.0

unde - Page Views: numarul de vizualizari ale paginii
Session Duration: durata medie a sesiunii in secunde
Bounce Rate: procentul de vizitatori care au parasit site-ul dupa vizualizarea unei singure pagini
Traffic Source: sursa traficului (Organic, Social, Paid)
Time on Page: timpul mediu petrecut pe pagina in secunde
Previous Visits: numarul de vizite anterioare ale utilizatorului
Conversion Rate: rata de conversie, adica procentul de vizitatori care au realizat o actiune dorita (de exemplu, achizitie, abonare la newsletter, etc.)
"""

# Afisam informatii cu privire la media, mediana, si alte statistici descriptive pentru fiecare variabila
print(data.describe())

"""
        Page Views  Session Duration  Bounce Rate  Time on Page  Previous Visits  Conversion Rate
count  2000.000000       2000.000000  2000.000000   2000.000000      2000.000000      2000.000000
mean      4.950500          3.022045     0.284767      4.027439         1.978500         0.982065
std       2.183903          3.104518     0.159781      2.887422         1.432852         0.065680
min       0.000000          0.003613     0.007868      0.068515         0.000000         0.343665
25%       3.000000          0.815828     0.161986      1.935037         1.000000         1.000000
50%       5.000000          1.993983     0.266375      3.315316         2.000000         1.000000
75%       6.000000          4.197569     0.388551      5.414627         3.000000         1.000000
max      14.000000         20.290516     0.844939     24.796182         9.000000         1.000000
"""

# Verificam daca exista valori lipsa
# print(data.isnull().sum())
if not data.isnull().sum().any():
    print("Nu exista valori lipsa in date.")

# Verificam corelatia dintre variabilele numerice pentru a identifica care variabile sunt cel mai bine corelate cu Time on Page
# print("\n=== ANALIZA CORELATIEI ===")
# correlation_matrix = data[['Page Views', 'Session Duration', 'Bounce Rate', 'Time on Page', 'Previous Visits']].corr() # .corr() calculeaza matricea de corelatie. O matrice de corelatie este o matrice care arata cat de puternic sunt corelate variabilele intre ele. Valorile variaza intre -1 si 1, unde 1 indica o corelatie pozitiva perfecta, -1 indica o corelatie negativa perfecta, iar 0 indica lipsa unei corelatii.
# print("\nMatricea de corelatie:")
# print(correlation_matrix)
# print("\nCorelatie cu Time on Page:")
# print(correlation_matrix['Time on Page'].sort_values(ascending=False))

"""
Corelatie cu Time on Page:
Time on Page        1.000000
Bounce Rate         0.039340
Page Views          0.023941
Session Duration   -0.013985
Previous Visits    -0.028729
Name: Time on Page, dtype: float64
Coeficientii modelului de regresie liniara: beta0 (intercept) = 4, beta1 (coeficient) = 0.03 ceea ce inseamna ca pentru fiecare vizualizare suplimentara a paginii, timpul petrecut pe pagina creste in medie cu 0.03 secunde, tinand cont de celelalte variabile din model. Interceptul de 4 indica faptul ca atunci cand numarul de vizualizari ale paginii este 0, timpul petrecut pe pagina este in medie de 4 secunde.
"""

# Selectam variabilele independente (X) si variabila dependenta (y)

# Regresie simple liniara pentru a prezice "Time on Page" in functie de "Page Views".
# Pentru cazul cu histogramele, alegem "Page Views" ca variabila independenta (X) si "Time on Page" ca variabila dependenta (y) pentru a vedea cum se coreleaza aceste doua variabile.
X_simple = data['Page Views'].values.reshape(-1, 1)  # Variabila independenta. Am ales "Page Views" pentru a prezice "Time on Page". reshape(-1, 1) transforma seria intr-un array 2D cu o singura coloana, necesar pentru modelele de invatare automata.

# regresia multipla
X_multiple = data[['Page Views', 'Session Duration', 'Bounce Rate', 'Previous Visits']].values  # Variabilele independente pentru regresia multipla. Am ales aceste variabile pentru a vedea cum se coreleaza cu "Time on Page" si pentru a imbunatati performanta modelului prin includerea mai multor factori care pot influenta timpul petrecut pe pagina.

# Pentru cazul cu graficul scatterplot, am ales "Session Duration" ca variabila independenta (X) si "Time on Page" ca variabila dependenta (y) pentru a vedea cum se coreleaza aceste doua variabile.
# X = data['Session Duration'].values.reshape(-1, 1)  # Variabila independenta. Am ales "Session Duration" pentru a prezice "Time on Page"
# values.reshape(-1, 1) este necesar pentru a transforma seria intr-un array 2D, deoarece multe modele de invatare automata asteapta ca datele sa fie in acest format.
# Exemplu: daca avem o serie de valori [1, 2, 3], values.reshape(-1, 1) va transforma aceasta serie intr-un array 2D cu o singura coloana:
# [[1], [2], [3]]   
# Puteam sa scriem si X = data[['Session Duration']] pentru a obtine acelasi rezultat, deoarece data[['Session Duration']] va returna un DataFrame cu o singura coloana, care este deja in formatul 2D necesar.
y = data['Time on Page']  # Variabila dependenta pentru regresia liniara, adica ceea ce vrem sa prezicem. Am ales "Time on Page" pentru a vedea cum se coreleaza cu "Session Duration".


# creeam modelul simplu de regresie liniara
model_simple = LinearRegression()
model_simple.fit(X_simple, y) # antrenam modelul pe datele noastre

# cream modelul de regresie multipla
model_multiple = LinearRegression()
model_multiple.fit(X_multiple, y)

# Sectiunea pentru regresia simpla liniara
# regresia liniara are formula de calcul: y = beta  0 + beta1 * x, unde beta0 este interceptul(valoarea lui y cand x=0) si beta1 este coeficientul pentru variabila independenta, adica cat de mult se schimba y pentru o unitate de schimbare in x.
# Afisam coeficientii modelului
beta0_simple = round(model_simple.intercept_) # Interceptul (valoarea lui y cand x=0)
beta1_simple = round(model_simple.coef_[0], 2) # Coeficientul pentru variabila independenta (cat de mult se schimba y pentru o unitate de schimbare in x), rotunjit la 2 zecimale
print(f"Coeficientii modelului de regresie liniara: beta0 (intercept) = {beta0_simple}, beta1 (coeficient) = {beta1_simple}")

# Facem predictii folosind modelul antrenat
y_pred_simple = model_simple.predict(X_simple)
# Afisam valorile reale si prezise
# print("\nValorile reale vs prezise:")
# for real, pred in zip(y, y_pred_simple):
#     print("Afisam valorile reale si prezise pentru timpul petrecut pe pagina:")
#     print(f"Real: {real}, Prezis: {round(pred, 2)}")

# Evaluam performanta modelului folosind R^2, MSE, si MAE
r2_simple = r2_score(y,y_pred_simple) # R^2 (coeficientul de determinare) masoara cat de bine se potriveste modelul datelor. Valoarea variaza intre 0 si 1, unde 1 indica o potrivire perfecta.
mae_simple = mean_absolute_error(y,y_pred_simple) # MAE (eroarea absoluta medie) masoara media valorilor absolute ale diferentelor dintre valorile reale si cele prezise. O valoare mai mica indica o performanta mai buna a modelului.
mse_simple = mean_squared_error(y,y_pred_simple) # MSE (eroarea patratica medie) masoara media patratelor diferentelor dintre valorile reale si cele prezise. O valoare mai mica indica o performanta mai buna a modelului.

# Cream datele de testare pentru a vizualiza linia de regresie
X_test_simple = np.linspace(X_simple.min(), X_simple.max(), 100).reshape(-1, 1)  # Generam 100 de puncte intre valoarea minima si maxima a lui X_simple
y_test_simple = model_simple.predict(X_test_simple)  # Prezicem valorile lui y pentru datele de testare


# Sectiunea pentru regresia multipla
# Afisam coeficientii modelului de regresie multipla
beta0_multiple = round(model_multiple.intercept_,2) # Interceptul (valoarea lui y cand toate variabilele independente sunt 0)
beta1_multiple = [round(coef, 2) for coef in model_multiple.coef_] # Coeficientii pentru fiecare variabila independenta, rotunjiti la 2 zecimale
print(f"\nCoeficientii modelului de regresie multipla: beta0 (intercept) = {beta0_multiple}, beta1 (coeficienti) = {beta1_multiple}")
# Coeficientii modelului de regresie multipla: beta0 (intercept) = -6.57, beta1 (coeficienti) = [np.float64(-0.01), np.float64(-0.06), np.float64(0.9), np.float64(-0.11), np.float64(10.99)]

# facem predictii folosind modelul de regresie multipla
y_pred_multiple = model_multiple.predict(X_multiple)

# Evaluam performanta modelului de regresie multipla folosind R^2, MSE, si MAE
r2_multiple = r2_score(y, y_pred_multiple)
mae_multiple = mean_absolute_error(y, y_pred_multiple)
mse_multiple = mean_squared_error(y, y_pred_multiple)

# Cream datele de testare pentru a vizualiza linia de regresie multipla
X_test_multiple = np.array([[5, 3, 0.2, 2]])  # Exemplu de date de testare pentru a vizualiza linia de regresie multipla. Aceste valori reprezinta: Page Views=5, Session Duration=3, Bounce Rate=0.2, Previous Visits=2, Conversion Rate=1
y_test_multiple = model_multiple.predict(X_test_multiple)  # Prezicem valoarea lui y pentru datele de testare folosind modelul de regresie multipla care duoa datele existente in fisierul .csv, ar trebui sa fie de aproximativ 2 secunde.




# # Verificam mai multe modele de regresie pentru a vedea daca putem imbunatati performanta modelului nostru. Vom incerca regresia polinomiala si SVM (Support Vector Machine) pentru a compara rezultatele cu cele obtinute prin regresia liniara.

# print("\n" + "="*70)
# print("TESTARE MULTIPLE TIPURI DE REGRESIE")
# print("="*70)

# # 1. Regresie Liniara Simpla (DEJA EXISTA)
# print(f"1. Regresie Liniara Simpla (Page Views):")
# print(f"   R² = {r2_simple:.4f}, MAE = {mae_simple:.2f}, MSE = {mse_simple:.2f}")

# # 2. Regresie Multipla (DEJA EXISTA)
# print(f"\n2. Regresie Liniara Multipla (toate variabilele):")
# print(f"   R² = {r2_multiple:.4f}, MAE = {mae_multiple:.2f}, MSE = {mse_multiple:.2f}")

# # 3. Regresie Polinomiala (grad 2)
# poly_features_2 = PolynomialFeatures(degree=2)
# X_poly_2 = poly_features_2.fit_transform(X_simple)
# model_poly_2 = LinearRegression()
# model_poly_2.fit(X_poly_2, y)
# y_pred_poly_2 = model_poly_2.predict(X_poly_2)
# r2_poly_2 = r2_score(y, y_pred_poly_2)
# mae_poly_2 = mean_absolute_error(y, y_pred_poly_2)
# mse_poly_2 = mean_squared_error(y, y_pred_poly_2)
# print(f"\n3. Regresie Polinomiala (grad 2):")
# print(f"   R² = {r2_poly_2:.4f}, MAE = {mae_poly_2:.2f}, MSE = {mse_poly_2:.2f}")

# # 4. Regresie Polinomiala (grad 3)
# poly_features_3 = PolynomialFeatures(degree=3)
# X_poly_3 = poly_features_3.fit_transform(X_simple)
# model_poly_3 = LinearRegression()
# model_poly_3.fit(X_poly_3, y)
# y_pred_poly_3 = model_poly_3.predict(X_poly_3)
# r2_poly_3 = r2_score(y, y_pred_poly_3)
# mae_poly_3 = mean_absolute_error(y, y_pred_poly_3)
# mse_poly_3 = mean_squared_error(y, y_pred_poly_3)
# print(f"\n4. Regresie Polinomiala (grad 3):")
# print(f"   R² = {r2_poly_3:.4f}, MAE = {mae_poly_3:.2f}, MSE = {mse_poly_3:.2f}")

# # 5. Support Vector Regression (SVR)
# model_svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
# model_svr.fit(X_simple, y)
# y_pred_svr = model_svr.predict(X_simple)
# r2_svr = r2_score(y, y_pred_svr)
# mae_svr = mean_absolute_error(y, y_pred_svr)
# mse_svr = mean_squared_error(y, y_pred_svr)
# print(f"\n5. Support Vector Regression (SVR):")
# print(f"   R² = {r2_svr:.4f}, MAE = {mae_svr:.2f}, MSE = {mse_svr:.2f}")

# print("\n" + "="*70)
# print("CONCLUZIE:")
# print("="*70)

# # Gasim cel mai bun model
# models_r2 = {
#     'Liniara Simpla': r2_simple,
#     'Liniara Multipla': r2_multiple,
#     'Polinomiala grad 2': r2_poly_2,
#     'Polinomiala grad 3': r2_poly_3,
#     'SVR': r2_svr
# }

# best_model = max(models_r2, key=models_r2.get)
# best_r2 = models_r2[best_model]

# print(f"Cel mai bun model: {best_model} cu R² = {best_r2:.4f}")

# if best_r2 < 0.1:
#     print("\n⚠️  ATENTIE: Toate modelele au R² < 0.1!")
#     print("Aceasta inseamna ca datele NU permit predictii utile.")
#     print("Posibile cauze:")
#     print("  - Datele sunt generate ALEATORIU")
#     print("  - Time on Page nu depinde de variabilele disponibile")
#     print("  - Lipsa variabile importante care influenteaza Time on Page")
# elif best_r2 < 0.3:
#     print(f"\n✓ Modelul {best_model} este cel mai bun, dar R² este inca slab.")
# else:
#     print(f"\n✓✓ Modelul {best_model} ofera predictii acceptabile!")



# # Cream un layout cu 2 subplots pentru analiza completa
# fig, axes = plt.subplots(1, 1, figsize=(8,3)) #. subplots(1, 2) creeaza o figura cu 1 rand si 2 coloane de grafice, iar figsize=(16, 6) seteaza dimensiunea totala a figurii la 16 inch latime si 6 inch inaltime.

# # Subplot 1: Regresia generala pentru toate sursele de trafic
# # sns.scatterplot(data=data, x='Session Duration', y='Time on Page', edgecolor='darkblue', hue='Traffic Source', ax=axes, s=30, alpha=0.7)
# # axes.plot(X_test, y_test, color='red', linewidth=1.5, label='Linia de regresie generala', linestyle='--')
# # axes.set_xlabel('Session Duration (seconds)', fontsize=11)
# # axes.set_ylabel('Time on Page (seconds)', fontsize=11)
# # axes.set_title('Regresie Generala: Time on Page vs Session Duration', fontsize=12, fontweight='bold')
# # axes.legend(loc='upper right', fontsize=7)
# # axes.grid(True, alpha=0.3)

# fig, axes = plt.subplots(1, 2, figsize=(16, 6)) #. subplots(1, 2) creeaza o figura cu 1 rand si 2 coloane de grafice, iar figsize=(16, 6) seteaza dimensiunea totala a figurii la 16 inch latime si 6 inch inaltime.

# # Subplot 1: Regresia simpla pentru toate sursele de trafic
# sns.scatterplot(data=data, x='Page Views', y='Time on Page', ax=axes[0], alpha=0.6, s=30, color='steelblue', edgecolor='darkblue')
# axes[0].plot(X_test_simple, y_test_simple, color='red', linewidth=2, label='Linia de regresie generala', linestyle='--')
# axes[0].set_xlabel('Page Views', fontsize=11)
# axes[0].set_ylabel('Time on Page (seconds)', fontsize=11)
# # Afisam valorile R^2, MAE, si MSE in titlul graficului pentru a evalua performanta modelului
# axes[0].set_title(f'Regresie Generala: Time on Page vs Page Views\nR^2 = {r2_simple:.2f}, MAE = {mae_simple:.2f}, MSE = {mse_simple:.2f}', fontsize=12, fontweight='bold')
# axes[0].set_xticks(np.arange(0, data['Page Views'].max() + 1, 1))  # Setam etichetele de pe axa x la fiecare valoare intre 0 si valoarea maxima a lui Page Views, cu un pas de 1
# axes[0].legend(loc='upper right', fontsize=7)
# axes[0].grid(True, alpha=0.3)

# # Subplot 2: Regresia multipla pentru toate sursele de trafic
# axes[1].scatter(y, y_pred_simple, alpha=0.5, color='blue', edgecolor='darkblue', label=f'Simple (R²={r2_simple:.4f}; MAE={mae_simple:.2f}; MSE={mse_simple:.2f})')
# axes[1].scatter(y, y_pred_multiple, alpha=0.5, s=20, color='green', edgecolor='darkgreen', label=f'Multiple (R²={r2_multiple:.4f}; MAE={mae_multiple:.2f}; MSE={mse_multiple:.2f})')
# axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
# axes[1].set_xlabel('Valori reale (Time on Page)', fontsize=11)
# axes[1].set_ylabel('Valori prezise (Time on Page)', fontsize=11)
# axes[1].set_title(f'COMPARATIE: Valori Reale vs Prezise', fontsize=12, fontweight='bold')
# axes[1].legend(loc='upper right', fontsize=7)
# axes[1].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# O sa creez urmatoarele grafice:
# - distributia numarului de vizite pe pagini, 
# - structura vizitatorilor pe categoriile de trafic, 
# - corelatia dintre timpul petrecut pe pagina si 
# - rata de conversie a vizitatorului, durata sesiunii si bounce rate.

# df = data.copy() # Cream o copie a DataFrame-ului original pentru a nu modifica datele originale
# # Creeaz un Layout cu linii si doua colloane pentru a afisa graficele
# fig, axes = plt.subplots(2, 2, figsize=(8, 5)) # subplots(2, 2) creeaza o figura cu 2 randuri si 2 coloane de grafice, iar figsize=(8, 5) seteaza dimensiunea totala a figurii la 8 inch latime si 5 inch inaltime.
# # 1. Distributia numarului de vizite pe pagini
# sns.histplot(df['Page Views'], bins=15, kde=True, color='steelblue', edgecolor='darkblue', ax=axes[0, 0]) 
# # Unde;
# # - bins=15 seteaza numarul de intervale pentru histograma
# # - kde=True adauga o estimare a densitatii nucleului
# # - color='steelblue' seteaza culoarea barelor
# # - edgecolor='darkblue' seteaza culoarea conturului barelor
# # - ax=axes[0, 0] specifica subplot-ul unde va fi desenat graficul

# axes[0, 0].set_title('Distributia numarului de vizite pe pagini', fontsize=7, fontweight='bold')
# axes[0, 0].set_xlabel('Page Views', fontsize=7)
# axes[0, 0].set_ylabel('Frecventa', fontsize=7)

# # 2. Structura vizitatorilor pe categoriile de trafic
# sns.countplot(data=df, x='Traffic Source', palette='Set2', ax=axes[0, 1])
# axes[0, 1].set_title('Structura vizitatorilor pe categoriile de trafic', fontsize=7, fontweight='bold')
# axes[0, 1].set_xlabel('Traffic Source', fontsize=7)
# axes[0, 1].set_ylabel('Numar de vizitatori', fontsize=7)

# # 3. Corelatia dintre timpul petrecut pe pagina si rata de conversie a vizitatorului
# sns.scatterplot(data=df, x='Time on Page', y='Conversion Rate', hue='Traffic Source', palette='Set1', ax=axes[1, 0], alpha=0.7, edgecolor='darkblue')
# axes[1, 0].set_title('Corelatia dintre timpul petrecut pe pagina si rata de conversie', fontsize=7, fontweight='bold')
# axes[1, 0].set_xlabel('Time on Page', fontsize=7)
# axes[1, 0].set_ylabel('Conversion Rate', fontsize=7)
# axes[1, 0].legend(title='Traffic Source', fontsize=7)

# # 4. Corelatia dintre timpul petrecut pe pagina, durata sesiunii si bounce rate
# sns.scatterplot(data=df, x='Session Duration', y='Bounce Rate', palette='coolwarm', ax=axes[1, 1], alpha=0.7, edgecolor='darkblue')
# axes[1, 1].set_title('Corelatia dintre timpul petrecut pe pagina, durata sesiunii si bounce rate', fontsize=7, fontweight='bold')
# axes[1, 1].set_xlabel('Session Duration', fontsize=7)
# axes[1, 1].set_ylabel('Bounce Rate', fontsize=7)
# axes[1, 1].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# sectiunea cu prelucrarea datelor din Web Analytic_Dataset.csv
data_second = pd.read_csv('Web Analytic_Dataset.csv')

# Functie pentru a converti timpul din format HH:MM:SS in secunde
def time_to_seconds(time_str):
    try:
        if pd.isna(time_str) or time_str == '' or time_str == '0':
            return 0
        parts = str(time_str).split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(s)
        return float(time_str)  # Daca e deja un numar
    except:
        return 0

# Curatam datele
print("\n=== CURATAREA DATELOR (Web Analytic Dataset) ===")
print("Coloane disponibile:", data_second.columns.tolist())

# Curatam coloanele numerice cu virgule (ex: 126,870 -> 126870)
numeric_cols_with_commas = ['Users', 'New Users', 'Sessions', 'Pageviews', 'Revenue']
for col in numeric_cols_with_commas:
    if col in data_second.columns:
        data_second[col] = data_second[col].astype(str).str.replace(',', '').astype(float)

# Curatam Bounce Rate - eliminam % si convertim la fractie (ex: 71.59% -> 0.7159)
if 'Bounce Rate' in data_second.columns:
    data_second['Bounce Rate'] = data_second['Bounce Rate'].astype(str).str.replace('%', '').astype(float) / 100

# Convertim Avg. Session Duration din HH:MM:SS in secunde
if 'Avg. Session Duration' in data_second.columns:
    data_second['Avg. Session Duration'] = data_second['Avg. Session Duration'].apply(time_to_seconds)

print("Date curatate cu succes!")
print(data_second.head())
print("\nStatistici dupa curatare:")
print(data_second.describe())


Xa_simple = data_second['Users'].values.reshape(-1, 1)  # Variabila independenta pentru regresia simpla. Am ales "Users" pentru a prezice "Revenue". reshape(-1, 1) transforma seria intr-un array 2D cu o singura coloana, necesar pentru modelele de invatare automata.
ya = data_second['Revenue']  # Variabila dependenta pentru regresia liniara
Xa_multiple = data_second[['Year', 'Month of the year', 'Users', 'New Users', 'Sessions', 'Bounce Rate', 'Pageviews', 'Avg. Session Duration']]

# creeam modelul simplu de regresie liniara
modela_simple = LinearRegression()
modela_simple.fit(Xa_simple, ya) # antrenam modelul pe datele noastre

# cream modelul de regresie multipla
modela_multiple = LinearRegression()
modela_multiple.fit(Xa_multiple, ya)

# Sectiunea pentru regresia simpla liniara
# regresia liniara are formula de calcul: y = beta  0 + beta1 * x, unde beta0 este interceptul(valoarea lui y cand x=0) si beta1 este coeficientul pentru variabila independenta, adica cat de mult se schimba y pentru o unitate de schimbare in x.
# Afisam coeficientii modelului
beta0_simple = round(modela_simple.intercept_) # Interceptul (valoarea lui y cand x=0)
beta1_simple = round(modela_simple.coef_[0], 2) # Coeficientul pentru variabila independenta (cat de mult se schimba y pentru o unitate de schimbare in x), rotunjit la 2 zecimale
print(f"Coeficientii modelului de regresie liniara: beta0 (intercept) = {beta0_simple}, beta1 (coeficient) = {beta1_simple}")

# Facem predictii folosind modelul antrenat
ya_pred_simple = modela_simple.predict(Xa_simple)
# Afisam valorile reale si prezise
# print("\nValorile reale vs prezise:")
# for real, pred in zip(y, y_pred_simple):
#     print("Afisam valorile reale si prezise pentru timpul petrecut pe pagina:")
#     print(f"Real: {real}, Prezis: {round(pred, 2)}")

# Evaluam performanta modelului folosind R^2, MSE, si MAE
r2a_simple = r2_score(ya, ya_pred_simple) # R^2 (coeficientul de determinare) masoara cat de bine se potriveste modelul datelor. Valoarea variaza intre 0 si 1, unde 1 indica o potrivire perfecta.
maea_simple = mean_absolute_error(ya, ya_pred_simple) # MAE (eroarea absoluta medie) masoara media valorilor absolute ale diferentelor dintre valorile reale si cele prezise. O valoare mai mica indica o performanta mai buna a modelului.
msea_simple = mean_squared_error(ya, ya_pred_simple) # MSE (eroarea patratica medie) masoara media patratelor diferentelor dintre valorile reale si cele prezise. O valoare mai mica indica o performanta mai buna a modelului.

# Cream datele de testare pentru a vizualiza linia de regresie
Xa_test_simple = np.linspace(Xa_simple.min(), Xa_simple.max(), 100).reshape(-1, 1)  # Generam 100 de puncte intre valoarea minima si maxima a lui Xa_simple
ya_test_simple = modela_simple.predict(Xa_test_simple)  # Prezicem valorile lui y pentru datele de testare


# Sectiunea pentru regresia multipla
# Afisam coeficientii modelului de regresie multipla
beta0_multiple = round(modela_multiple.intercept_,2) # Interceptul (valoarea lui y cand toate variabilele independente sunt 0)
beta1_multiple = [round(coef, 2) for coef in modela_multiple.coef_] # Coeficientii pentru fiecare variabila independenta, rotunjiti la 2 zecimale
print(f"\nCoeficientii modelului de regresie multipla: beta0 (intercept) = {beta0_multiple}, beta1 (coeficienti) = {beta1_multiple}")
# Coeficientii modelului de regresie multipla: beta0 (intercept) = -6.57, beta1 (coeficienti) = [np.float64(-0.01), np.float64(-0.06), np.float64(0.9), np.float64(-0.11), np.float64(10.99)]

# facem predictii folosind modelul de regresie multipla
ya_pred_multiple = modela_multiple.predict(Xa_multiple)

# Evaluam performanta modelului de regresie multipla folosind R^2, MSE, si MAE
r2a_multiple = r2_score(ya, ya_pred_multiple)
maea_multiple = mean_absolute_error(ya, ya_pred_multiple)
msea_multiple = mean_squared_error(ya, ya_pred_multiple)

# Cream datele de testare pentru a vizualiza linia de regresie multipla
Xa_test_multiple = np.array([[2023, 6, 5000, 3000, 7000, 0.2, 15000, 180]])  # Exemplu de date de testare pentru a vizualiza linia de regresie multipla. Aceste valori reprezinta: Year=2023, Month of the year=6, Users=5000, New Users=3000, Sessions=7000, Bounce Rate=0.2, Pageviews=15000, Avg. Session Duration=180
ya_test_multiple = modela_multiple.predict(Xa_test_multiple)  # Prezicem valoarea lui y pentru datele de testare folosind modelul de regresie multipla care duoa datele existente in fisierul .csv, ar trebui sa fie de aproximativ 2 secunde.

# Verificam mai multe modele de regresie pentru a vedea daca putem imbunatati performanta modelului nostru. Vom incerca regresia polinomiala si SVM (Support Vector Machine) pentru a compara rezultatele cu cele obtinute prin regresia liniara.

print("\n" + "="*70)
print("TESTARE MULTIPLE TIPURI DE REGRESIE")
print("="*70)

# 1. Regresie Liniara Simpla (DEJA EXISTA)
print(f"1. Regresie Liniara Simpla (Users vs Revenue):")
print(f"   R² = {r2a_simple:.4f}, MAE = {maea_simple:.2f}, MSE = {msea_simple:.2f}")

# 2. Regresie Multipla (DEJA EXISTA)
print(f"\n2. Regresie Liniara Multipla (toate variabilele):")
print(f"   R² = {r2a_multiple:.4f}, MAE = {maea_multiple:.2f}, MSE = {msea_multiple:.2f}")

# 3. Regresie Polinomiala (grad 2)
polya_features_2 = PolynomialFeatures(degree=2)
Xa_poly_2 = polya_features_2.fit_transform(Xa_simple)
modela_poly_2 = LinearRegression()
modela_poly_2.fit(Xa_poly_2, ya)
ya_pred_poly_2 = modela_poly_2.predict(Xa_poly_2)
r2a_poly_2 = r2_score(ya, ya_pred_poly_2)
maea_poly_2 = mean_absolute_error(ya, ya_pred_poly_2)
msea_poly_2 = mean_squared_error(ya, ya_pred_poly_2)
print(f"\n3. Regresie Polinomiala (grad 2):")
print(f"   R² = {r2a_poly_2:.4f}, MAE = {maea_poly_2:.2f}, MSE = {msea_poly_2:.2f}")

# 4. Regresie Polinomiala (grad 3)
polya_features_3 = PolynomialFeatures(degree=3)
Xa_poly_3 = polya_features_3.fit_transform(Xa_simple)
modela_poly_3 = LinearRegression()
modela_poly_3.fit(Xa_poly_3, ya)
ya_pred_poly_3 = modela_poly_3.predict(Xa_poly_3)
r2a_poly_3 = r2_score(ya, ya_pred_poly_3)
maea_poly_3 = mean_absolute_error(ya, ya_pred_poly_3)
msea_poly_3 = mean_squared_error(ya, ya_pred_poly_3)
print(f"\n4. Regresie Polinomiala (grad 3):")
print(f"   R² = {r2a_poly_3:.4f}, MAE = {maea_poly_3:.2f}, MSE = {msea_poly_3:.2f}")

# 5. Support Vector Regression (SVR)
model_svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model_svr.fit(Xa_simple, ya)
ya_pred_svr = model_svr.predict(Xa_simple)
r2a_svr = r2_score(ya, ya_pred_svr)
maea_svr = mean_absolute_error(ya, ya_pred_svr)
msea_svr = mean_squared_error(ya, ya_pred_svr)
print(f"\n5. Support Vector Regression (SVR):")
print(f"   R² = {r2a_svr:.4f}, MAE = {maea_svr:.2f}, MSE = {msea_svr:.2f}")

print("\n" + "="*70)
print("CONCLUZIE:")
print("="*70)

# Gasim cel mai bun model
modelsa_r2 = {
    'Liniara Simpla': r2a_simple,
    'Liniara Multipla': r2a_multiple,
    'Polinomiala grad 2': r2a_poly_2,
    'Polinomiala grad 3': r2a_poly_3,
    'SVR': r2a_svr
}

besta_model = max(modelsa_r2, key=modelsa_r2.get)
besta_r2 = modelsa_r2[besta_model]

print(f"Cel mai bun model: {besta_model} cu R² = {besta_r2:.4f}")

if besta_r2 < 0.1:
    print("\n⚠️  ATENTIE: Toate modelele au R² < 0.1!")
    print("Aceasta inseamna ca datele NU permit predictii utile.")
    print("Posibile cauze:")
    print("  - Datele sunt generate ALEATORIU")
    print("  - Revenue nu depinde de variabilele disponibile")
    print("  - Lipsa variabile importante care influenteaza Revenue")
elif besta_r2 < 0.3:
    print(f"\n✓ Modelul {besta_model} este cel mai bun, dar R² este inca slab.")
else:
    print(f"\n✓✓ Modelul {besta_model} ofera predictii acceptabile!")


# ============================================================================
# SECTIUNEA DE VIZUALIZARE - GRAFICE COMPARATIVE
# ============================================================================

print("\n" + "="*70)
print("CREARE GRAFICE COMPARATIVE")
print("="*70)

# Cream un layout cu 4 subplots (2 randuri x 2 coloane)
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# ---- SUBPLOT 1: Regresia Simpla (Users vs Revenue) ----
axes[0, 0].scatter(Xa_simple, ya, alpha=0.5, s=30, color='steelblue', edgecolor='darkblue', label='Date reale')
axes[0, 0].plot(Xa_test_simple, ya_test_simple, color='red', linewidth=2, label='Linia de regresie', linestyle='--')
axes[0, 0].set_xlabel('Users', fontsize=7, fontweight='bold')
axes[0, 0].set_ylabel('Revenue ($)', fontsize=7, fontweight='bold')
axes[0, 0].set_title(f'REGRESIE SIMPLA: Users vs Revenue\nR² = {r2a_simple:.4f} | MAE = ${maea_simple:,.0f} | MSE = {msea_simple:,.0f}', 
                      fontsize=7, fontweight='bold')
axes[0, 0].tick_params(axis='both', which='major', labelsize=7)
axes[0, 0].legend(loc='upper left', fontsize=7)
axes[0, 0].grid(True, alpha=0.3)

# ---- SUBPLOT 2: Predicted vs Actual - Comparatie Simple vs Multiple ----
axes[0, 1].scatter(ya, ya_pred_simple, alpha=0.6, s=40, color='blue', edgecolor='darkblue', 
                   label=f'Simple: R²={r2a_simple:.4f}')
axes[0, 1].scatter(ya, ya_pred_multiple, alpha=0.6, s=40, color='green', edgecolor='darkgreen', 
                   label=f'Multiple: R²={r2a_multiple:.4f}')
axes[0, 1].plot([ya.min(), ya.max()], [ya.min(), ya.max()], 'r--', lw=2, label='Predictie perfecta')
axes[0, 1].set_xlabel('Revenue Real ($)', fontsize=7, fontweight='bold')
axes[0, 1].set_ylabel('Revenue Prezis ($)', fontsize=7, fontweight='bold')
axes[0, 1].set_title(f'COMPARATIE: Valori Reale vs Prezise\nMultiple MAE: ${maea_multiple:,.0f} | Simple MAE: ${maea_simple:,.0f}', 
                      fontsize=7, fontweight='bold')
axes[0, 1].tick_params(axis='both', which='major', labelsize=7)
axes[0, 1].legend(loc='upper left', fontsize=7)
axes[0, 1].grid(True, alpha=0.3)

# ---- SUBPLOT 3: Regresia Polinomiala grad 2 ----
# Cream predictii pentru vizualizare continua
Xa_poly_2_viz = polya_features_2.transform(Xa_test_simple)
ya_poly_2_viz = modela_poly_2.predict(Xa_poly_2_viz)

axes[1, 0].scatter(Xa_simple, ya, alpha=0.5, s=30, color='steelblue', edgecolor='darkblue', label='Date reale')
axes[1, 0].plot(Xa_test_simple, ya_poly_2_viz, color='purple', linewidth=2, label='Curba polinomiala (grad 2)', linestyle='-')
axes[1, 0].set_xlabel('Users', fontsize=7, fontweight='bold')
axes[1, 0].set_ylabel('Revenue ($)', fontsize=7, fontweight='bold')
axes[1, 0].set_title(f'REGRESIE POLINOMIALA (grad 2)\nR² = {r2a_poly_2:.4f} | MAE = ${maea_poly_2:,.0f} | MSE = {msea_poly_2:,.0f}', 
                      fontsize=7, fontweight='bold')
axes[1, 0].tick_params(axis='both', which='major', labelsize=7)
axes[1, 0].legend(loc='upper left', fontsize=7)
axes[1, 0].grid(True, alpha=0.3)

# ---- SUBPLOT 4: Importanta variabilelor (Coeficienti) ----
feature_names = ['Year', 'Month', 'Users', 'New Users', 'Sessions', 'Bounce Rate', 'Pageviews', 'Avg. Duration']
coefficients = modela_multiple.coef_
colors = ['green' if c > 0 else 'red' for c in coefficients]

axes[1, 1].barh(feature_names, coefficients, color=colors, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Coeficient (Impact asupra Revenue)', fontsize=7, fontweight='bold')
axes[1, 1].set_ylabel('Variabile', fontsize=7, fontweight='bold')
axes[1, 1].set_title(f'IMPORTANTA VARIABILELOR in Regresia Multipla\n(Coeficientii modelului)', 
                      fontsize=7, fontweight='bold')
axes[1, 1].tick_params(axis='both', which='major', labelsize=7)
axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=1)
axes[1, 1].grid(True, alpha=0.3, axis='x')

# Adaugam valorile coeficientilor pe bare
for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
    axes[1, 1].text(coef, i, f' {coef:.2f}', va='center', fontsize=7, fontweight='bold')

# Ajustam spatiile intre grafice pentru a evita suprapunerea textelor
plt.tight_layout(pad=3.0, h_pad=3.5, w_pad=3.5)
plt.savefig('analiza_regresie_revenue.png', dpi=300, bbox_inches='tight')
print("✓ Graficele au fost salvate in 'analiza_regresie_revenue.png'")
plt.show()


# ============================================================================
# A. TRAIN/TEST SPLIT PENTRU VALIDARE MAI ROBUSTA
# ============================================================================

print("\n" + "="*70)
print("A. TRAIN/TEST SPLIT - VALIDARE ROBUSTA")
print("="*70)

from sklearn.model_selection import train_test_split

# Impartim datele in 80% antrenare si 20% testare
X_train, X_test, y_train, y_test = train_test_split(Xa_multiple, ya, test_size=0.2, random_state=42)

print(f"Date totale: {len(Xa_multiple)}")
print(f"Date antrenare: {len(X_train)} ({len(X_train)/len(Xa_multiple)*100:.1f}%)")
print(f"Date testare: {len(X_test)} ({len(X_test)/len(Xa_multiple)*100:.1f}%)")

# Antrenam modelul doar pe datele de antrenare
model_train = LinearRegression()
model_train.fit(X_train, y_train)

# Facem predictii pe setul de TEST (date nevazute)
y_train_pred = model_train.predict(X_train)
y_test_pred = model_train.predict(X_test)

# Evaluam performanta pe ambele seturi
r2_train = r2_score(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)

r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print("\n--- Performanta pe SET de ANTRENARE ---")
print(f"R² = {r2_train:.4f}, MAE = ${mae_train:,.0f}, MSE = {mse_train:,.0f}")

print("\n--- Performanta pe SET de TESTARE (date nevazute) ---")
print(f"R² = {r2_test:.4f}, MAE = ${mae_test:,.0f}, MSE = {mse_test:,.0f}")

# Verificam overfitting
if abs(r2_train - r2_test) < 0.05:
    print("\n✓ Model echilibrat - diferenta mica intre train si test (< 5%)")
elif r2_train > r2_test + 0.1:
    print("\n⚠️  Posibil OVERFITTING - modelul se potriveste prea bine pe datele de antrenare")
else:
    print("\n✓ Model valid - performanta consistenta")


# ============================================================================
# B. ANALIZA DE IMPORTANTA - CARE VARIABILE CONTRIBUIE CEL MAI MULT
# ============================================================================

print("\n" + "="*70)
print("B. ANALIZA DE IMPORTANTA A VARIABILELOR")
print("="*70)

# Cream un DataFrame cu variabilele si coeficientii lor
importance_df = pd.DataFrame({
    'Variabila': feature_names,
    'Coeficient': modela_multiple.coef_,
    'Coeficient_Abs': np.abs(modela_multiple.coef_)
}).sort_values('Coeficient_Abs', ascending=False)

print("\nVariabilele ordonate dupa IMPORTANTA (valoare absoluta a coeficientului):")
print("-" * 70)
for idx, row in importance_df.iterrows():
    impact = "POZITIV (+)" if row['Coeficient'] > 0 else "NEGATIV (-)"
    print(f"{row['Variabila']:<20} | Coef: {row['Coeficient']:>10.4f} | Impact: {impact}")

print("\n" + "="*70)
print("INTERPRETARE:")
print("="*70)
top_var = importance_df.iloc[0]
print(f"✓ Cea mai IMPORTANTA variabila: {top_var['Variabila']}")
print(f"  - Coeficient: {top_var['Coeficient']:.4f}")
if top_var['Coeficient'] > 0:
    print(f"  - Cu cat creste {top_var['Variabila']}, cu atat creste Revenue-ul")
else:
    print(f"  - Cu cat creste {top_var['Variabila']}, cu atat scade Revenue-ul")

print(f"\n✓ Top 3 variabile cele mai importante:")
for i, row in importance_df.head(3).iterrows():
    print(f"  {row['Variabila']}: coeficient {row['Coeficient']:.4f}")


# ============================================================================
# C. PREDICTII PRACTICE - "Daca am X users si Y sessions, care va fi Revenue-ul?"
# ============================================================================

print("\n" + "="*70)
print("C. PREDICTII PRACTICE")
print("="*70)

# Exemplu 1: Scenariul mediu (pe baza mediilor din date)
print("\n--- Exemplu 1: Scenariul MEDIU (bazat pe medii) ---")
avg_values = data_second[['Year', 'Month of the year', 'Users', 'New Users', 'Sessions', 
                           'Bounce Rate', 'Pageviews', 'Avg. Session Duration']].mean()
print("\nValori medii:")
for col, val in avg_values.items():
    if col == 'Bounce Rate':
        print(f"  {col}: {val:.2%}")
    elif col in ['Year', 'Month of the year']:
        print(f"  {col}: {val:.0f}")
    else:
        print(f"  {col}: {val:,.0f}")

avg_input = avg_values.values.reshape(1, -1)
predicted_revenue_avg = modela_multiple.predict(avg_input)[0]
print(f"\n→ REVENUE PREZIS: ${predicted_revenue_avg:,.2f}")

# Exemplu 2: Scenariul optimist (valori crescute cu 30%)
print("\n--- Exemplu 2: Scenariul OPTIMIST (+30% la metrici cheie) ---")
optimistic_values = avg_values.copy()
optimistic_values['Users'] *= 1.3
optimistic_values['New Users'] *= 1.3
optimistic_values['Sessions'] *= 1.3
optimistic_values['Pageviews'] *= 1.3
optimistic_values['Bounce Rate'] *= 0.8  # Scadem bounce rate

print("\nValori optimiste:")
for col, val in optimistic_values.items():
    if col == 'Bounce Rate':
        print(f"  {col}: {val:.2%}")
    elif col in ['Year', 'Month of the year']:
        print(f"  {col}: {val:.0f}")
    else:
        print(f"  {col}: {val:,.0f}")

optimistic_input = optimistic_values.values.reshape(1, -1)
predicted_revenue_opt = modela_multiple.predict(optimistic_input)[0]
print(f"\n→ REVENUE PREZIS: ${predicted_revenue_opt:,.2f}")
print(f"→ CRESTERE fata de mediu: ${predicted_revenue_opt - predicted_revenue_avg:,.2f} ({(predicted_revenue_opt/predicted_revenue_avg - 1)*100:.1f}%)")

# Exemplu 3: Scenariul personalizat
print("\n--- Exemplu 3: Scenariul PERSONALIZAT ---")
print("Introduceti valorile pentru predictie personalizata:")
print("(Apasati Enter pentru a folosi valoarea medie)")

def get_input_or_default(prompt, default_val, is_percentage=False):
    user_input = input(f"{prompt} (default: {default_val:,.2f}): ").strip()
    if user_input == "":
        return default_val
    try:
        val = float(user_input)
        if is_percentage and val > 1:
            val = val / 100  # Convertim din % in fractie
        return val
    except:
        return default_val

custom_values = {}
for col in feature_names:
    orig_col = ['Year', 'Month of the year', 'Users', 'New Users', 'Sessions', 
                'Bounce Rate', 'Pageviews', 'Avg. Session Duration'][feature_names.index(col)]
    default = avg_values[orig_col]
    is_pct = (orig_col == 'Bounce Rate')
    custom_values[col] = get_input_or_default(f"  {col}", default, is_pct)

custom_input = np.array([custom_values[name] for name in feature_names]).reshape(1, -1)
predicted_revenue_custom = modela_multiple.predict(custom_input)[0]
print(f"\n→ REVENUE PREZIS (personalizat): ${predicted_revenue_custom:,.2f}")

print("\n" + "="*70)
print("ANALIZA COMPLETA FINALIZATA!")
print("="*70)
print(f"✓ Cel mai bun model: {besta_model} (R² = {besta_r2:.4f})")
print(f"✓ Performanta pe date de test: R² = {r2_test:.4f}, MAE = ${mae_test:,.0f}")
print(f"✓ Variabila cea mai importanta: {importance_df.iloc[0]['Variabila']}")
print(f"✓ Grafice salvate in: analiza_regresie_revenue.png")
print("="*70)
