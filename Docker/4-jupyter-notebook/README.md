# 📓 Varianta 4: Jupyter Notebook Containerizat

## 🎯 Ce face?

Rulează Jupyter Lab în container cu:
- ✅ Python 3.13.9 și toate dependențele ML
- ✅ Acces complet la proiect (read/write)
- ✅ Interfață Jupyter Lab în browser
- ✅ Fără autentificare (pentru demo local)
- ✅ Persistență notebook-uri

**Comisia poate rula notebook-ul fără nicio instalare!**

---

## 🚀 Cum să pornești

### **Pas 1: Deschide acest folder în terminal**
```powershell
cd C:\Users\User\Desktop\repos\Project_repo\docker\4-jupyter-notebook
```

### **Pas 2: Build containerul**
```powershell
docker-compose build
```

### **Pas 3: Pornește Jupyter Lab**
```powershell
docker-compose up
```

**Așteptă să vezi:**
```
jupyter-ml-notebook | [I] Jupyter Server is running at:
jupyter-ml-notebook | [I] http://127.0.0.1:8888/lab
```

### **Pas 4: Accesează Jupyter Lab**
Deschide browser la: **http://localhost:8888/lab**

**Nu e nevoie de token sau parolă!**

---

## 📂 Navigare în Jupyter Lab

După ce deschizi http://localhost:8888/lab:

### **1. Explorer (stânga)**
Vei vedea structura completă a proiectului:
```
/workspace/
├── notebooks/
│   └── MLflow_ml_model_web_traffic.ipynb  ← Notebook-ul tău
├── sources/
│   └── mlflow_ml_model_web_traffic.py
├── data/
│   ├── raw/
│   │   └── ga-sessions.csv
│   ├── processed/
│   │   ├── X_simple.csv
│   │   └── X_enhanced.csv
│   └── predictions/
├── models/
├── visualizations/
└── ...
```

### **2. Deschide notebook-ul**
- Click pe `notebooks/`
- Click pe `MLflow_ml_model_web_traffic.ipynb`
- Notebook-ul se deschide în tab nou

### **3. Rulează celule**
- **O celulă**: `Shift + Enter`
- **Toate celulele**: Menu → Run → Run All Cells
- **Restart kernel**: Menu → Kernel → Restart Kernel

---

## 🧪 Testare Rapidă

### **Test 1: Verifică Python și pachete**
Creează un nou notebook:
1. Click "+" în Jupyter Lab
2. Selectează "Python 3 (ipykernel)"
3. Scrie în prima celulă:
```python
import sys
import pandas as pd
import numpy as np
import sklearn
import mlflow

print(f"Python: {sys.version}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"MLflow: {mlflow.__version__}")
```
4. `Shift + Enter` pentru rulare

**Output așteptat:**
```
Python: 3.13.9
Pandas: 2.3.3
NumPy: 2.4.6
Scikit-learn: 1.8.0
MLflow: 3.12.0
```

### **Test 2: Încarcă date**
Celulă nouă:
```python
import pandas as pd

df = pd.read_csv("data/raw/ga-sessions.csv")
print(f"Date încărcate: {len(df)} rânduri, {len(df.columns)} coloane")
print(df.head())

# Seturi procesate (generate de scripturile de antrenare):
# pd.read_csv("data/processed/X_enhanced.csv")
# pd.read_csv("data/processed/X_simple.csv")
```

### **Test 3: Rulează notebook-ul complet**
Deschide `notebooks/MLflow_ml_model_web_traffic.ipynb` și:
- Menu → Run → Run All Cells
- Așteaptă ~30-60 secunde
- Verifică că toate celulele se execută fără erori

---

## 🔧 Comenzi Utile

### **Pornește în background**
```powershell
docker-compose up -d
```

### **Vezi logs**
```powershell
docker-compose logs -f
```

### **Oprește Jupyter**
```powershell
docker-compose down
```

### **Restart Jupyter**
```powershell
docker-compose restart
```

---

## 🎓 Pentru Comisie

### **Demonstrație Jupyter Containerizat:**

**Scenariul 1: Zero Setup Demo**
```
Comisie: "Vreau să rulez notebook-ul"
Tu: "docker-compose up"
Comisie: "Gata, deschid browser la localhost:8888"
Tu: "Notebook-ul e în folders/notebooks/, click și Run All"
```

**Scenariul 2: Cross-Platform Demo**
```
Comisie (Windows): "docker-compose up" → funcționează
Comisie (Mac): "docker-compose up" → funcționează identic
Comisie (Linux): "docker-compose up" → funcționează identic
```

**Scenariul 3: Modificări Live**
```
1. Deschide notebook în Jupyter Lab (browser)
2. Modifică parametri (ex: test_size=0.3 în loc de 0.2)
3. Rulează celulele
4. Vezi rezultate noi instant
5. Salvează → modificările se păstrează pe disk local
```

### **Avantaje:**
- ✅ **Zero installation**: Nu trebuie instalat Python, Jupyter, pachete
- ✅ **Izolat**: Nu afectează mediul local
- ✅ **Portabil**: Rulează pe orice sistem cu Docker
- ✅ **Interactiv**: Modifici și experimentezi în timp real
- ✅ **Persistent**: Modificările se salvează local

---

## 📊 Features Jupyter Lab

### **1. Code Completion**
- Scrie `df.` → apasă `Tab` → vezi metodele disponibile

### **2. Inline Help**
- `Shift + Tab` pe o funcție → vezi documentația

### **3. Magic Commands**
```python
%matplotlib inline  # Grafice în notebook
%timeit  # Măsoară timp execuție
%%time  # Măsoară timp celulă
!ls  # Rulează comenzi shell
```

### **4. Export**
- File → Export Notebook As → HTML/PDF/Markdown

---

## 🔐 Securitate

**⚠️ ATENȚIE: Configurația curentă NU are parolă!**

Pentru demo local e OK, dar pentru producție:

### **Adaugă parolă:**
```powershell
# Generează hash parolă
docker-compose run jupyter python -c "from jupyter_server.auth import passwd; print(passwd('your-password'))"

# Copiază hash-ul și editează docker-compose.yml:
CMD ["jupyter", "lab", "--NotebookApp.password='sha1:...'"]
```

---

## 📂 Salvare Notebook-uri

### **Notebook-urile sunt salvate pe disk local!**

Când modifici un notebook în Jupyter Lab (browser), schimbările se salvează automat în:
```
C:\Users\User\Desktop\repos\Project_repo\notebooks\
```

**Poți vedea modificările în VS Code simultan!**

---

## 🧪 Creare Notebook-uri Noi

### **În Jupyter Lab:**
1. Click "+" în toolbar
2. Selectează "Python 3 (ipykernel)"
3. Scrie cod în celule
4. `Shift + Enter` pentru rulare
5. File → Save → Notebook salvat în `/workspace/notebooks/`

---

## 🔄 Sincronizare cu VS Code

### **Workflow hibrid:**

**Opțiunea 1: Dezvoltare în VS Code, Rulare în Jupyter**
1. Editează notebook în VS Code
2. Salvează (`Ctrl + S`)
3. Refresh browser (F5)
4. Rulează celule în Jupyter Lab

**Opțiunea 2: Dezvoltare în Jupyter, Vizualizare în VS Code**
1. Editează în Jupyter Lab (browser)
2. Salvează (`Ctrl + S`)
3. VS Code detectează schimbarea automat
4. Vezi diff-ul în Source Control

---

## ❓ Troubleshooting

### **Eroare: "port 8888 already in use"**
```powershell
# Schimbă portul în docker-compose.yml
ports:
  - "8889:8888"  # Acum accesezi la localhost:8889
```

### **Nu văd proiectul în Jupyter Lab**
- Verifică că ești în folder-ul `/workspace/` în Jupyter
- Click pe iconița folder (stânga sus) pentru root

### **Kernel nu pornește**
```powershell
# Restart container
docker-compose restart
```

### **Modificările nu se salvează**
- Verifică volumes în docker-compose.yml
- Asigură-te că ai permisiuni write pe folder

### **Graficele nu apar**
Adaugă la începutul notebook-ului:
```python
%matplotlib inline
import matplotlib.pyplot as plt
```

---

## 📦 Extensions Jupyter Lab

### **Instalare extensions (opțional):**

```powershell
# Intră în container
docker-compose exec jupyter bash

# Instalează extension
pip install jupyterlab-git
jupyter lab build

# Restart container
docker-compose restart
```

**Extensions utile:**
- `jupyterlab-git` - Git integration
- `jupyterlab-lsp` - Language Server Protocol (autocomplete avansat)
- `jupyterlab-plotly` - Grafice interactive

---

## 🚀 Alternative: JupyterHub (Multi-user)

Pentru prezentare cu mai mulți utilizatori:

```yaml
# docker-compose-hub.yml
services:
  jupyterhub:
    image: jupyterhub/jupyterhub
    ports:
      - "8000:8000"
    # ... configurație multi-user
```

---

**✅ Jupyter Lab containerizat gata de folosit!**

**Link rapid:** http://localhost:8888/lab  
**Notebook-ul tău:** `notebooks/MLflow_ml_model_web_traffic.ipynb`
