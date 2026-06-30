# 🔧 SOLUȚIE: Artifacts Nu Apar în MLflow UI Docker

## ❌ PROBLEMA:

Când rulezi scriptul Python și loghezi artifacts la serverul MLflow Docker, tab-ul "Artifacts" din UI afișează:
```
No Artifacts Recorded
Use the log artifact APIs to store file outputs from MLflow runs.
```

**Cauza**: Serverul Docker MLflow nu poate accesa path-urile Windows absolute (`C:\Users\...`) unde scriptul Python creează fișierele artifacts.

---

## ✅ SOLUȚIA 1: Folosește MLflow LOCAL (fără server Docker)

Aceasta este cea mai simplă soluție pentru **dezvoltare și prezentare**.

### **Cum funcționează:**
- Scriptul salvează **toate datele local** în folderul `mlruns/` din proiect
- Vizualizarea se face prin **MLflow UI local** (nu Docker)
- Artifacts sunt 100% accesibile și vizibile

### **Pași:**

**1. Modifică scriptul să NU folosească server Docker**

Șterge sau comentează această linie din `sources/mlflow_ml_model_web_traffic.py`:
```python
# mlflow.set_tracking_uri("http://localhost:5000")  # <-- COMENTEAZĂ aceasta
```

**2. Oprește serverul Docker (dacă rulează)**
```powershell
cd docker\1-mlflow-tracking
docker-compose down
```

**3. Rulează scriptul normal**
```powershell
cd C:\Users\User\Desktop\repos\Project_repo
python sources\mlflow_ml_model_web_traffic.py
```

**4. Pornește MLflow UI LOCAL**
```powershell
mlflow ui --port 5000
```

**5. Accesează UI-ul**
- Deschide: http://localhost:5000
- Toate artifacts vor fi vizibile! ✅

---

## ✅ SOLUȚIA 2: Configurare Avansată cu Docker (pentru producție)

Această soluție necesită modificări mai complexe dar permite tracking centralizat real.

### **Modificări necesare:**

**1. Modifică scriptul să salveze artifacts în `mlruns/`**

Înlocuiește path-urile pentru salvare artifacts:
```python
# ÎNAINTE (nu funcționează cu Docker):
viz_dir = os.path.join(PROJECT_ROOT, 'visualizations')
target_dist_path = os.path.join(viz_dir, '01_target_distribution.png')

# DUPĂ (funcționează cu Docker):
run_id = mlflow.active_run().info.run_id
artifact_dir = os.path.join(PROJECT_ROOT, 'mlruns', '1', run_id, 'artifacts', 'visualizations')
os.makedirs(artifact_dir, exist_ok=True)
target_dist_path = os.path.join(artifact_dir, '01_target_distribution.png')
```

**2. Modifică `docker-compose.yml`**

```yaml
command: mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:////workspace/mlruns/mlflow.db \
  --default-artifact-root /workspace/mlruns \
  --serve-artifacts \
  --allowed-hosts '*' --cors-allowed-origins '*'
```

**3. Restart server Docker**
```powershell
cd docker\1-mlflow-tracking
docker-compose down
docker-compose up -d
```

---

## 🎓 PENTRU PREZENTARE LA COMISIE - RECOMANDARE:

**Folosește SOLUȚIA 1 (MLflow LOCAL)** pentru demonstrație:

### **Avantaje:**
- ✅ **Simplu**: Fără configurări complexe Docker
- ✅ **Funcționează garantat**: Toate artifacts vizibile
- ✅ **Rapid**: Nu necesită restart containere
- ✅ **Professional**: Arată workflow real de dezvoltare ML

### **Demonstrație:**
1. **Oprește Docker** (dacă rulează):
   ```powershell
   cd docker\1-mlflow-tracking
   docker-compose down
   ```

2. **Modifică scriptul** - comentează linia tracking URI:
   ```python
   # mlflow.set_tracking_uri("http://localhost:5000")
   ```

3. **Rulează training**:
   ```powershell
   python sources\mlflow_ml_model_web_traffic.py
   ```

4. **Pornește UI local**:
   ```powershell
   mlflow ui --port 5000
   ```

5. **Arată comisiei**:
   - Parametri: 20 features, test_size=0.2
   - Metrici: R²=0.4167, MAE=195.03, RMSE=365.33
   - **Artifacts**: 8 grafice PNG + 3 modele pickle + metadata JSON ✅

### **Explică comisiei:**

> **"Am configurat MLflow pentru tracking local, care salvează toate experimentele în folderul `mlruns/`. 
> Acest setup este ideal pentru dezvoltare și permite accesibilitate completă la toate artifacts-urile modelului.
> Pentru producție, acest setup poate fi migrat ușor la un server MLflow centralizat cu storage cloud (S3, Azure Blob, etc.)."**

---

## 📊 COMPARAȚIE SOLUȚII:

| Aspect | Soluție 1: Local | Soluție 2: Docker |
|--------|------------------|-------------------|
| **Complexitate** | Simplă | Complexă |
| **Artifacts vizibile** | ✅ DA | ⚠️ Necesită config |
| **Setup time** | < 1 minut | 10-15 minute |
| **Pentru prezentare** | ✅ Recomandat | ❌ Risc tehnic |
| **Pentru producție** | ❌ Nu scalează | ✅ Recomandat |

---

## 🚀 QUICK FIX - Acum, pentru prezentare:

**1. Oprește Docker**
```powershell
cd C:\Users\User\Desktop\repos\Project_repo\docker\1-mlflow-tracking
docker-compose down
```

**2. Modifică scriptul**

Deschide: `sources\mlflow_ml_model_web_traffic.py`

Găsește linia 58:
```python
mlflow.set_tracking_uri("http://localhost:5000")
```

Comentează-o:
```python
# mlflow.set_tracking_uri("http://localhost:5000")  # DEZACTIVAT pentru tracking local
```

**3. Rulează din nou**
```powershell
cd C:\Users\User\Desktop\repos\Project_repo
python sources\mlflow_ml_model_web_traffic.py
```

**4. Vezi artifacts în UI**
```powershell
mlflow ui --port 5000
```

Deschide: http://localhost:5000 → Experiments → Web_Traffic_Prediction → Ultimul run → **Artifacts** ✅

---

## 🔍 VERIFICARE RAPIDĂ:

Dacă ai deja run-uri locale în `mlruns/`, poți verifica imediat:
```powershell
dir mlruns\1\*\artifacts
```

Ar trebui să vezi foldere cu PNG-uri și fișiere pickle! ✅

---

**✅ REZUMAT**: Pentru prezentare, folosește MLflow LOCAL (Soluția 1) pentru a evita problemele de artifact storage cu Docker. Este mai simplu, mai rapid și garantează că toate artifacts sunt vizibile comisiei!
