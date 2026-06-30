# 🐳 Docker Containers pentru Proiectul Web Traffic ML

Acest folder conține 4 configurații Docker pentru diferite cazuri de utilizare ale proiectului.

---

## 📁 Structura Folderului

```
docker/
├── 1-mlflow-tracking/        → MLflow Tracking Server
├── 2-reproducible-env/       → Mediu reproducibil complet
├── 3-model-api/              → API REST pentru predicții
├── 4-jupyter-notebook/       → Jupyter Notebook containerizat
└── README_DOCKER.md          → Acest fișier
```

---

## 🚀 Ghid Rapid

### **Varianta 1: MLflow Tracking Server**
📍 Folder: `1-mlflow-tracking/`  
🎯 Scop: Server de tracking pentru experimente ML  
⏱️ Setup: 2 minute

### **Varianta 2: Mediu Reproducibil**
📍 Folder: `2-reproducible-env/`  
🎯 Scop: Rulare training script în container izolat  
⏱️ Setup: 3 minute

### **Varianta 3: Model API**
📍 Folder: `3-model-api/`  
🎯 Scop: API REST pentru predicții în timp real  
⏱️ Setup: 3 minute

### **Varianta 4: Jupyter Notebook**
📍 Folder: `4-jupyter-notebook/`  
🎯 Scop: Notebook interactiv în browser  
⏱️ Setup: 2 minute

---

## 📋 Prerequisite

### **1. Instalează Docker Desktop**
- Windows: https://docs.docker.com/desktop/install/windows-install/
- Verificare: `docker --version` și `docker-compose --version`

### **2. Asigură-te că Docker rulează**
```powershell
docker ps
# Ar trebui să returneze lista containerelor (chiar dacă e goală)
```

### **3. Structură date proiect**
```
data/raw/ga-sessions.csv           # Dataset brut
data/processed/X_simple.csv        # Features simple (generate la antrenare)
data/processed/X_enhanced.csv      # Features enhanced (generate la antrenare)
data/predictions/                  # Rezultate validare și predicții
```

---

## 🎓 Comenzi Docker Utile

```powershell
# Verifică versiunea
docker --version
docker-compose --version

# Pornește containers
docker-compose up

# Pornește în background
docker-compose up -d

# Oprește containers
docker-compose down

# Vezi containers active
docker ps

# Vezi logs
docker-compose logs -f

# Rebuild containers
docker-compose build --no-cache
```

---

## 📖 Cum să folosești fiecare variantă

Fiecare folder conține propriul **README.md** cu instrucțiuni detaliate.

**Pașii generali:**
1. Deschide folderul variantei dorite în terminal
2. Citește README.md din acel folder
3. Rulează `docker-compose up`
4. Accesează aplicația în browser

---

## ❓ Întrebări Frecvente

**Q: Trebuie să instalez Python, pandas, etc?**  
A: NU! Docker instalează tot automat în container.

**Q: Pot rula pe Mac/Linux?**  
A: DA! Docker funcționează identic pe toate platformele.

**Q: Cum opresc containerele?**  
A: Apasă `Ctrl+C` în terminal sau rulează `docker-compose down`.

**Q: Datele se pierd când opresc containerul?**  
A: NU! Folosim volumes pentru persistență.

---

## 🆘 Suport

Dacă întâmpini probleme:
1. Verifică că Docker Desktop rulează
2. Citește README.md din folderul specific
3. Verifică logs: `docker-compose logs`
4. Restart: `docker-compose down` apoi `docker-compose up`

---

**Creat pentru prezentare comisie licență 2026** 🎓
