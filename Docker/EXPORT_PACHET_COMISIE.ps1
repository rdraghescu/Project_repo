# ========================================
# SCRIPT EXPORT COMPLET - Pachet Comisie
# ========================================
# Acest script automatizează crearea pachetului complet pentru comisie
# Include: imagine Docker, configurații, instrucțiuni

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "EXPORT PACHET COMISIE - MLflow Docker" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Configurare paths
$PROJECT_ROOT = "C:\Users\User\Desktop\repos\Project_repo"
$DOCKER_DIR = "$PROJECT_ROOT\docker\1-mlflow-tracking"
$EXPORT_DIR = "C:\Users\User\Desktop\PACHET_COMISIE"
$IMAGE_NAME = "1-mlflow-tracking-mlflow:latest"

# ========================================
# PAS 1: Verificare Docker
# ========================================
Write-Host "[1/6] Verificare Docker instalat..." -ForegroundColor Yellow

try {
    $dockerVersion = docker --version
    Write-Host "   Docker gasit: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "   ERROR: Docker nu este instalat sau nu este in PATH!" -ForegroundColor Red
    Write-Host "   Instaleaza Docker Desktop: https://www.docker.com/products/docker-desktop/" -ForegroundColor Red
    exit 1
}

# ========================================
# PAS 2: Verificare Imagine Există
# ========================================
Write-Host "`n[2/6] Verificare imagine Docker exista..." -ForegroundColor Yellow

$imageExists = docker images --format "{{.Repository}}:{{.Tag}}" | Select-String -Pattern "^1-mlflow-tracking-mlflow:latest$"

if (-not $imageExists) {
    Write-Host "   Imaginea nu exista! Construiesc imaginea..." -ForegroundColor Yellow
    
    Set-Location $DOCKER_DIR
    docker-compose build
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "   ERROR: Build-ul imaginii a esuat!" -ForegroundColor Red
        exit 1
    }
    Write-Host "   Imagine construita cu succes!" -ForegroundColor Green
} else {
    Write-Host "   Imagine gasita: $IMAGE_NAME" -ForegroundColor Green
}

# ========================================
# PAS 3: Creare Director Export
# ========================================
Write-Host "`n[3/6] Creare director export..." -ForegroundColor Yellow

if (Test-Path $EXPORT_DIR) {
    Write-Host "   Director deja exista: $EXPORT_DIR" -ForegroundColor Yellow
    $response = Read-Host "   Stergi continutul existent? (y/n)"
    if ($response -eq 'y') {
        Remove-Item "$EXPORT_DIR\*" -Recurse -Force
        Write-Host "   Continut sters!" -ForegroundColor Green
    }
} else {
    New-Item -ItemType Directory -Path $EXPORT_DIR -Force | Out-Null
    Write-Host "   Director creat: $EXPORT_DIR" -ForegroundColor Green
}

# ========================================
# PAS 4: Export Imagine Docker
# ========================================
Write-Host "`n[4/6] Export imagine Docker..." -ForegroundColor Yellow
Write-Host "   Acest pas poate dura 1-2 minute..." -ForegroundColor Gray

$imagePath = "$EXPORT_DIR\mlflow-tracking-image.tar"

docker save -o $imagePath $IMAGE_NAME

if ($LASTEXITCODE -eq 0) {
    $imageSize = (Get-Item $imagePath).Length / 1MB
    Write-Host "   Imagine exportata: mlflow-tracking-image.tar" -ForegroundColor Green
    Write-Host "   Dimensiune: $([math]::Round($imageSize, 2)) MB" -ForegroundColor Green
} else {
    Write-Host "   ERROR: Export-ul imaginii a esuat!" -ForegroundColor Red
    exit 1
}

# ========================================
# PAS 5: Copiere Fisiere Configurare
# ========================================
Write-Host "`n[5/6] Copiere fisiere configurare..." -ForegroundColor Yellow

# docker-compose.yml
Copy-Item "$DOCKER_DIR\docker-compose.yml" "$EXPORT_DIR\" -Force
Write-Host "   docker-compose.yml copiat" -ForegroundColor Green

# Dockerfile
Copy-Item "$DOCKER_DIR\Dockerfile" "$EXPORT_DIR\" -Force
Write-Host "   Dockerfile copiat" -ForegroundColor Green

# README
Copy-Item "$DOCKER_DIR\README.md" "$EXPORT_DIR\README_ORIGINAL.md" -Force
Write-Host "   README copiat" -ForegroundColor Green

# ========================================
# PAS 6: Creare Arhiva ZIP (opțional)
# ========================================
Write-Host "`n[6/6] Creare arhiva ZIP..." -ForegroundColor Yellow

$zipPath = "C:\Users\User\Desktop\PACHET_COMISIE_MLflow_Docker.zip"

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Write-Host "   Compresie in curs... (poate dura 2-3 minute)" -ForegroundColor Gray

try {
    Compress-Archive -Path "$EXPORT_DIR\*" -DestinationPath $zipPath -CompressionLevel Optimal
    $zipSize = (Get-Item $zipPath).Length / 1MB
    Write-Host "   Arhiva creata: PACHET_COMISIE_MLflow_Docker.zip" -ForegroundColor Green
    Write-Host "   Dimensiune: $([math]::Round($zipSize, 2)) MB" -ForegroundColor Green
} catch {
    Write-Host "   ATENTIE: Compresia a esuat! Foloseste folderul necomprimat." -ForegroundColor Yellow
}

# ========================================
# FINALIZARE
# ========================================
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "EXPORT COMPLET!" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Locatie fisiere:" -ForegroundColor White
Write-Host "   Folder: $EXPORT_DIR" -ForegroundColor Yellow
if (Test-Path $zipPath) {
    Write-Host "   Arhiva: $zipPath" -ForegroundColor Yellow
}

Write-Host "`nContinut pachet:" -ForegroundColor White
Write-Host "   - mlflow-tracking-image.tar (~500-600MB)" -ForegroundColor Gray
Write-Host "   - docker-compose.yml" -ForegroundColor Gray
Write-Host "   - Dockerfile" -ForegroundColor Gray
Write-Host "   - README_ORIGINAL.md" -ForegroundColor Gray
Write-Host "   - INSTRUCTIUNI_RULARE.md" -ForegroundColor Gray

Write-Host "`nUrmatorii pasi:" -ForegroundColor White
Write-Host "   1. Copiaza folderul '$EXPORT_DIR' pe USB/cloud" -ForegroundColor Green
Write-Host "      SAU" -ForegroundColor Yellow
Write-Host "   2. Trimite arhiva ZIP (daca e creata)" -ForegroundColor Green
Write-Host "   3. Comisia va urma instructiunile din INSTRUCTIUNI_RULARE.md" -ForegroundColor Green

Write-Host "`n========================================`n" -ForegroundColor Cyan

# Deschide folder
$response = Read-Host "Deschide folderul export? (y/n)"
if ($response -eq 'y') {
    explorer $EXPORT_DIR
}
