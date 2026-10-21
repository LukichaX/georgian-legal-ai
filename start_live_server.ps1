Write-Host "=== Georgian Legal AI v7.0 - Production Live Server (Hot-Swap Enabled) ===" -ForegroundColor Cyan

# 1. Cloudflare Tunnel binary check
$cfExe = "cloudflared.exe"
if (-not (Test-Path $cfExe)) {
    $sysCf = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
    $sysCf64 = "C:\Program Files\cloudflared\cloudflared.exe"
    if (Test-Path $sysCf) {
        $cfExe = $sysCf
    } elseif (Test-Path $sysCf64) {
        $cfExe = $sysCf64
    } else {
        Write-Host "[*] Downloading cloudflared.exe into project folder..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile "cloudflared.exe"
        $cfExe = ".\cloudflared.exe"
    }
}

# 2. Build Vite React frontend (if present)
if (Test-Path "frontend\package.json") {
    Write-Host "[1/3] Building Vite React frontend..." -ForegroundColor Green
    Push-Location frontend
    cmd.exe /c npm run build
    Pop-Location
} else {
    Write-Host "[1/3] Frontend folder not built yet - running API server mode..." -ForegroundColor Yellow
}

# 3. Start Cloudflare Tunnel pointing to FastAPI port 8000
Write-Host "[2/3] Starting Cloudflare Tunnel (port 8000) in a new window..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$cfExe' tunnel --url http://localhost:8000"
Start-Sleep -Seconds 2

# 4. Start Unified FastAPI Server (with internal Zero-Downtime Hot-Swap)
Write-Host "[3/3] Starting FastAPI Server (src/server.py) on http://0.0.0.0:8000 ..." -ForegroundColor Green
$pythonExe = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

while ($true) {
    & $pythonExe -m uvicorn src.server:app --host 0.0.0.0 --port 8000
    Write-Host "[!] Server stopped. Restarting in 3 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
}