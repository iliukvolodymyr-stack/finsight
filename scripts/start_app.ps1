# FinSight AI - Desktop Application Launcher
$Host.UI.RawUI.WindowTitle = "FinSight AI - Runner"

function Stop-PortProcess([int]$port) {
    try {
        $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if ($connections) {
            $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
            foreach ($p in $pids) {
                if ($p -gt 0 -and $p -ne $PID) {
                    Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
                }
            }
        }
    } catch {}
}

Clear-Host
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "                FinSight AI Platform                      " -ForegroundColor Yellow
Write-Host "   Кросплатформна інтелектуальна фінансова система        " -ForegroundColor DarkCyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "D:\fin_tech"

# 1. Ensure TimescaleDB + Redis containers are running (if Docker is active)
Write-Host "[1/4] Перевірка Docker (TimescaleDB + Redis)..." -ForegroundColor Green
try {
    docker info *> $null
    if ($LASTEXITCODE -eq 0) {
        docker compose -f infra/compose/docker-compose.dev.yml up -d --wait *> $null
    } else {
        Write-Host "  [!] Docker не запущений — API може не бачити БД." -ForegroundColor Yellow
    }
} catch {
    Write-Host "  [!] Docker не знайдено." -ForegroundColor Yellow
}

# 2. Clean existing lingering ports if any
Stop-PortProcess 8000
Stop-PortProcess 5173

# 3. Start Backend API Server
Write-Host "[2/4] Запуск FastAPI бекенду (порт 8000)..." -ForegroundColor Green
$apiProcess = Start-Process -FilePath "py" `
    -ArgumentList "-3 -m uv run uvicorn --app-dir services/api app.main:app --host 127.0.0.1 --port 8000" `
    -WorkingDirectory "D:\fin_tech" `
    -WindowStyle Hidden `
    -PassThru

# 4. Start Desktop Client Dev Server
Write-Host "[3/4] Запуск десктопного веб-клієнту (порт 5173)..." -ForegroundColor Green
$webProcess = Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c pnpm --filter desktop dev" `
    -WorkingDirectory "D:\fin_tech" `
    -WindowStyle Hidden `
    -PassThru

# 5. Wait for server readiness
Write-Host "[4/4] Очікування ініціалізації сервісів..." -ForegroundColor Green
Start-Sleep -Seconds 3

# 5. Open browser
Write-Host "[+] Відкриття браузера..." -ForegroundColor Cyan
Start-Process "http://localhost:5173"

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " ✓ FinSight AI успішно працює!" -ForegroundColor Green
Write-Host "   - Веб-інтерфейс:     http://localhost:5173" -ForegroundColor White
Write-Host "   - Документація API:  http://localhost:8000/docs" -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Натисніть клавішу [Enter] або закрийте це вікно, щоб зупинити сервери." -ForegroundColor Yellow
Write-Host ""

# Wait for user input to terminate
[void][System.Console]::ReadLine()

Write-Host "Зупинка сервісів FinSight AI..." -ForegroundColor Magenta

if ($apiProcess -and -not $apiProcess.HasExited) {
    cmd.exe /c "taskkill /PID $($apiProcess.Id) /T /F" 2>$null
}
if ($webProcess -and -not $webProcess.HasExited) {
    cmd.exe /c "taskkill /PID $($webProcess.Id) /T /F" 2>$null
}

Stop-PortProcess 8000
Stop-PortProcess 5173

Write-Host "Всі сервери успішно зупинено." -ForegroundColor Green
Start-Sleep -Seconds 1
