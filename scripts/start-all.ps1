# Arranca MLflow, el backend y el frontend, cada uno en su propia ventana.
#   powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1
#   ... -WithDemoMcp   para levantar tambien el servidor MCP de ejemplo

param(
    [switch]$WithDemoMcp,
    [int]$BackendPort = 8090,
    [int]$MlflowPort = 5000,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root "backend\.venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "No existe el entorno virtual. Ejecuta antes: scripts\setup.ps1"
}

function Start-Panel([string]$Title, [string]$WorkDir, [string]$Command) {
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "`$host.UI.RawUI.WindowTitle = '$Title'; Set-Location '$WorkDir'; $Command"
    ) | Out-Null
    Write-Host "  $Title" -ForegroundColor Green
}

Write-Host "Arrancando servicios..." -ForegroundColor Cyan

Start-Panel "MLflow" $root @"
& '$python' -m mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db ``
    --artifacts-destination ./mlflow/artifacts --host 0.0.0.0 --port $MlflowPort
"@

# El backend espera a que MLflow responda; si no lo hace arranca igualmente
# (MLFLOW_FAIL_SOFT) pero sin trazabilidad.
Start-Sleep -Seconds 8

Start-Panel "Backend" (Join-Path $root "backend") @"
& '$python' -m uvicorn app.main:app --host 0.0.0.0 --port $BackendPort
"@

Start-Panel "Frontend" (Join-Path $root "frontend") "npm run dev -- --port $FrontendPort"

if ($WithDemoMcp) {
    Start-Panel "MCP demo" $root "& '$python' scripts\demo_mcp_server.py --port 3333"
}

$ip = (Get-NetIPAddress -AddressFamily IPv4 |
       Where-Object { $_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -ne "WellKnown" } |
       Select-Object -First 1).IPAddress

Write-Host "`nURLs:" -ForegroundColor Cyan
Write-Host "  UI          http://localhost:$FrontendPort"
Write-Host "  Backend     http://localhost:$BackendPort/docs"
Write-Host "  MLflow      http://localhost:$MlflowPort"
if ($WithDemoMcp) { Write-Host "  MCP demo    http://localhost:3333/mcp" }
if ($ip) {
    Write-Host "`nDesde otra maquina de la red:" -ForegroundColor Cyan
    Write-Host "  UI          http://${ip}:$FrontendPort"
    Write-Host "  Backend     http://${ip}:$BackendPort   <- ponlo en el campo 'Backend' de la UI"
    Write-Host "`nSi no conecta, abre los puertos en el firewall:" -ForegroundColor Yellow
    Write-Host "  scripts\open-firewall.ps1   (requiere admin)"
}
