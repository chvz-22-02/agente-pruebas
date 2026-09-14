# Instala todas las dependencias del proyecto (backend y frontend).
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "== Backend ==" -ForegroundColor Cyan
Push-Location (Join-Path $root "backend")

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "Falta 'uv'. Instalalo con: winget install astral-sh.uv"
}

# MLflow y el SDK de MCP no publican ruedas para Python 3.14 todavia.
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -e .

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Creado backend\.env a partir de .env.example" -ForegroundColor Green
}
Pop-Location

Write-Host "`n== Frontend ==" -ForegroundColor Cyan
Push-Location (Join-Path $root "frontend")
npm install
# npm 12 bloquea los scripts de instalacion; esbuild necesita el suyo para
# descargar su binario nativo, sin el Vite no arranca.
npm install-scripts approve esbuild
Pop-Location

Write-Host "`n== Ollama ==" -ForegroundColor Cyan
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "Ollama ya esta instalado: $(ollama --version)" -ForegroundColor Green
    Write-Host "Descarga el modelo con:  ollama pull qwen3:8b"
} else {
    Write-Host "Ollama no esta instalado. Instalalo con:" -ForegroundColor Yellow
    Write-Host "    winget install Ollama.Ollama"
    Write-Host "    ollama pull qwen3:8b"
}

Write-Host "`nListo. Arranca todo con: scripts\start-all.ps1" -ForegroundColor Green
