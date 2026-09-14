# Abre en el firewall de Windows los puertos que necesita el acceso desde otra
# maquina de la red local. Requiere ejecutar PowerShell como administrador.
#
#   powershell -ExecutionPolicy Bypass -File scripts\open-firewall.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\open-firewall.ps1 -Remove

param(
    [switch]$Remove,
    [int[]]$Ports = @(8090, 5173, 5000)
)

$ErrorActionPreference = "Stop"

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
           ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    throw "Ejecuta esta ventana de PowerShell como administrador."
}

foreach ($port in $Ports) {
    $name = "AgentePruebasMCP-$port"
    Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    if ($Remove) {
        Write-Host "Regla eliminada: $name" -ForegroundColor Yellow
        continue
    }
    # Solo perfiles Private/Domain: no se expone el servicio en redes publicas.
    New-NetFirewallRule -DisplayName $name -Direction Inbound -Action Allow `
        -Protocol TCP -LocalPort $port -Profile Private,Domain | Out-Null
    Write-Host "Puerto $port abierto (perfiles Private y Domain)" -ForegroundColor Green
}
