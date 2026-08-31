# OceanViz 3D — one-click launcher for Windows
# Usage: .\run.ps1

$ErrorActionPreference = "Stop"

Write-Host "OceanViz 3D — SIH26067 Grand Finale Prototype" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# Install missing packages if needed
$packages = @("fastapi", "uvicorn", "numpy", "Pillow", "matplotlib", "aiofiles", "python-pptx")
$missing = $packages | Where-Object { (pip show $_ 2>&1) -match "WARNING: Package\(s\) not found" }
if ($missing) {
    Write-Host "Installing missing packages: $missing" -ForegroundColor Yellow
    pip install $missing
}

# Find and free port 8765
$listener = Get-NetTCPConnection -LocalPort 8765 -ErrorAction SilentlyContinue
if ($listener) {
    $proc = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "Port 8765 is in use by $($proc.Name) (PID $($proc.Id)). Stopping..." -ForegroundColor Yellow
        Stop-Process -Id $proc.Id -Force
        Start-Sleep -Seconds 1
    }
}

# Start server
Write-Host "Starting FastAPI server on http://localhost:8765" -ForegroundColor Green
python main.py
