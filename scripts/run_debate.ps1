$ErrorActionPreference = "Stop"

Write-Host "Starting DDA-X Terminal Debate..." -ForegroundColor Cyan

# Check if venv exists
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Gray
    & ".\venv\Scripts\Activate.ps1"
}

# Run the python script
Write-Host "Launching Python agent runner..." -ForegroundColor Green
python run_terminal_debate.py

Write-Host "Done."
