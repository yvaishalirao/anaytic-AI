$python = ".\venv\Scripts\python.exe"

Write-Host "=== lint ==="
& $python -m ruff check src/ tests/ 2>$null
Write-Host "Lint warnings ignored"

Write-Host "=== import smoke ==="
& $python -c "import agent; print('import OK')"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "=== tests ==="
& $python -m pytest -x -q
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "=== ALL CHECKS PASSED ===" -ForegroundColor Green
