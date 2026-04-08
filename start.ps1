Write-Host "Starting agent service..."

# Activate virtual environment
& .\venv\Scripts\Activate.ps1

# Start agent service in background
$agent = Start-Process python -ArgumentList "-m agent.agent_service" -PassThru

Write-Host "Agent service PID: $($agent.Id)"

Write-Host "Starting Streamlit UI..."

# Run Streamlit (foreground)
streamlit run src/ui/app.py

# After Streamlit stops, kill agent service
Write-Host "Stopping agent service..."
Stop-Process -Id $agent.Id -Force