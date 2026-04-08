#!/usr/bin/env bash
set -e

# Activate virtual environment
source venv/Scripts/activate

echo "Starting agent service..."
python -m agent.agent_service &
AGENT_PID=$!
echo "Agent service PID: $AGENT_PID"
echo "Starting Streamlit UI..."
streamlit run src/ui/app.py
kill $AGENT_PID
