#!/usr/bin/env bash
# OceanViz 3D — one-click launcher for Linux/macOS

set -e

echo "OceanViz 3D — SIH26067 Grand Finale Prototype"
echo "=============================================="

# Check Python
python3 --version || python --version

# Install dependencies if pip available
pip install -r requirements.txt

# Free port 8765 if needed
if lsof -ti:8765 >/dev/null 2>&1; then
    echo "Port 8765 in use, stopping old process..."
    kill -9 $(lsof -ti:8765) || true
    sleep 1
fi

# Start server
echo "Starting FastAPI server on http://localhost:8765"
python main.py
