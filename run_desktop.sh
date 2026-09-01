#!/bin/bash
# NetShield AI Desktop Application Launcher
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

if [ -f "venv/bin/python" ]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "Launching NetShield AI Standalone Desktop Window Application..."
exec "$PYTHON_BIN" desktop_app.py "$@"
