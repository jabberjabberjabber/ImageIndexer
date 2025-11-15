#!/bin/bash

echo "Starting..."

VENV_NAME="llmii_env"

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

if ! command_exists python3; then
    echo "Python 3 is not found. Please ensure Python 3 is installed and added to your PATH."
    read -p "Press Enter to exit..."
    exit 1
fi

if ! command_exists exiftool; then
    bash setup.sh
fi

if [ ! -d "$VENV_NAME" ]; then
    bash setup.sh
fi

source "$VENV_NAME/bin/activate"

python3 -m src.llmii_gui

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Python script failed. See above for details."
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Done."

deactivate