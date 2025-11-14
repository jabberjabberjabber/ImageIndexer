#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REQUIREMENTS_FILE="$SCRIPT_DIR/../requirements.txt"
VENV_NAME="SCRIPT_DIR/llmii_env"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

if ! command_exists python3; then
    echo "Python 3 is not found. Please ensure Python 3 is installed and added to your PATH."
	read -p "Press Enter to exit..."
    exit 1
fi

# Check if exiftool is installed
if ! command_exists exiftool; then
    echo "exiftool is not found. Attempting to install..."
    
    # Try to install based on the OS
    if [[ "$(uname)" == "Darwin" ]]; then
        if command_exists brew; then
            echo "Installing exiftool using Homebrew..."
            brew install exiftool
        else
            echo "Homebrew not found. Please install Homebrew first, then run 'brew install exiftool'"
			read -p "Press Enter to exit..."
            exit 1
        fi
    elif [[ "$(uname)" == "Linux" ]]; then
        if command_exists apt-get; then
            echo "Installing exiftool using apt..."
            sudo apt-get update && sudo apt-get install -y libimage-exiftool-perl
        elif command_exists dnf; then
            echo "Installing exiftool using dnf..."
            sudo dnf install -y perl-Image-ExifTool
        elif command_exists yum; then
            echo "Installing exiftool using yum..."
            sudo yum install -y perl-Image-ExifTool
        elif command_exists pacman; then
            echo "Installing exiftool using pacman..."
            sudo pacman -S --noconfirm perl-image-exiftool
        else
            echo "Could not determine package manager. Please install exiftool manually."
			read -p "Press Enter to exit..."
            exit 1
        fi
    else
        echo "Unsupported operating system. Please install exiftool manually."
		read -p "Press Enter to exit..."
        exit 1
    fi
    
    if ! command_exists exiftool; then
        echo "Failed to install exiftool. Please install it manually."
		read -p "Press Enter to exit..."
        exit 1
    else
        echo "exiftool has been installed successfully."
    fi
else
    echo "exiftool is already installed."
fi

if [ ! -d "$VENV_NAME" ]; then
    echo "Creating new virtual environment: $VENV_NAME"
    python3 -m venv "$VENV_NAME"
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please check your Python installation."
		read -p "Press Enter to exit..."
        exit 1
    fi
else
    echo "Virtual environment $VENV_NAME already exists."
fi

source "$VENV_NAME/bin/activate"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "requirements.txt not found."
	read -p "Press Enter to exit..."
    exit 1
fi

python3 -m pip install --upgrade pip

echo "Installing packages from requirements.txt..."
pip install -r "$REQUIREMENTS_FILE"
if [ $? -ne 0 ]; then
    echo "Failed to install some packages. Please check your internet connection and requirements.txt file."
	read -p "Press Enter to exit..."
    exit 1
fi

clear

python3 -m src.llmii_setup --update

deactivate

read -p "Press Enter to exit..."
