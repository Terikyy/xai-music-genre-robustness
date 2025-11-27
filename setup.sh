#!/bin/bash

# Setup script for XAI Music Genre Robustness Project
# This script:
# 1. Creates and activates a Python virtual environment
# 2. Installs required packages from requirements.txt
# 3. Downloads the GTZAN dataset from Kaggle (if not already present)

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Project directories
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_RAW_DIR="$PROJECT_DIR/data/raw"
VENV_DIR="$PROJECT_DIR/venv"

echo -e "${GREEN}=== XAI Music Genre Robustness Setup ===${NC}\n"

# ============================
# 1. Python Environment Setup
# ============================
echo -e "${GREEN}[1/3] Setting up Python environment...${NC}"

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists.${NC}"
    source "$VENV_DIR/bin/activate"
    
    # Check if all packages are installed
    echo "Checking installed packages..."
    MISSING_PACKAGES=false
    while IFS= read -r package || [ -n "$package" ]; do
        # Skip empty lines and comments
        [[ -z "$package" || "$package" =~ ^#.*$ ]] && continue
        
        # Extract package name (without version specifiers)
        package_name=$(echo "$package" | sed 's/[>=<].*//' | xargs)
        
        if ! pip show "$package_name" &> /dev/null; then
            echo -e "${YELLOW}Missing package: $package_name${NC}"
            MISSING_PACKAGES=true
            break
        fi
    done < "$PROJECT_DIR/requirements.txt"
    
    if [ "$MISSING_PACKAGES" = false ]; then
        echo -e "${GREEN}All required packages are already installed.${NC}"
    else
        echo "Installing missing packages..."
        pip install -r "$PROJECT_DIR/requirements.txt"
        echo -e "${GREEN}Packages installed successfully.${NC}"
    fi
else
    echo "Creating new virtual environment..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    echo "Upgrading pip..."
    pip install --upgrade pip
    
    echo "Installing required packages..."
    pip install -r "$PROJECT_DIR/requirements.txt"
    echo -e "${GREEN}Virtual environment created and packages installed.${NC}"
fi

# ============================
# 2. Kaggle Setup
# ============================
echo -e "\n${GREEN}[2/3] Checking Kaggle configuration...${NC}"

# Check if kaggle is installed
if ! pip show kaggle &> /dev/null; then
    echo "Installing kaggle package..."
    pip install kaggle
fi

# Check for Kaggle API token
if [ -z "$KAGGLE_API_TOKEN" ]; then
    echo -e "${RED}Error: Kaggle API token not found!${NC}"
    echo -e "Please follow these steps:"
    echo -e "  1. Go to https://www.kaggle.com/settings/account"
    echo -e "  2. Scroll to 'API' section and click 'Create New Token'"
    echo -e "  3. Copy the export command (e.g., export KAGGLE_API_TOKEN=KGAT_...)"
    echo -e "  4. Run the export command in your terminal (sets environment variable)"
    echo -e "  5. Re-run this setup script"
    echo -e "\n${YELLOW}Tip: Add the export command to your ~/.zshrc for persistence${NC}"
    exit 1
fi

echo -e "${GREEN}Kaggle API token found.${NC}"

# ============================
# 3. Dataset Download
# ============================
echo -e "\n${GREEN}[3/3] Checking dataset...${NC}"

# Check dataset: set flags for presence and non-empty directories
genres_present=false
images_present=false
csv1_present=false
csv2_present=false

if [ -d "$DATA_RAW_DIR/genres_original" ] && [ -n "$(ls -A "$DATA_RAW_DIR/genres_original" 2>/dev/null)" ]; then
    genres_present=true
fi

if [ -d "$DATA_RAW_DIR/images_original" ] && [ -n "$(ls -A "$DATA_RAW_DIR/images_original" 2>/dev/null)" ]; then
    images_present=true
fi

if [ -f "$DATA_RAW_DIR/features_3_sec.csv" ]; then
    csv1_present=true
fi

if [ -f "$DATA_RAW_DIR/features_30_sec.csv" ]; then
    csv2_present=true
fi

if [ "$genres_present" = true ] && [ "$images_present" = true ] && [ "$csv1_present" = true ] && [ "$csv2_present" = true ]; then
    echo -e "${GREEN}Dataset already exists in data/raw/${NC}"
    echo -e "${GREEN}All required files and directories are present.${NC}"
else
    # If any part exists but not all, clear the raw directory before downloading
    if [ "$genres_present" = true ] || [ "$images_present" = true ] || [ "$csv1_present" = true ] || [ "$csv2_present" = true ]; then
        echo -e "${YELLOW}Partial dataset detected in $DATA_RAW_DIR. Clearing directory to prepare clean download...${NC}"
        mkdir -p "$DATA_RAW_DIR"
        # Remove contents only
        rm -rf "$DATA_RAW_DIR"/*
        echo -e "${GREEN}Cleared $DATA_RAW_DIR.${NC}"
    fi

    echo "Downloading GTZAN dataset from Kaggle..."
    echo "Dataset: andradaolteanu/gtzan-dataset-music-genre-classification"
    
    # Create data/raw directory if it doesn't exist
    mkdir -p "$DATA_RAW_DIR"
    
    # Download dataset
    cd "$DATA_RAW_DIR"
    kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
    
    echo "Extracting dataset..."
    unzip -q gtzan-dataset-music-genre-classification.zip
    
    # Clean up zip file
    rm gtzan-dataset-music-genre-classification.zip
    
    # Move files if they're in a subdirectory
    if [ -d "Data" ]; then
        mv Data/genres_original ./
        mv Data/images_original ./
        mv Data/features_*.csv ./ 2>/dev/null || true
        rmdir Data 2>/dev/null || true
    fi
    
    cd "$PROJECT_DIR"
    
    echo -e "${GREEN}Dataset downloaded and extracted successfully.${NC}"
fi

# ============================
# Setup Complete
# ============================
echo -e "\n${GREEN}=== Setup Complete! ===${NC}"
echo -e "\nTo activate the virtual environment, run:"
echo -e "  ${YELLOW}source venv/bin/activate${NC}"
echo -e "\nProject structure:"
echo -e "  - Virtual environment: ${YELLOW}venv/${NC}"
echo -e "  - Dataset: ${YELLOW}data/raw/${NC}"
echo -e "  - Notebooks: ${YELLOW}notebooks/${NC}"
echo -e "  - Source code: ${YELLOW}src/${NC}\n"
