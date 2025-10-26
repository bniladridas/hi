#!/bin/bash

# Script to run the token classification model

set -e

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Token Classification Model${NC}"

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo -e "${GREEN}Using Docker to run the model${NC}"
    docker build -t token-classification-model .
    docker run --rm -it token-classification-model
else
    echo -e "${YELLOW}Using local Python environment${NC}"

    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo -e "${BLUE}Creating virtual environment...${NC}"
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Install dependencies if needed
    if [ ! -f "venv/installed" ]; then
        echo -e "${BLUE}Installing dependencies...${NC}"
        pip install -r requirements.txt
        touch venv/installed
    fi

    # Run the model
    echo -e "${GREEN}Running model training...${NC}"
    python src/model.py
fi
