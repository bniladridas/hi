#!/bin/bash

# Script to run the token classification model

set -e

echo "🚀 Starting Token Classification Model"

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "🐳 Using Docker to run the model"
    docker build -t token-classification-model .
    docker run --rm -it token-classification-model
else
    echo "🐍 Using local Python environment"

    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Install dependencies if needed
    if [ ! -f "venv/installed" ]; then
        echo "📦 Installing dependencies..."
        pip install -r requirements.txt
        touch venv/installed
    fi

    # Run the model
    echo "🏃 Running model training..."
    python src/model.py
fi
