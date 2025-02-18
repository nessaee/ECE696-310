#!/bin/bash

# Script for downloading datasets and models
set -e  # Exit on error

# Set up logging
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/download_$(date +%Y%m%d_%H%M%S).log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Download datasets and models for the project"
    echo ""
    echo "Options:"
    echo "  --datasets DATASETS    Comma-separated list of datasets (imdb,ag_news,wikitext,samsum)"
    echo "  --models MODELS        Comma-separated list of models (gpt2-small,gpt-neo-125m,distilgpt2)"
    echo "  --task TASK           Task type (classification,language-modeling,summarization)"
    echo "  --help                Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)
            DATASETS=$2
            shift 2
            ;;
        --models)
            MODELS=$2
            shift 2
            ;;
        --task)
            TASK=$2
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate arguments
if [ -z "$DATASETS" ] || [ -z "$MODELS" ] || [ -z "$TASK" ]; then
    echo "Error: Missing required arguments"
    usage
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    log "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment and set PYTHONPATH
source venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

# Install requirements
log "Installing requirements..."
pip install -r requirements.txt

# Download datasets and models
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

# For classification tasks, download datasets first
if [ "$TASK" = "classification" ]; then
    for dataset in "${DATASET_ARRAY[@]}"; do
        log "Downloading dataset: $dataset"
        python src/main.py --mode download --dataset "$dataset" 2>&1 | tee -a "$LOG_FILE"
    done

    # Then download models with dataset information
    for model in "${MODEL_ARRAY[@]}"; do
        for dataset in "${DATASET_ARRAY[@]}"; do
            log "Downloading model: $model for dataset: $dataset"
            python src/main.py --mode download --model "$model" --task "$TASK" --dataset "$dataset" 2>&1 | tee -a "$LOG_FILE"
        done
    done
else
    # For other tasks, order doesn't matter
    for dataset in "${DATASET_ARRAY[@]}"; do
        log "Downloading dataset: $dataset"
        python src/main.py --mode download --dataset "$dataset" 2>&1 | tee -a "$LOG_FILE"
    done

    for model in "${MODEL_ARRAY[@]}"; do
        log "Downloading model: $model"
        python src/main.py --mode download --model "$model" --task "$TASK" 2>&1 | tee -a "$LOG_FILE"
    done
fi

log "Download completed successfully!"
