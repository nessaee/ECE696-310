#!/bin/bash

# Script for running baseline model evaluation
set -e  # Exit on error

# Set up logging
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/baseline_eval_$(date +%Y%m%d_%H%M%S).log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Run baseline evaluation for models"
    echo ""
    echo "Options:"
    echo "  --model MODEL         Model to evaluate (gpt2-small,gpt-neo-125m,distilgpt2)"
    echo "  --dataset DATASET     Dataset to use (imdb,ag_news,wikitext,samsum)"
    echo "  --output-dir DIR      Directory to save results (default: results)"
    echo "  --help               Show this help message"
    exit 1
}

# Default values
OUTPUT_DIR="results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL=$2
            shift 2
            ;;
        --dataset)
            DATASET=$2
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR=$2
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
if [ -z "$MODEL" ] || [ -z "$DATASET" ]; then
    echo "Error: Missing required arguments"
    usage
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Activate virtual environment and set PYTHONPATH
source venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

# Run evaluation
log "Starting baseline evaluation for model: $MODEL on dataset: $DATASET"
python src/main.py \
    --mode evaluate \
    --model "$MODEL" \
    --dataset "$DATASET" \
    2>&1 | tee -a "$LOG_FILE"

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    log "Evaluation completed successfully!"
    log "Results saved to: $OUTPUT_DIR"
else
    log "Error: Evaluation failed!"
    exit 1
fi
