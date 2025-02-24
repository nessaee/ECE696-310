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
    echo "  --task TASK           Task type (classification,language-modeling,summarization)"
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
        --task)
            TASK=$2
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
if [ -z "$MODEL" ] || [ -z "$DATASET" ] || [ -z "$TASK" ]; then
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

# Validate task type
case "$TASK" in
    "classification"|"language-modeling"|"summarization")
        ;;
    *)
        log "Error: Invalid task type $TASK"
        exit 1
        ;;
esac

# Set experiment name
EXPERIMENT_NAME="baseline_${MODEL}_${DATASET}"

# Build command
CMD="python src/main.py --mode evaluate --model $MODEL --dataset $DATASET --task $TASK --is-baseline"

# Add experiment name if output dir is provided
if [ ! -z "$OUTPUT_DIR" ]; then
    CMD="$CMD --experiment-name $EXPERIMENT_NAME"
    
        # Get run number from training directory or create new one
    MODEL_DIR="${OUTPUT_DIR}/${MODEL}"
    if [ -f "${MODEL_DIR}/.current_run" ]; then
        RUN_NUM=$(cat "${MODEL_DIR}/.current_run")
    else
        # Get next run number using Python utility
        RUN_NUM=$(python -c "from pathlib import Path; from src.utils.run_utils import get_next_run_number; print(get_next_run_number(Path('${OUTPUT_DIR}/${MODEL}')))")
    fi
    
    # Setup output directory
    OUTPUT_DIR="${OUTPUT_DIR}/${MODEL}/run_${RUN_NUM}/baseline"
    mkdir -p "$OUTPUT_DIR/metrics"
    mkdir -p "$OUTPUT_DIR/analysis"
    
    # Export output directory for the evaluator
    export RESULTS_DIR="$OUTPUT_DIR"
fi

# Run evaluation
log "Command: $CMD"
eval "$CMD" 2>&1 | tee -a "$LOG_FILE"

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    log "Evaluation completed successfully!"
    log "Results saved to: $OUTPUT_DIR"
else
    log "Error: Evaluation failed!"
    exit 1
fi
