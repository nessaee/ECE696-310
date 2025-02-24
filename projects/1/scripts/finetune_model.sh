#!/bin/bash

# Script for fine-tuning models
set -e  # Exit on error

# Set up logging
LOG_DIR="logs"
# Log directory is created by config.py
LOG_FILE="$LOG_DIR/finetune_$(date +%Y%m%d_%H%M%S).log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Fine-tune a model on a specific dataset"
    echo ""
    echo "Options:"
    echo "  --model MODEL              Model to fine-tune (gpt2-small,gpt-neo-125m,distilgpt2)"
    echo "  --dataset DATASET          Dataset to use (imdb,ag_news,wikitext,samsum)"
    echo "  --epochs NUM               Number of training epochs (default: 3)"
    echo "  --learning-rate RATE       Learning rate (default: from config)"
    echo "  --batch-size SIZE          Batch size (default: from config)"
    echo "  --eval-steps STEPS         Steps between evaluations (default: 1)"
    echo "  --save-steps STEPS         Steps between model saves (default: 1)"
    echo "  --output-dir DIR           Directory to save model (default: models)"

    echo "  --help                     Show this help message"
    exit 1
}

# Default values
EPOCHS=3
EVAL_STEPS=1
SAVE_STEPS=1
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
        --epochs)
            EPOCHS=$2
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE=$2
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE=$2
            shift 2
            ;;
        --eval-steps)
            EVAL_STEPS=$2
            shift 2
            ;;
        --save-steps)
            SAVE_STEPS=$2
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

# Set experiment name
EXPERIMENT_NAME="train_${MODEL}_${DATASET}"

# Build command
CMD="python src/main.py --mode train --model $MODEL --dataset $DATASET"
CMD="$CMD --num_epochs $EPOCHS --eval_steps $EVAL_STEPS --save_steps $SAVE_STEPS"

# Get next run number using Python utility
RUN_NUM=$(python -c "from pathlib import Path; from src.utils.run_utils import get_next_run_number; print(get_next_run_number(Path('${OUTPUT_DIR}/${MODEL}')))")

# Setup output directory structure
BASE_DIR="${OUTPUT_DIR}/${MODEL}/run_${RUN_NUM}"
OUTPUT_DIR="${BASE_DIR}/train"

# Create all required directories
mkdir -p "${OUTPUT_DIR}/metrics"
mkdir -p "${OUTPUT_DIR}/analysis"
mkdir -p "${OUTPUT_DIR}/weights"

# Save current run number for baseline evaluation
mkdir -p "${BASE_DIR}"
echo "$RUN_NUM" > "${BASE_DIR}/.current_run"

# Add experiment name and set output dir
CMD="$CMD --experiment-name $EXPERIMENT_NAME"
export RESULTS_DIR="$OUTPUT_DIR"

echo "Output dir: $OUTPUT_DIR"
echo "Command: $CMD"


# Add optional arguments if provided
if [ ! -z "$LEARNING_RATE" ]; then
    CMD="$CMD --learning_rate $LEARNING_RATE"
fi

if [ ! -z "$BATCH_SIZE" ]; then
    CMD="$CMD --batch_size $BATCH_SIZE"
fi



# Run fine-tuning
log "Starting fine-tuning for model: $MODEL on dataset: $DATASET"
log "Command: $CMD"

eval "$CMD" 2>&1 | tee -a "$LOG_FILE"

# Check if fine-tuning was successful
if [ $? -eq 0 ]; then
    log "Fine-tuning completed successfully!"
    log "Model saved to: $OUTPUT_DIR"
else
    log "Error: Fine-tuning failed!"
    exit 1
fi
