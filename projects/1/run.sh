#!/bin/bash

# Set tokenizer parallelism to avoid warnings
export TOKENIZERS_PARALLELISM=false

# run.sh - Experiment Pipeline Runner
# 
# This script orchestrates the entire experiment pipeline for training and evaluating
# language models on different datasets. It handles data downloading, baseline evaluation,
# model fine-tuning, and performance visualization.
#
# Usage: ./run.sh
# The script will execute a predefined set of experiments with different models,
# datasets, and hyperparameters.

# Helper function to run a single experiment pipeline
run_experiment() {
    local model=$1      # Model name (e.g., 'distilgpt2', 'gpt2-small')
    local dataset=$2    # Dataset name (e.g., 'imdb', 'wikitext')
    local task=$3       # Task type (e.g., 'classification', 'language-modeling')
    local epochs=$4     # Number of training epochs
    
    echo "Starting experiment..."
    echo "Configuration:"
    echo "  - Model: $model"
    echo "  - Dataset: $dataset"
    echo "  - Task: $task"
    echo "  - Epochs: $epochs"
    
    # 1. Download data and model
    echo "\nStep 1: Downloading data and model..."
    ./scripts/download_data.sh --datasets "$dataset" --models "$model" --task "$task"
    
    # 2. Run baseline evaluation
    echo "\nStep 2: Running baseline evaluation..."
    ./scripts/evaluate_baseline.sh \
        --model "$model" \
        --dataset "$dataset" \
        --task "$task"
    
    # 3. Fine-tune model
    echo "\nStep 3: Fine-tuning model..."
    ./scripts/finetune_model.sh \
        --model "$model" \
        --dataset "$dataset" \
        --epochs "$epochs"
    
    # 4. Generate visualizations
    echo "\nStep 4: Generating performance visualizations..."
    python scripts/performance_analysis.py "results/$model"
    
    echo "\nExperiment complete!"
    echo "Results saved in results/$model/analysis/"
}

# Execute experiments
echo "Starting experiments..."
echo "===================="

# IMDB Classification Experiments
echo "\nRunning IMDB Classification experiments..."
echo "-------------------------------------------"
# Compare different epoch lengths
run_experiment "distilgpt2" "imdb" "classification" 3
run_experiment "gpt2-small" "imdb" "classification" 3

# WikiText-2 Language Modeling Experiments
# echo "\nRunning WikiText-2 Language Modeling experiments..."
# echo "------------------------------------------------"
# run_experiment "gpt2-small" "wikitext" "language-modeling" 3
# run_experiment "distilgpt2" "wikitext" "language-modeling" 3

echo "\nAll experiments completed successfully!"
