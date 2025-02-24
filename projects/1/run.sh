#!/bin/bash

# Helper function to run experiment pipeline
run_experiment() {
    local model=$1
    local dataset=$2
    local task=$3
    local epochs=$4
    echo "Starting experiment..."
    echo "Model: $model, Dataset: $dataset, Task: $task"
    
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
    python src/visualization/plot_metrics.py \
        "results/$model" \
        --save-dir "results/$model/analysis"
    
    echo "\nExperiment complete!"
}

# Run experiments

echo "Starting IMDB Classification experiments..."
#########################################################################################
run_experiment "distilgpt2" "imdb" "classification" 3
run_experiment "distilgpt2" "imdb" "classification" 7
run_experiment "gpt2-small" "imdb" "classification" 3
run_experiment "gpt2-small" "imdb" "classification" 7

run_experiment "distilgpt2" "imdb" "classification" 3
run_experiment "gpt2-small" "imdb" "classification" 3
run_experiment "distilgpt2" "imdb" "classification" 3
run_experiment "gpt2-small" "imdb" "classification" 3
run_experiment "distilgpt2" "imdb" "classification" 3
run_experiment "gpt2-small" "imdb" "classification" 3
run_experiment "distilgpt2" "imdb" "classification" 3
run_experiment "gpt2-small" "imdb" "classification" 3
echo "\nStarting WikiText-2 Language Modeling experiments..."
#########################################################################################
run_experiment "gpt2-small" "wikitext" "language-modeling" 3
run_experiment "distilgpt2" "wikitext" "language-modeling" 3
