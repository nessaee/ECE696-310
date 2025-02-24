#!/bin/bash

# Set tokenizer parallelism to avoid warnings
export TOKENIZERS_PARALLELISM=false

# process.sh - Results Processing Script
#
# This script processes and analyzes the results from all experiments. It:
# 1. Consolidates multiple runs for each model
# 2. Generates performance analysis visualizations and metrics
#
# Usage: ./process.sh

# Define models to process
models=("gpt2-small" "distilgpt2")

echo "Processing experiment results..."
echo "============================="

# Process each model's results
for model in "${models[@]}"; do
    echo "\nProcessing $model results..."
    echo "-------------------------"
    
    # Consolidate multiple runs
    echo "1. Consolidating runs..."
    python3 scripts/consolidate_runs.py "results/$model"
    
    # Generate performance analysis
    echo "\n2. Generating performance analysis..."
    python3 scripts/performance_analysis.py "results/$model"
    
    echo "Completed processing $model results."
done

echo "\nAll results processed successfully!"
echo "Results can be found in the 'results/<model>/analysis' directories."
