#!/bin/bash

# Helper function to analyze a single experiment directory
analyze_experiment() {
    local exp_dir=$1
    local output_dir=$2
    
    echo "Analyzing experiment in: $exp_dir"
    
    # Check for metrics files
    if [ -f "$exp_dir/metrics.csv" ]; then
        echo "Found CSV metrics, analyzing..."
        python src/visualization/analyze_training.py \
            "$exp_dir/metrics.csv" \
            --output-dir "$output_dir"
    fi
    
    if [ -f "$exp_dir/metrics.json" ]; then
        echo "Found JSON metrics, analyzing..."
        python src/visualization/analyze_json_metrics.py \
            "$exp_dir/metrics.json" \
            --output-dir "$output_dir"
    fi
}

# Main script
if [ $# -eq 0 ]; then
    # No arguments provided, analyze all experiment directories
    echo "No directories specified, analyzing all experiments in results/..."
    
    # Find all experiment directories
    for exp_dir in results/train_*; do
        if [ -d "$exp_dir" ]; then
            # Create analysis subdirectory
            analysis_dir="$exp_dir/analysis"
            mkdir -p "$analysis_dir"
            
            # Analyze the experiment
            analyze_experiment "$exp_dir" "$analysis_dir"
            
            echo "Analysis complete! Results saved in: $analysis_dir"
        fi
    done
else
    # Process specified experiment directories
    for exp_dir in "$@"; do
        if [ ! -d "$exp_dir" ]; then
            echo "Error: Directory not found: $exp_dir"
            continue
        fi
        
        # Create analysis subdirectory
        analysis_dir="$exp_dir/analysis"
        mkdir -p "$analysis_dir"
        
        # Analyze the experiment
        analyze_experiment "$exp_dir" "$analysis_dir"
        
        echo "Analysis complete! Results saved in: $analysis_dir"
    done
fi
