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
        
        # Determine task type from metadata
        local task="classification"  # Default task
        if [ -f "$exp_dir/metrics.json" ]; then
            local dataset=$(jq -r '.metadata.dataset' "$exp_dir/metrics.json")
            if [ "$dataset" = "wikitext" ]; then
                task="language-modeling"
            fi
        fi
        
        # Use appropriate visualization script
        if [ "$task" = "language-modeling" ]; then
            echo "Detected language modeling task, using LM metrics analyzer..."
            python src/visualization/analyze_lm_metrics.py \
                "$exp_dir/metrics.json" \
                --output-dir "$output_dir"
        else
            echo "Detected classification task, using classification metrics analyzer..."
            python src/visualization/analyze_json_metrics.py \
                "$exp_dir/metrics.json" \
                --output-dir "$output_dir"
        fi
    fi
}

# Main script
if [ $# -eq 0 ]; then
    # No arguments provided, analyze all experiment directories
    echo "No directories specified, analyzing all experiments in results/..."
    
    # Find all experiment directories
    for exp_dir in report/data/train_*; do
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
