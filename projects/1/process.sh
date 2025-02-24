#!/bin/bash

# Process experiment results
OUTPUT_DIR="analysis_output/distilgpt2"
python -m src.visualization.analyze_results results/distilgpt2 --output-dir $OUTPUT_DIR

# Print summary of results
echo "Analysis complete! Results saved to $OUTPUT_DIR/"
echo "Generated files:"
echo "  - $OUTPUT_DIR/all_results.csv"
echo "  - $OUTPUT_DIR/summary_statistics.csv"
echo "  - $OUTPUT_DIR/*.png (training curves)"
