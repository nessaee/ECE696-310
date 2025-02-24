# LLM Fine-tuning Project

This project involves fine-tuning and evaluating small open-source language models (like GPT-2 and DistilGPT-2) on various NLP tasks. The project includes a comprehensive pipeline for model training, evaluation, and performance analysis.

## Project Structure

```
.
├── data/               # Dataset storage
├── models/            # Model checkpoints and configs
├── results/           # Experimental results
│   ├── gpt2-small/    # Results for GPT-2 Small
│   └── distilgpt2/    # Results for DistilGPT-2
├── scripts/           # Analysis and processing scripts
│   ├── consolidate_runs.py    # Consolidates experiment results
│   └── performance_analysis.py # Generates performance visualizations
├── src/               # Source code
│   ├── data/          # Data processing scripts
│   ├── models/        # Model training code
│   └── evaluation/    # Evaluation scripts
├── process.sh         # Results processing script
├── run.sh            # Experiment runner script
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

### 1. Execute Training Pipeline
Use `run.sh` to execute the complete experiment pipeline:
```bash
./run.sh
```

This script will:
- Download required datasets and models
- Run baseline evaluations
- Fine-tune models with specified configurations
- Generate initial performance visualizations

### 2. Process Results
After experiments complete, use `process.sh` to analyze results:
```bash
./process.sh
```

This will:
- Consolidate results from multiple runs
- Generate comprehensive performance visualizations
- Create summary statistics and reports

## Key Scripts

### consolidate_runs.py
Consolidates and analyzes experimental results:
```bash
python scripts/consolidate_runs.py results/gpt2-small
```

Features:
- Combines metrics from multiple training runs
- Generates baseline performance comparisons
- Creates training progression visualizations
- Outputs standardized CSV files for further analysis

### performance_analysis.py
Generates detailed performance visualizations:
```bash
python scripts/performance_analysis.py results/gpt2-small
```

Features:
- Training progression plots
- Metric comparison visualizations
- Summary statistics generation

## Supported Tasks and Datasets

### Classification
- IMDB Movie Reviews (50K reviews)
  - Binary sentiment classification
  - Metrics: Accuracy, Precision, Recall, F1-score

### Language Modeling
- WikiText-2 (under 1M tokens)
  - Next token prediction
  - Metrics: Perplexity, Loss

## Results

Results are organized by model in the `results/` directory:
```
results/
├── gpt2-small/
│   ├── run_1/        # Individual run results
│   ├── run_2/
│   └── analysis/     # Consolidated analysis
│       ├── training_progression.csv
│       ├── baseline_performance.csv
│       └── performance_summary.csv
└── distilgpt2/
    └── ...
```

Each run directory contains:
- `metrics.json`: Training metrics per epoch
- `test_metrics.json`: Baseline/test performance
- `config.json`: Run configuration

The `analysis/` directory contains consolidated results and visualizations.
