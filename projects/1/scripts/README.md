# Project Scripts

This directory contains shell scripts for automating various tasks in the project.

## Available Scripts

### 1. Download Data (`download_data.sh`)
Downloads datasets and models for the project.

```bash
# Example: Download IMDB dataset and GPT-2 Small model for classification
./download_data.sh \
    --datasets imdb \
    --models gpt2-small \
    --task classification

# Example: Download multiple datasets and models
./download_data.sh \
    --datasets imdb,wikitext \
    --models gpt2-small,distilgpt2 \
    --task language-modeling
```

### 2. Evaluate Baseline (`evaluate_baseline.sh`)
Run baseline evaluation for models.

```bash
# Example: Evaluate GPT-2 Small on IMDB dataset
./evaluate_baseline.sh \
    --model gpt2-small \
    --dataset imdb \
    --output-dir results/baseline

# Example: Evaluate with custom output directory
./evaluate_baseline.sh \
    --model gpt2-small \
    --dataset wikitext \
    --output-dir results/language_modeling
```

### 3. Fine-tune Model (`finetune_model.sh`)
Fine-tune a model on a specific dataset.

```bash
# Example: Basic fine-tuning
./finetune_model.sh \
    --model gpt2-small \
    --dataset imdb \
    --epochs 3

# Example: Advanced fine-tuning with custom parameters
./finetune_model.sh \
    --model gpt2-small \
    --dataset imdb \
    --epochs 5 \
    --learning-rate 2e-5 \
    --batch-size 16 \
    --eval-steps 100 \
    --save-steps 500 \
    --use-wandb
```

## Common Options

- All scripts support the `--help` flag for detailed usage information
- All scripts create logs in the `logs` directory
- Results and models are saved in their respective directories (`results/` and `models/`)

## Prerequisites

1. Make sure you have activated the virtual environment:
```bash
source venv/bin/activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Logging

- All scripts create detailed logs in the `logs` directory
- Logs include timestamps and error messages
- Each run creates a new log file with a unique timestamp

## Error Handling

- Scripts will exit immediately if an error occurs (`set -e`)
- Error messages are logged to both console and log file
- Exit codes indicate success (0) or failure (non-zero)
