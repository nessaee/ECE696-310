"""
Base configuration module for the project.
"""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Directory paths
DATA_DIR = PROJECT_ROOT / 'data'  # Raw and processed data
MODELS_DIR = PROJECT_ROOT / 'models'  # Model checkpoints and configs
OUTPUT_DIR = PROJECT_ROOT / 'output'  # All output files
CONFIGS_DIR = PROJECT_ROOT / 'configs'  # Configuration files

# Output subdirectories
RESULTS_DIR = OUTPUT_DIR / 'results'  # Training and evaluation results
ANALYSIS_DIR = OUTPUT_DIR / 'analysis'  # Analysis outputs and visualizations
LOGS_DIR = OUTPUT_DIR / 'logs'  # Log files

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, CONFIGS_DIR, 
                RESULTS_DIR, ANALYSIS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    'gpt2-small': {
        'name': 'gpt2',
        'max_length': 512,
        'batch_size': 8,
        'learning_rate': 5e-5,
    },
    'gpt-neo-125m': {
        'name': 'EleutherAI/gpt-neo-125m',
        'max_length': 512,
        'batch_size': 4,
        'learning_rate': 3e-5,
    },
    'distilgpt2': {
        'name': 'distilgpt2',
        'max_length': 512,
        'batch_size': 8,
        'learning_rate': 5e-5,
    }
}

# Dataset configurations
DATASET_CONFIGS = {
    'imdb': {
        'name': 'imdb',
        'task': 'classification',
        'num_labels': 2,
        'max_length': 512,
        'batch_size': 16,
        'text_column': 'text',
        'label_column': 'label'
    },
    'ag_news': {
        'name': 'ag_news',
        'task': 'classification',
        'num_labels': 4,
        'max_length': 256,
        'batch_size': 32,
        'text_column': 'text',
        'label_column': 'label'
    },
    'wikitext': {
        'name': 'wikitext',
        'subset': 'wikitext-2-raw-v1',
        'task': 'language-modeling',
        'max_length': 512,
        'batch_size': 8,
        'text_column': 'text'
    },
    'samsum': {
        'name': 'samsum',
        'task': 'summarization',
        'max_length': 512,
        'target_max_length': 128,
        'batch_size': 8,
        'text_column': 'dialogue',
        'summary_column': 'summary'
    }
}
