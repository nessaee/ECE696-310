"""
Script to download and set up models and datasets for the LLM project.
"""

import os
from pathlib import Path
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch

# Define model options
AVAILABLE_MODELS = {
    'gpt2-small': 'gpt2',
    'gpt-neo-125m': 'EleutherAI/gpt-neo-125m',
    'distilgpt2': 'distilgpt2'
}

# Define dataset options
AVAILABLE_DATASETS = {
    'imdb': ('imdb', 'plain_text'),
    'ag_news': ('ag_news', None),
    'wikitext': ('wikitext', 'wikitext-2-raw-v1'),
    'samsum': ('samsum', None)
}

def setup_directories():
    """Create necessary directories if they don't exist."""
    base_dir = Path(__file__).parent.parent.parent
    dirs = ['data', 'models']
    for dir_name in dirs:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
    return base_dir

def download_model(model_name, task='language-modeling', base_dir=None):
    """
    Download and save the specified model.
    
    Args:
        model_name (str): Name of the model to download
        task (str): Task type ('language-modeling' or 'classification')
        base_dir (Path): Base directory for saving models
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    model_id = AVAILABLE_MODELS[model_name]
    save_dir = base_dir / 'models' / model_name
    
    print(f"Downloading {model_name} from {model_id}...")
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(save_dir)
    
    # Download model based on task
    if task == 'language-modeling':
        model = AutoModelForCausalLM.from_pretrained(model_id)
    elif task == 'classification':
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    model.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")
    return save_dir

def download_dataset(dataset_name, base_dir=None):
    """
    Download and save the specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset to download
        base_dir (Path): Base directory for saving datasets
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(AVAILABLE_DATASETS.keys())}")
    
    dataset_id, subset = AVAILABLE_DATASETS[dataset_name]
    save_dir = base_dir / 'data' / dataset_name
    
    print(f"Downloading dataset {dataset_name}...")
    
    # Load dataset
    if subset:
        dataset = load_dataset(dataset_id, subset)
    else:
        dataset = load_dataset(dataset_id)
    
    # Save dataset info
    dataset_info = {
        'name': dataset_name,
        'num_train': len(dataset['train']),
        'num_test': len(dataset['test']),
        'features': list(dataset['train'].features.keys())
    }
    
    # Save dataset info to a file
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / 'info.txt', 'w') as f:
        for key, value in dataset_info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Dataset info saved to {save_dir}")
    return dataset

def main():
    parser = argparse.ArgumentParser(description='Download models and datasets for LLM project')
    parser.add_argument('--models', nargs='+', choices=list(AVAILABLE_MODELS.keys()),
                      help='Models to download')
    parser.add_argument('--datasets', nargs='+', choices=list(AVAILABLE_DATASETS.keys()),
                      help='Datasets to download')
    parser.add_argument('--task', choices=['language-modeling', 'classification'],
                      default='language-modeling', help='Task type for model configuration')
    
    args = parser.parse_args()
    base_dir = setup_directories()
    
    if args.models:
        for model_name in args.models:
            try:
                download_model(model_name, args.task, base_dir)
            except Exception as e:
                print(f"Error downloading model {model_name}: {str(e)}")
    
    if args.datasets:
        for dataset_name in args.datasets:
            try:
                download_dataset(dataset_name, base_dir)
            except Exception as e:
                print(f"Error downloading dataset {dataset_name}: {str(e)}")

if __name__ == "__main__":
    main()
