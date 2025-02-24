"""
Main script for running experiments.
"""
import argparse
import os
from pathlib import Path
import json
import time
import csv
from datetime import datetime

from src.models.model_handler import ModelHandler
from src.data.dataset import DatasetHandler
from src.evaluation.evaluator import Evaluator
from src.training.trainer import Trainer
from src.utils.config import MODEL_CONFIGS, DATASET_CONFIGS, RESULTS_DIR
from src.utils.logging_config import setup_logging
from src.utils.metrics_tracker import MetricsTracker

def save_results_to_csv(results: dict, filepath: Path, metadata: dict = None):
    """Save results dictionary to CSV format with optional metadata."""
    # Flatten nested dictionary
    flat_dict = {}
    def flatten(d, parent_key=''):
        for k, v in d.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            if isinstance(v, dict):
                flatten(v, new_key)
            else:
                flat_dict[new_key] = v
    
    flatten(results)
    if metadata:
        flat_dict.update(metadata)
    
    # Write to CSV
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(flat_dict.keys()))
        writer.writeheader()
        writer.writerow(flat_dict)

def run_baseline_evaluation(args):
    """Run baseline model evaluation with logging."""
    # Initialize handlers
    model_handler = ModelHandler(args.model, args.dataset)
    dataset_handler = DatasetHandler(args.dataset, model_handler.tokenizer)
    
    # Use model name from config to ensure consistency
    model_name = model_handler.model_config['name']
    
    # Use provided experiment name or generate one
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"baseline_eval_{model_name}_{args.dataset}_{timestamp}"
    
    logger = setup_logging(experiment_name)
    logger.info(f"Starting baseline evaluation for model={args.model}, dataset={args.dataset}")
    
    try:
        # Initialize handlers
        model_handler = ModelHandler(args.model, args.dataset)
        dataset_handler = DatasetHandler(args.dataset, model_handler.tokenizer)
        evaluator = Evaluator(model_handler, dataset_handler, is_baseline=True)
        
        # Run evaluation
        results = evaluator.evaluate('test')
        
        # Add metadata
        metadata = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'model': args.model,
            'dataset': args.dataset,
            'split': 'test'
        }
        
        # Log results
        logger.info(f"Evaluation results:\n{json.dumps(results, indent=2)}")
        print("\nBaseline Evaluation Results:")
        print(json.dumps(results, indent=2))
        
        # Save results
        save_dir = Path("results") / experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save in both CSV and JSON formats
        save_results_to_csv(results, save_dir / "metrics.csv", metadata)
        with open(save_dir / "metrics.json", 'w') as f:
            json.dump({**results, 'metadata': metadata}, f, indent=2)
            
        logger.info(f"Saved results to {save_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

def run_training(args):
    """Run model fine-tuning with enhanced logging and checkpointing."""
    # Use provided experiment name or generate one with timestamp
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"train_{args.model}_{args.dataset}_{timestamp}"
    
    logger = setup_logging(experiment_name)
    logger.info(f"Starting training experiment: {experiment_name}")
    logger.info(f"Args: {vars(args)}")
    
    try:
        # Initialize handlers
        model_handler = ModelHandler(args.model, args.dataset)
        dataset_handler = DatasetHandler(args.dataset, model_handler.tokenizer)
        
        # Initialize trainer with experiment name
        trainer = Trainer(
            model_handler,
            dataset_handler,
            experiment_name=experiment_name
        )
        
        # Initialize evaluator with metrics tracker
        evaluator = Evaluator(
            model_handler, 
            dataset_handler, 
            is_baseline=False,
            metrics_tracker=MetricsTracker(
                experiment_name='test',
                output_dir=trainer.base_dir,  # Use same base dir as trainer
                model_name=model_handler.model_config['name']
            )
        )
        
        # Run training
        metrics = trainer.train(
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps
        )
        
        # Add metadata
        metadata = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'model': args.model,
            'dataset': args.dataset,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'eval_steps': args.eval_steps,
            'save_steps': args.save_steps
        }
        
        # Save final results
        save_dir = Path("results") / experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save in both CSV and JSON formats
        save_results_to_csv(metrics, save_dir / "metrics.csv", metadata)
        with open(save_dir / "metrics.json", 'w') as f:
            json.dump({**metrics, 'metadata': metadata}, f, indent=2)
        
        logger.info(f"Training completed successfully. Results saved to {save_dir}")
        print("\nTraining Results:")
        print(json.dumps(metrics, indent=2))
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def download_data(args):
    """Download dataset and cache it."""
    try:
        from datasets import load_dataset
        print(f"Downloading dataset {args.dataset}...")
        if DATASET_CONFIGS[args.dataset].get('subset'):
            dataset = load_dataset(DATASET_CONFIGS[args.dataset]['name'],
                                 DATASET_CONFIGS[args.dataset]['subset'])
        else:
            dataset = load_dataset(DATASET_CONFIGS[args.dataset]['name'])
        print(f"Successfully downloaded dataset {args.dataset}")
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        raise

def download_model(args):
    """Download model and cache it."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
        print(f"Downloading model {args.model}...")
        model_name = MODEL_CONFIGS[args.model]['name']
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Download appropriate model based on task
        if args.task == 'classification':
            if not args.dataset:
                # For classification tasks, we need the dataset to know num_labels
                print("Warning: Using default 2 labels for classification as no dataset specified")
                num_labels = 2
            else:
                num_labels = DATASET_CONFIGS[args.dataset]['num_labels']
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print(f"Successfully downloaded model {args.model}")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        raise

def main():
    # Set tokenizer parallelism if not already set
    if 'TOKENIZERS_PARALLELISM' not in os.environ:
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Get results directory from environment or use default
    results_dir = os.environ.get('RESULTS_DIR', 'results')
    
    # Directories are created by config.py
    
    parser = argparse.ArgumentParser(description='LLM Fine-tuning Project')
    parser.add_argument('--mode', choices=['evaluate', 'train', 'download'], required=True,
                      help='Mode to run in')
    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()),
                      help='Model to use')
    parser.add_argument('--dataset', choices=list(DATASET_CONFIGS.keys()),
                      help='Dataset to use')
    parser.add_argument('--task', choices=['classification', 'language-modeling', 'summarization'],
                      help='Task type for model configuration')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float,
                      help='Learning rate (if None, use config default)')
    parser.add_argument('--eval_steps', type=int, default=1,
                      help='Number of steps between evaluations')
    parser.add_argument('--save_steps', type=int, default=1,
                      help='Number of steps between model saves')

    parser.add_argument('--is-baseline', action='store_true',
                      help='Whether this is a baseline evaluation')
    parser.add_argument('--experiment-name', type=str,
                      help='Name of the experiment')
    
    args = parser.parse_args()
    
    if args.mode == 'download':
        if args.dataset:
            download_data(args)
        if args.model:
            if not args.task:
                parser.error("--task is required when downloading a model")
            if args.task == 'classification' and not args.dataset:
                print("Warning: For classification tasks, specifying a dataset is recommended")
            download_model(args)
    elif args.mode == 'evaluate':
        if not (args.model and args.dataset):
            parser.error("--model and --dataset are required for evaluation")
        run_baseline_evaluation(args)
    else:
        if not (args.model and args.dataset):
            parser.error("--model and --dataset are required for training")
        run_training(args)

if __name__ == '__main__':
    main()
