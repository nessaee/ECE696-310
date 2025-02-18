"""
Main script for running experiments.
"""
import argparse
from pathlib import Path
import json

from src.models.model_handler import ModelHandler
from src.data.dataset import DatasetHandler
from src.evaluation.evaluator import Evaluator
from src.training.trainer import Trainer
from src.utils.config import MODEL_CONFIGS, DATASET_CONFIGS

def run_baseline_evaluation(args):
    """Run baseline model evaluation."""
    # Initialize handlers
    model_handler = ModelHandler(args.model, args.dataset)
    dataset_handler = DatasetHandler(args.dataset, model_handler.tokenizer)
    evaluator = Evaluator(model_handler, dataset_handler)
    
    # Run evaluation
    results = evaluator.evaluate('test')
    print("\nBaseline Evaluation Results:")
    print(json.dumps(results, indent=2))

def run_training(args):
    """Run model fine-tuning."""
    # Initialize handlers
    model_handler = ModelHandler(args.model, args.dataset)
    dataset_handler = DatasetHandler(args.dataset, model_handler.tokenizer)
    evaluator = Evaluator(model_handler, dataset_handler)
    
    # Initialize trainer
    trainer = Trainer(
        model_handler,
        dataset_handler,
        evaluator,
        use_wandb=args.use_wandb
    )
    
    # Run training
    metrics = trainer.train(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps
    )
    
    print("\nTraining Results:")
    print(json.dumps(metrics, indent=2))

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
    parser.add_argument('--use_wandb', action='store_true',
                      help='Use Weights & Biases for tracking')
    
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
