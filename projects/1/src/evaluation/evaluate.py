"""
Script for evaluating model performance on different tasks.
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import math
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from data.config import MODEL_CONFIGS, DATASET_CONFIGS

class BaselineEvaluator:
    def __init__(self, model_name, dataset_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.device = device
        self.model_config = MODEL_CONFIGS[model_name]
        self.dataset_config = DATASET_CONFIGS[dataset_name]
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load appropriate model based on task
        if self.dataset_config['task'] == 'classification':
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=self.dataset_config['num_labels']
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        self.model.to(device)
        self.model.eval()
        
        # Load dataset
        self.load_dataset()

    def load_dataset(self):
        """Load and preprocess the dataset based on the task."""
        if self.dataset_name == 'imdb':
            self.dataset = load_dataset('imdb')
            self.preprocess_fn = self.preprocess_classification
        elif self.dataset_name == 'wikitext':
            self.dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
            self.preprocess_fn = self.preprocess_language_modeling
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported yet")

    def preprocess_classification(self, examples):
        """Preprocess classification dataset."""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=self.dataset_config['max_length'],
            padding='max_length'
        )

    def preprocess_language_modeling(self, examples):
        """Preprocess language modeling dataset."""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=self.model_config['max_length']
        )

    def prepare_dataloader(self, split='test'):
        """Prepare dataloader for evaluation."""
        dataset = self.dataset[split]
        dataset = dataset.map(
            self.preprocess_fn,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        if self.dataset_config['task'] == 'classification':
            dataset = dataset.map(
                lambda x: {'labels': x['label']},
                remove_columns=['label']
            )
            collate_fn = DataCollatorWithPadding(self.tokenizer)
        else:
            collate_fn = DataCollatorForLanguageModeling(
                self.tokenizer,
                mlm=False
            )
        
        return DataLoader(
            dataset,
            batch_size=self.dataset_config['batch_size'],
            collate_fn=collate_fn
        )

    def evaluate_classification(self):
        """Evaluate classification performance."""
        dataloader = self.prepare_dataloader('test')
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                predictions = outputs.logits.argmax(-1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        report = classification_report(all_labels, all_preds)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report
        }

    def evaluate_language_modeling(self):
        """Evaluate language modeling performance (perplexity)."""
        dataloader = self.prepare_dataloader('test')
        total_loss = 0
        total_length = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item() * batch['input_ids'].size(0)
                total_length += batch['input_ids'].size(0)
        
        avg_loss = total_loss / total_length
        perplexity = math.exp(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity
        }

    def evaluate(self):
        """Main evaluation method."""
        if self.dataset_config['task'] == 'classification':
            results = self.evaluate_classification()
        else:
            results = self.evaluate_language_modeling()
        
        # Save results
        output_dir = Path(__file__).parent.parent.parent / 'results'
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / f'baseline_{self.model_name}_{self.dataset_name}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate baseline model performance')
    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()), required=True)
    parser.add_argument('--dataset', choices=list(DATASET_CONFIGS.keys()), required=True)
    
    args = parser.parse_args()
    
    evaluator = BaselineEvaluator(args.model, args.dataset)
    results = evaluator.evaluate()
    print("\nEvaluation Results:")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
