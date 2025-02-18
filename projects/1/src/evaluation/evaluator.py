"""
Model evaluation module.
"""
from typing import Dict, Any
import torch
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import math

from src.models.model_handler import ModelHandler
from src.data.dataset import DatasetHandler
from src.utils.config import RESULTS_DIR

class Evaluator:
    """Handles model evaluation for different tasks."""
    
    def __init__(self, model_handler: ModelHandler, dataset_handler: DatasetHandler):
        """
        Initialize evaluator.
        
        Args:
            model_handler: Initialized ModelHandler instance
            dataset_handler: Initialized DatasetHandler instance
        """
        self.model_handler = model_handler
        self.dataset_handler = dataset_handler
        self.device = model_handler.device
        self.task = dataset_handler.config['task']
    
    def evaluate_classification(self, split: str = 'test') -> Dict[str, Any]:
        """Evaluate classification performance."""
        dataloader = self.dataset_handler.get_dataloader(split, shuffle=False)
        model = self.model_handler.model
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating on {split}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                predictions = outputs.logits.argmax(-1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_score': f1_score(all_labels, all_preds, average='weighted'),
            'classification_report': classification_report(all_labels, all_preds)
        }
    
    def evaluate_language_modeling(self, split: str = 'test') -> Dict[str, float]:
        """Evaluate language modeling performance (perplexity)."""
        dataloader = self.dataset_handler.get_dataloader(split, shuffle=False)
        model = self.model_handler.model
        model.eval()
        
        total_loss = 0
        total_length = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating on {split}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                total_loss += outputs.loss.item() * batch['input_ids'].size(0)
                total_length += batch['input_ids'].size(0)
        
        avg_loss = total_loss / total_length
        perplexity = math.exp(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity
        }
    
    def evaluate_summarization(self, split: str = 'test') -> Dict[str, float]:
        """Evaluate summarization performance (ROUGE scores)."""
        from rouge_score import rouge_scorer
        
        dataloader = self.dataset_handler.get_dataloader(split, shuffle=False)
        model = self.model_handler.model
        model.eval()
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating on {split}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                generated_ids = model.generate(
                    batch['input_ids'],
                    max_length=self.dataset_handler.config['target_max_length'],
                    num_beams=4,
                    early_stopping=True
                )
                
                generated_texts = self.model_handler.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                reference_texts = self.model_handler.tokenizer.batch_decode(
                    batch['labels'], skip_special_tokens=True
                )
                
                batch_scores = [
                    scorer.score(ref, gen)
                    for ref, gen in zip(reference_texts, generated_texts)
                ]
                scores.extend(batch_scores)
        
        # Average scores
        final_scores = {}
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            final_scores[f'{metric}_precision'] = np.mean([s[metric].precision for s in scores])
            final_scores[f'{metric}_recall'] = np.mean([s[metric].recall for s in scores])
            final_scores[f'{metric}_fmeasure'] = np.mean([s[metric].fmeasure for s in scores])
        
        return final_scores
    
    def evaluate(self, split: str = 'test') -> Dict[str, Any]:
        """
        Main evaluation method.
        
        Args:
            split: Dataset split to evaluate on
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.task == 'classification':
            results = self.evaluate_classification(split)
        elif self.task == 'language-modeling':
            results = self.evaluate_language_modeling(split)
        else:
            results = self.evaluate_summarization(split)
        
        # Save results
        model_name = self.model_handler.model_config['name']
        dataset_name = self.dataset_handler.config['name']
        results_file = RESULTS_DIR / f'evaluation_{model_name}_{dataset_name}_{split}.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        return results
