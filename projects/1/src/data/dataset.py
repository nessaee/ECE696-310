"""
Dataset handling module.
"""
from typing import Optional, Dict, Any
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, DataCollatorWithPadding, DataCollatorForLanguageModeling

from ..utils.config import DATASET_CONFIGS

class DatasetHandler:
    """Handles dataset loading, preprocessing, and dataloader creation."""
    
    def __init__(self, dataset_name: str, tokenizer: PreTrainedTokenizer):
        """
        Initialize dataset handler.
        
        Args:
            dataset_name: Name of the dataset to load
            tokenizer: Tokenizer to use for preprocessing
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        self.config = DATASET_CONFIGS[dataset_name]
        self.tokenizer = tokenizer
        self.dataset = self._load_dataset()
    
    def _load_dataset(self):
        """Load the dataset based on configuration."""
        if 'subset' in self.config:
            dataset = load_dataset(self.config['name'], self.config['subset'])
        else:
            dataset = load_dataset(self.config['name'])
            
        # Create validation split if it doesn't exist
        if 'validation' not in dataset:
            # Split training data into train and validation
            train_valid = dataset['train'].train_test_split(test_size=0.1, seed=42)
            dataset = dataset.copy()
            dataset['train'] = train_valid['train']
            dataset['validation'] = train_valid['test']
            
        return dataset
    
    def _preprocess_classification(self, examples: Dict[str, Any]):
        """Preprocess classification dataset."""
        # Tokenize the texts
        tokenized = self.tokenizer(
            examples[self.config['text_column']],
            truncation=True,
            max_length=self.config['max_length'],
            padding='max_length'
        )
        
        # Convert labels to list for batched processing
        if isinstance(examples[self.config['label_column']], (list, np.ndarray)):
            labels = examples[self.config['label_column']]
        else:
            labels = [examples[self.config['label_column']]]
            
        tokenized['labels'] = labels
        return tokenized
    
    def _preprocess_language_modeling(self, examples: Dict[str, Any]):
        """Preprocess language modeling dataset."""
        return self.tokenizer(
            examples[self.config['text_column']],
            truncation=True,
            max_length=self.config['max_length']
        )
    
    def _preprocess_summarization(self, examples: Dict[str, Any]):
        """Preprocess summarization dataset."""
        model_inputs = self.tokenizer(
            examples[self.config['text_column']],
            truncation=True,
            max_length=self.config['max_length'],
            padding='max_length'
        )
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples[self.config['summary_column']],
                truncation=True,
                max_length=self.config['target_max_length'],
                padding='max_length'
            )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    def get_dataloader(self, split: str = 'train', shuffle: bool = True) -> DataLoader:
        """
        Get dataloader for specified split.
        
        Args:
            split: Dataset split ('train', 'test', 'validation')
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for the specified split
        """
        dataset = self.dataset[split]
        
        # Select appropriate preprocessing function
        if self.config['task'] == 'classification':
            preprocess_fn = self._preprocess_classification
        elif self.config['task'] == 'language-modeling':
            preprocess_fn = self._preprocess_language_modeling
        else:
            preprocess_fn = self._preprocess_summarization
        
        # Preprocess dataset
        if self.config['task'] == 'classification':
            # For classification, keep the label column during initial preprocessing
            columns_to_remove = [col for col in dataset.column_names if col != self.config['label_column']]
        else:
            columns_to_remove = dataset.column_names
            
        processed_dataset = dataset.map(
            preprocess_fn,
            batched=True,
            remove_columns=columns_to_remove
        )
        
        # Select appropriate collate function
        if self.config['task'] == 'language-modeling':
            collate_fn = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        else:
            collate_fn = DataCollatorWithPadding(self.tokenizer)
        
        return DataLoader(
            processed_dataset,
            batch_size=self.config['batch_size'],
            shuffle=shuffle,
            collate_fn=collate_fn
        )
