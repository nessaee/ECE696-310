"""
Dataset handling module for managing dataset loading, preprocessing, and analysis.

This module provides a unified interface for:
1. Loading and caching datasets
2. Preprocessing text data for different tasks
3. Creating data loaders with appropriate collation
4. Analyzing dataset statistics and generating reports
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
import torch
from transformers import PreTrainedTokenizer, DataCollatorWithPadding, DataCollatorForLanguageModeling

from ..utils.config import DATASET_CONFIGS, DATA_DIR

# Set up logging
logger = logging.getLogger(__name__)

class DatasetHandler:
    """Handles dataset loading, preprocessing, analysis, and dataloader creation."""
    
    def __init__(self, dataset_name: str, tokenizer: PreTrainedTokenizer):
        # Ensure tokenizer parallelism is set
        if 'TOKENIZERS_PARALLELISM' not in os.environ:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        """
        Initialize dataset handler.
        
        Args:
            dataset_name: Name of the dataset to load
            tokenizer: Tokenizer to use for preprocessing
            
        Raises:
            ValueError: If dataset_name is not supported
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(DATASET_CONFIGS.keys())}")
        
        self.config = DATASET_CONFIGS[dataset_name]
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        
        # Set up caching
        self.cache_dir = DATA_DIR / dataset_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and validate dataset
        self.dataset = self._load_dataset()
        self._validate_dataset()
        self._save_dataset_info()
    
    def _load_dataset(self) -> DatasetDict:
        """Load dataset from cache or download if not available."""
        cache_dir = self.cache_dir / 'cache'
        
        if cache_dir.exists():
            logger.info(f"Loading cached dataset from {cache_dir}")
            try:
                return DatasetDict.load_from_disk(str(cache_dir))
            except Exception as e:
                logger.warning(f"Failed to load cached dataset: {e}. Downloading fresh copy.")
        
        logger.info(f"Downloading dataset {self.dataset_name}")
        try:
            # Load dataset
            if 'subset' in self.config:
                raw_dataset = load_dataset(self.config['name'], self.config['subset'])
            else:
                raw_dataset = load_dataset(self.config['name'])
            
            # Convert to DatasetDict if needed
            if not isinstance(raw_dataset, DatasetDict):
                dataset = DatasetDict(raw_dataset)
            else:
                dataset = raw_dataset
            
            # Create validation split if it doesn't exist
            if 'validation' not in dataset:
                logger.info("Creating validation split")
                train_valid = dataset['train'].train_test_split(test_size=0.1, seed=42)
                dataset['train'] = train_valid['train']
                dataset['validation'] = train_valid['test']
            
            # Save to cache
            dataset.save_to_disk(str(cache_dir))
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise RuntimeError(f"Failed to load dataset {self.dataset_name}: {str(e)}")
    
    def _validate_dataset(self):
        """Validate dataset structure and contents."""
        required_splits = ['train', 'test']
        missing_splits = [split for split in required_splits if split not in self.dataset]
        if missing_splits:
            raise ValueError(f"Dataset missing required splits: {missing_splits}")
        
        # Validate required columns
        required_columns = [self.config['text_column']]
        if self.config['task'] == 'classification':
            required_columns.append(self.config['label_column'])
        elif self.config['task'] == 'summarization':
            required_columns.append(self.config['summary_column'])
        
        for split in self.dataset:
            missing_columns = [col for col in required_columns if col not in self.dataset[split].column_names]
            if missing_columns:
                raise ValueError(f"Split {split} missing required columns: {missing_columns}")
        
        # Validate data types and content
        self._validate_split('train')
        self._validate_split('test')
    
    def _validate_split(self, split: str):
        """Validate specific dataset split."""
        dataset = self.dataset[split]
        
        # Check for empty texts
        text_column = self.config['text_column']
        empty_texts = [i for i, text in enumerate(dataset[text_column]) if not text.strip()]
        if empty_texts:
            logger.warning(f"Found {len(empty_texts)} empty texts in {split} split")
        
        # Validate labels for classification
        if self.config['task'] == 'classification':
            label_column = self.config['label_column']
            unique_labels = set(dataset[label_column])
            expected_labels = set(range(self.config['num_labels']))
            if not unique_labels.issubset(expected_labels):
                raise ValueError(f"Invalid labels in {split} split. Found {unique_labels}, expected subset of {expected_labels}")
    
    def _save_dataset_info(self):
        """Save dataset information and statistics."""
        info = {
            'name': self.dataset_name,
            'task': self.config['task'],
            'splits': {}
        }
        
        for split in self.dataset:
            split_info = {
                'num_examples': len(self.dataset[split]),
                'columns': self.dataset[split].column_names
            }
            
            # Calculate text length statistics
            text_lengths = [len(text.split()) for text in self.dataset[split][self.config['text_column']]]
            split_info['text_length_stats'] = {
                'min': min(text_lengths),
                'max': max(text_lengths),
                'mean': sum(text_lengths) / len(text_lengths)
            }
            
            # Add task-specific statistics
            if self.config['task'] == 'classification':
                label_counts = pd.Series(self.dataset[split][self.config['label_column']]).value_counts().to_dict()
                split_info['label_distribution'] = label_counts
            
            info['splits'][split] = split_info
        
        # Save info
        info_file = self.cache_dir / 'dataset_info.json'
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Dataset info saved to {info_file}")
    
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
    
    def decode_batch(self, input_ids) -> list:
        """Decode a batch of input IDs back to text.
        
        Args:
            input_ids: Tensor of input IDs from the tokenizer
            
        Returns:
            List of decoded text strings
        """
        # Convert to list if tensor
        if hasattr(input_ids, 'cpu'):
            input_ids = input_ids.cpu().tolist()
            
        # Decode each sequence, skipping special tokens
        decoded = []
        for seq in input_ids:
            # Skip padding tokens
            if isinstance(seq, list):
                seq = [id for id in seq if id != self.tokenizer.pad_token_id]
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            decoded.append(text)
            
        return decoded
    
    def get_dataloader(self, split: str = 'train', shuffle: bool = True) -> DataLoader:
        """
        Get dataloader for specified split.
        
        Args:
            split: Dataset split ('train', 'test', 'validation')
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for the specified split
            
        Raises:
            ValueError: If split is not found in dataset
        """
        if split not in self.dataset:
            raise ValueError(f"Split {split} not found. Available splits: {list(self.dataset.keys())}")
            
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
            remove_columns=columns_to_remove,
            desc=f"Preprocessing {split} split"
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
            collate_fn=collate_fn,
            num_workers=os.cpu_count() or 1,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information and statistics."""
        info_file = self.cache_dir / 'dataset_info.json'
        if not info_file.exists():
            raise FileNotFoundError(f"Dataset info file not found: {info_file}")
        
        with open(info_file, 'r') as f:
            return json.load(f)
    
    def get_sample(self, split: str, n: int = 5) -> List[Dict[str, Any]]:
        """Get sample examples from the dataset.
        
        Args:
            split: Dataset split to sample from
            n: Number of samples to return
            
        Returns:
            List of sample examples with text and labels/summaries
            
        Raises:
            ValueError: If split is not found in dataset
        """
        if split not in self.dataset:
            raise ValueError(f"Split {split} not found in dataset")
        
        samples = []
        for i in range(min(n, len(self.dataset[split]))):
            example = self.dataset[split][i]
            sample = {
                'text': example[self.config['text_column']],
            }
            
            if self.config['task'] == 'classification':
                sample['label'] = example[self.config['label_column']]
            elif self.config['task'] == 'summarization':
                sample['summary'] = example[self.config['summary_column']]
            
            samples.append(sample)
        
        return samples
