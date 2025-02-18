"""
Data management module for handling dataset loading, validation, and caching.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
import hashlib
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import logging
from tqdm import tqdm

from ..utils.config import DATASET_CONFIGS, DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """Handles dataset loading, validation, and caching."""
    
    def __init__(self, dataset_name: str):
        """
        Initialize data manager.
        
        Args:
            dataset_name: Name of the dataset to manage
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Dataset {dataset_name} not supported. Available datasets: {list(DATASET_CONFIGS.keys())}")
        
        self.config = DATASET_CONFIGS[dataset_name]
        self.dataset_name = dataset_name
        self.cache_dir = DATA_DIR / dataset_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or download dataset
        self.dataset = self._load_dataset()
        
        # Validate and process dataset
        self._validate_dataset()
        self._save_dataset_info()
    
    def _load_dataset(self) -> DatasetDict:
        """Load dataset from cache or download if not available."""
        cache_file = self.cache_dir / 'dataset_cache.json'
        
        if cache_file.exists():
            logger.info(f"Loading cached dataset from {cache_file}")
            try:
                return DatasetDict.load_from_disk(str(self.cache_dir / 'cache'))
            except Exception as e:
                logger.warning(f"Failed to load cached dataset: {e}. Downloading fresh copy.")
        
        logger.info(f"Downloading dataset {self.dataset_name}")
        try:
            if 'subset' in self.config:
                dataset = load_dataset(self.config['name'], self.config['subset'])
            else:
                dataset = load_dataset(self.config['name'])
            
            # Save to cache
            dataset.save_to_disk(str(self.cache_dir / 'cache'))
            return dataset
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {self.dataset_name}: {e}")
    
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
    
    def get_data_loader_kwargs(self, split: str) -> Dict[str, Any]:
        """Get kwargs for creating data loader."""
        return {
            'batch_size': self.config['batch_size'],
            'shuffle': split == 'train',
            'num_workers': os.cpu_count() or 1,
            'pin_memory': torch.cuda.is_available()
        }
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information and statistics."""
        info_file = self.cache_dir / 'dataset_info.json'
        if not info_file.exists():
            raise FileNotFoundError(f"Dataset info file not found: {info_file}")
        
        with open(info_file, 'r') as f:
            return json.load(f)
    
    def get_sample(self, split: str, n: int = 5) -> List[Dict[str, Any]]:
        """Get sample examples from the dataset."""
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
