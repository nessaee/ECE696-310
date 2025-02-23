"""
Dataset handler module for managing different datasets
"""
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from src.data.imdb_dataset import get_imdb_loaders
from src.data.wikitext_dataset import get_wikitext_loaders

class DatasetHandler:
    """Handles dataset loading and preprocessing"""
    
    def __init__(self, dataset_name: str, tokenizer: PreTrainedTokenizer, config: Optional[Dict] = None):
        """
        Initialize dataset handler
        
        Args:
            dataset_name: Name of the dataset ('imdb' or 'wikitext-2')
            tokenizer: Tokenizer to use
            config: Optional configuration dictionary
        """
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.config = config or {}
        
        # Set default configuration
        self.config.setdefault('batch_size', 8)
        self.config.setdefault('num_workers', 4)
        self.config.setdefault('max_length', 128)
        
        # Set task type based on dataset
        if dataset_name == 'imdb':
            self.config['task'] = 'classification'
            self.config['num_labels'] = 2
        elif dataset_name == 'wikitext-2':
            self.config['task'] = 'language-modeling'
            self.config['num_labels'] = None  # Not applicable for LM
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Load dataloaders
        self.dataloaders = self._load_dataloaders()
    
    def _load_dataloaders(self) -> Dict[str, DataLoader]:
        """Load appropriate dataloaders based on dataset name"""
        if self.dataset_name == 'imdb':
            return get_imdb_loaders(
                self.tokenizer,
                batch_size=self.config['batch_size'],
                max_length=self.config['max_length'],
                num_workers=self.config['num_workers']
            )
        elif self.dataset_name == 'wikitext-2':
            return get_wikitext_loaders(
                self.tokenizer,
                batch_size=self.config['batch_size'],
                block_size=self.config['max_length'],
                num_workers=self.config['num_workers']
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def get_dataloader(self, split: str) -> DataLoader:
        """Get dataloader for a specific split"""
        if split not in self.dataloaders:
            raise ValueError(f"Unknown split: {split}. Available splits: {list(self.dataloaders.keys())}")
        return self.dataloaders[split]
