"""
WikiText-2 dataset loader and processor
"""
from typing import Dict, List, Optional, Union
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

class WikiTextDataset(Dataset):
    """Dataset for WikiText-2 language modeling"""
    
    def __init__(self, split: str, tokenizer: PreTrainedTokenizer, block_size: int = 128):
        """
        Initialize WikiText dataset
        
        Args:
            split: Dataset split ('train', 'validation', or 'test')
            tokenizer: Tokenizer to use
            block_size: Maximum sequence length
        """
        # Load dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split, cache_dir='data')
        
        # Tokenize all texts
        self.examples = []
        for item in dataset:
            if not item['text'].strip():  # Skip empty lines
                continue
            tokenized = tokenizer(item['text'], truncation=True, max_length=block_size,
                                padding='max_length', return_tensors='pt')
            self.examples.append({
                'input_ids': tokenized['input_ids'][0],
                'attention_mask': tokenized['attention_mask'][0],
                'labels': tokenized['input_ids'][0].clone()  # For language modeling, labels are the same as inputs
            })
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]

def get_wikitext_loaders(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    block_size: int = 128,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Get DataLoaders for WikiText-2 dataset
    
    Args:
        tokenizer: Tokenizer to use
        batch_size: Batch size for training
        block_size: Maximum sequence length
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary containing train, validation, and test DataLoaders
    """
    # Create datasets
    train_dataset = WikiTextDataset('train', tokenizer, block_size)
    val_dataset = WikiTextDataset('validation', tokenizer, block_size)
    test_dataset = WikiTextDataset('test', tokenizer, block_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'validation': val_loader,
        'test': test_loader
    }
