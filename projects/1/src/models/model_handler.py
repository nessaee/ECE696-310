"""
Model handling module.
"""
from typing import Optional, Union, Type
import torch
from transformers import (
    PreTrainedModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer
)

from src.utils.config import MODEL_CONFIGS, DATASET_CONFIGS, MODELS_DIR

class ModelHandler:
    """Handles model loading, saving, and configuration."""
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        device: Optional[str] = None
    ):
        """
        Initialize model handler.
        
        Args:
            model_name: Name of the model to load
            dataset_name: Name of the dataset (for task-specific configuration)
            device: Device to load model on ('cuda' or 'cpu')
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model {model_name} not supported")
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        self.model_config = MODEL_CONFIGS[model_name]
        self.dataset_config = DATASET_CONFIGS[dataset_name]
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = self._load_tokenizer(model_name)
        self.model = self._load_model(model_name)
    
    def _load_tokenizer(self, model_name: str) -> PreTrainedTokenizer:
        """Load and configure tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_config['name'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def _load_model(self, model_name: str) -> PreTrainedModel:
        """Load and configure model based on task."""
        model_path = MODELS_DIR / model_name
        
        # Ensure model knows about padding token
        model_kwargs = {
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        
        # Add task-specific configuration
        if self.dataset_config['task'] == 'classification':
            model_class = AutoModelForSequenceClassification
            model_kwargs['num_labels'] = self.dataset_config['num_labels']
        else:
            model_class = AutoModelForCausalLM
        
        # Load model (from local path if available, otherwise from HuggingFace)
        try:
            model = model_class.from_pretrained(model_path, **model_kwargs)
        except:
            model = model_class.from_pretrained(self.model_config['name'], **model_kwargs)
        
        # Ensure model config has padding token
        model.config.pad_token_id = self.tokenizer.pad_token_id
        
        return model.to(self.device)
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save model and tokenizer."""
        save_path = output_dir or MODELS_DIR / self.model_config['name']
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def get_optimizer(self, learning_rate: Optional[float] = None):
        """Get optimizer for the model."""
        learning_rate = learning_rate or self.model_config['learning_rate']
        return torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
