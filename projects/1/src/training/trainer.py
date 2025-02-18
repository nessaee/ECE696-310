"""
Model training module.
"""
from typing import Optional, Dict, Any
import torch
from tqdm import tqdm
import wandb
from pathlib import Path

from src.models.model_handler import ModelHandler
from src.data.dataset import DatasetHandler
from src.evaluation.evaluator import Evaluator
from src.utils.config import MODELS_DIR

class Trainer:
    """Handles model training and fine-tuning."""
    
    def __init__(
        self,
        model_handler: ModelHandler,
        dataset_handler: DatasetHandler,
        evaluator: Evaluator,
        use_wandb: bool = False
    ):
        """
        Initialize trainer.
        
        Args:
            model_handler: Initialized ModelHandler instance
            dataset_handler: Initialized DatasetHandler instance
            evaluator: Initialized Evaluator instance
            use_wandb: Whether to use Weights & Biases for tracking
        """
        self.model_handler = model_handler
        self.dataset_handler = dataset_handler
        self.evaluator = evaluator
        self.use_wandb = use_wandb
        
        self.device = model_handler.device
        self.model = model_handler.model
        self.task = dataset_handler.config['task']
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        with tqdm(dataloader, desc=f"Training epoch {epoch}") as pbar:
            for step, batch in enumerate(pbar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                if self.wandb_enabled:
                    wandb.log({
                        'train_loss': loss.item(),
                        'epoch': epoch,
                        'step': step
                    })
        
        return total_loss / len(dataloader)
    
    def train(
        self,
        num_epochs: int,
        learning_rate: Optional[float] = None,
        eval_steps: Optional[int] = None,
        save_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            learning_rate: Learning rate (if None, use config default)
            eval_steps: Number of steps between evaluations
            save_steps: Number of steps between model saves
            
        Returns:
            Dictionary containing training metrics
        """
        train_dataloader = self.dataset_handler.get_dataloader('train')
        optimizer = self.model_handler.get_optimizer(learning_rate)
        
        # Initialize wandb if requested and API key is configured
        self.wandb_enabled = False
        if self.use_wandb:
            try:
                wandb.init(
                    project="llm-finetuning",
                    config={
                        "model": self.model_handler.model_config['name'],
                        "dataset": self.dataset_handler.config['name'],
                        "task": self.task,
                        "learning_rate": learning_rate or self.model_handler.model_config['learning_rate'],
                        "num_epochs": num_epochs,
                        "batch_size": self.dataset_handler.config['batch_size']
                    }
                )
                self.wandb_enabled = True
                print("Successfully initialized Weights & Biases logging")
            except Exception as e:
                print(f"Warning: Could not initialize Weights & Biases: {str(e)}")
                print("Training will continue without W&B logging")
        
        best_metric = float('inf')
        metrics = {}
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_dataloader, optimizer, epoch)
            metrics[f'epoch_{epoch}_train_loss'] = train_loss
            
            # Evaluate if needed
            if eval_steps and (epoch + 1) % eval_steps == 0:
                try:
                    eval_metrics = self.evaluator.evaluate('validation')
                    metrics[f'epoch_{epoch}_eval'] = eval_metrics
                    
                    # Save best model
                    current_metric = eval_metrics.get('loss', eval_metrics.get('perplexity', float('inf')))
                    if current_metric < best_metric:
                        best_metric = current_metric
                        self.save_model(is_best=True)
                except Exception as e:
                    print(f"Warning: Evaluation failed: {str(e)}")
                    print("Continuing training without evaluation...")
            
            # Save checkpoint if needed
            if save_steps and (epoch + 1) % save_steps == 0:
                self.save_model(epoch=epoch)
        
        # Final evaluation
        try:
            final_metrics = self.evaluator.evaluate('test')
            metrics['final_test'] = final_metrics
        except Exception as e:
            print(f"Warning: Final evaluation failed: {str(e)}")
            print("Training completed but evaluation could not be performed.")
        
        if self.wandb_enabled:
            wandb.log(metrics)
            wandb.finish()
        
        return metrics
    
    def save_model(self, epoch: Optional[int] = None, is_best: bool = False):
        """Save model checkpoint."""
        model_name = self.model_handler.model_config['name']
        dataset_name = self.dataset_handler.config['name']
        
        if is_best:
            save_path = MODELS_DIR / f"{model_name}_{dataset_name}_best"
        else:
            save_path = MODELS_DIR / f"{model_name}_{dataset_name}_epoch_{epoch}"
        
        self.model_handler.save_model(str(save_path))
