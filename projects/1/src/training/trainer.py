"""
Model training module with robust checkpointing and logging.

This module handles model training, evaluation, and checkpointing with comprehensive
logging and error handling. It supports:
- Automatic checkpointing and model saving
- Integration with Weights & Biases
- Graceful error recovery
- Detailed progress logging
"""
from typing import Optional, Dict, Any
import torch
from tqdm import tqdm
import wandb
from pathlib import Path
import logging
import json
import time
from datetime import datetime

from src.models.model_handler import ModelHandler
from src.data.dataset import DatasetHandler
from src.evaluation.evaluator import Evaluator
from src.utils.config import MODELS_DIR, RESULTS_DIR
from src.utils.logging_config import setup_logging
from src.utils.metrics_tracker import MetricsTracker

class Trainer:
    """Handles model training and fine-tuning."""
    
    def __init__(
        self,
        model_handler: ModelHandler,
        dataset_handler: DatasetHandler,
        evaluator: Evaluator,
        use_wandb: bool = False,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model_handler: Initialized ModelHandler instance
            dataset_handler: Initialized DatasetHandler instance
            evaluator: Initialized Evaluator instance
            use_wandb: Whether to use Weights & Biases for tracking
        """
        # Setup logging
        self.experiment_name = experiment_name or f"{model_handler.model_config['name']}_{dataset_handler.config['name']}_{int(time.time())}"
        self.logger = setup_logging(self.experiment_name)
        
        # Initialize components
        self.model_handler = model_handler
        self.dataset_handler = dataset_handler
        self.evaluator = evaluator
        self.use_wandb = use_wandb
        
        self.device = model_handler.device
        self.model = model_handler.model
        self.task = dataset_handler.config['task']
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(self.experiment_name, RESULTS_DIR)
        
        # Create checkpoint directory
        self.checkpoint_dir = MODELS_DIR / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment config
        self.config = {
            'model': model_handler.model_config['name'],
            'dataset': dataset_handler.config['name'],
            'task': self.task,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.checkpoint_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Initialized trainer for experiment: {self.experiment_name}")
        self.logger.info(f"Config: {json.dumps(self.config, indent=2)}")
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch with detailed logging and error handling.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer instance
            scheduler: Optional learning rate scheduler
            epoch: Current epoch number
            
        Returns:
            Dictionary containing training metrics for the epoch
        """
        self.model.train()
        metrics = {'loss': 0.0, 'steps': 0}
        start_time = time.time()
        
        with tqdm(dataloader, desc=f"Training epoch {epoch}") as pbar:
            for step, batch in enumerate(pbar):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if scheduler is not None:
                        scheduler.step()
                    
                    # Update metrics
                    metrics['loss'] += loss.item()
                    metrics['steps'] += 1
                    
                    # Update progress bar
                    current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
                    step_metrics = {
                        'loss': loss.item(),
                        'avg_loss': metrics['loss'] / metrics['steps'],
                        'learning_rate': current_lr
                    }
                    
                    pbar.set_postfix(step_metrics)
                    
                    # Log metrics
                    self.metrics_tracker.log_training_step(step_metrics, step, epoch)
                    
                    # Log to wandb
                    if self.wandb_enabled:
                        wandb.log({
                            'train/loss': loss.item(),
                            'train/avg_loss': metrics['loss'] / metrics['steps'],
                            'train/learning_rate': current_lr,
                            'epoch': epoch,
                            'step': step
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error in training step {step}: {str(e)}")
                    continue
        
        # Compute epoch metrics
        metrics['avg_loss'] = metrics['loss'] / metrics['steps']
        metrics['time'] = time.time() - start_time
        
        # Log epoch metrics
        self.metrics_tracker.log_epoch(metrics, epoch)
        
        self.logger.info(f"Epoch {epoch} completed in {metrics['time']:.2f}s with avg_loss={metrics['avg_loss']:.4f}")
        
        return metrics
    
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
        
        # Setup learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate or self.model_handler.model_config['learning_rate'],
            epochs=num_epochs,
            steps_per_epoch=len(train_dataloader),
            pct_start=0.1,  # Warm up for 10% of training
            anneal_strategy='cos'
        )
        
        # Initialize wandb if requested and API key is configured
        self.wandb_enabled = False
        if self.use_wandb:
            try:
                wandb.init(
                    project="llm-finetuning",
                    name=self.experiment_name,
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
                self.logger.info("Successfully initialized Weights & Biases logging")
            except Exception as e:
                self.logger.warning(f"Could not initialize Weights & Biases: {str(e)}")
                self.logger.warning("Training will continue without W&B logging")
        
        best_metric = float('inf')
        metrics = {}
        
        for epoch in range(num_epochs):
            # Train for one epoch
            epoch_metrics = self.train_epoch(train_dataloader, optimizer, scheduler, epoch)
            metrics[f'epoch_{epoch}'] = epoch_metrics
            
            # Evaluate if needed
            if eval_steps and (epoch + 1) % eval_steps == 0:
                try:
                    eval_metrics = self.evaluator.evaluate('validation')
                    metrics[f'epoch_{epoch}_eval'] = eval_metrics
                    
                    # Save best model
                    current_metric = eval_metrics.get('loss', eval_metrics.get('perplexity', float('inf')))
                    if current_metric < best_metric:
                        best_metric = current_metric
                        self.save_checkpoint(
                            epoch=epoch,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            metrics=eval_metrics,
                            is_best=True
                        )
                except Exception as e:
                    self.logger.error(f"Evaluation failed: {str(e)}")
                    self.logger.warning("Continuing training without evaluation...")
            
            # Save checkpoint if needed
            if save_steps and (epoch + 1) % save_steps == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics=epoch_metrics
                )
        
        # Final evaluation
        try:
            final_metrics = self.evaluator.evaluate('test')
            metrics['final_test'] = final_metrics
        except Exception as e:
            self.logger.error(f"Final evaluation failed: {str(e)}")
            self.logger.warning("Training completed but evaluation could not be performed.")
        
        if self.wandb_enabled:
            wandb.log(metrics)
            wandb.finish()
        
        return metrics
    
    def save_checkpoint(self, epoch: int, optimizer: torch.optim.Optimizer,
                     scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                     metrics: Optional[Dict] = None, is_best: bool = False):
        """Save a complete training checkpoint.
        
        Args:
            epoch: Current epoch number
            optimizer: Current optimizer state
            scheduler: Optional scheduler state
            metrics: Optional metrics to save
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics or {},
            'config': self.config
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model if needed
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")
        
        # Cleanup old checkpoints (keep only last 3)
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > 3:
            for checkpoint in checkpoints[:-3]:
                checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {checkpoint}")
