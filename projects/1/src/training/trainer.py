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
import os
class Trainer:
    """Handles model training and fine-tuning."""
    
    def __init__(
        self,
        model_handler: ModelHandler,
        dataset_handler: DatasetHandler,
        experiment_name: Optional[str] = None,
        evaluator: Optional[Evaluator] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model_handler: Initialized ModelHandler instance
            dataset_handler: Initialized DatasetHandler instance
            experiment_name: Optional name for the experiment
            evaluator: Optional evaluator instance for evaluation during training
        """
        # Store handlers
        self.model_handler = model_handler
        self.dataset_handler = dataset_handler
        self.evaluator = evaluator
        
        # Get model name and experiment name
        self.model_name = model_handler.model_config['name']
        self.experiment_name = experiment_name or f"{self.model_name}_{dataset_handler.config['name']}"
        
        # Setup timestamp
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get output directory from environment or use default
        self.base_dir = Path(os.environ.get('RESULTS_DIR', RESULTS_DIR))
        self.train_dir = self.base_dir
        
        # Setup subdirectories
        self.metrics_dir = self.train_dir / 'metrics'
        self.analysis_dir = self.train_dir / 'analysis'
        self.weights_dir = self.train_dir / 'weights'
        
        # Create directories
        for dir_path in [self.metrics_dir, self.weights_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(
            experiment_name='train',
            output_dir=self.train_dir,
            model_name=self.model_name
        )
            
        # Setup logging
        self.logger = setup_logging(self.experiment_name, log_dir=self.analysis_dir)
        
        # Initialize components
        self.device = model_handler.device
        self.model = model_handler.model
        self.task = dataset_handler.config['task']
        
        # Save experiment config
        self.config = {
            'model_name': self.model_name,
            'dataset': dataset_handler.config['name'],
            'task': self.task,
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        # Save config and log
        config_path = self.analysis_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        self.logger.info(f"Initialized trainer for experiment: {self.experiment_name}")
        self.logger.info(f"Output directory: {self.train_dir}")
        self.logger.info(f"Config: {json.dumps(self.config, indent=4)}")
    
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
                    
                    # Initialize sample variables
                    sample_inputs = None
                    sample_outputs = None
                    accuracy = None
                    
                    # Calculate additional metrics for classification
                    if self.task == 'classification':
                        logits = outputs.logits
                        predictions = torch.argmax(logits, dim=-1)
                        labels = batch['labels']
                        
                        # Accuracy
                        correct = (predictions == labels).float().sum()
                        total = labels.size(0)
                        accuracy = correct / total
                        
                        # Store predictions and labels for F1, precision, recall
                        if 'predictions' not in metrics:
                            metrics['predictions'] = []
                            metrics['labels'] = []
                        metrics['predictions'].extend(predictions.cpu().tolist())
                        metrics['labels'].extend(labels.cpu().tolist())
                        
                        # Store input/output samples (first batch only)
                        if step == 0:
                            try:
                                sample_inputs = {
                                    'text': self.dataset_handler.decode_batch(batch['input_ids'][:3]),
                                    'labels': labels[:3].cpu().tolist()
                                }
                                sample_outputs = {
                                    'predictions': predictions[:3].cpu().tolist(),
                                    'logits': logits[:3].detach().cpu().tolist()
                                }
                            except Exception as e:
                                self.logger.warning(f"Could not create samples: {str(e)}")
                    
                    # Update progress bar
                    current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
                    step_metrics = {
                        'loss': loss.item(),
                        'avg_loss': metrics['loss'] / metrics['steps'],
                        'learning_rate': current_lr
                    }
                    if accuracy is not None:
                        step_metrics['accuracy'] = accuracy.item()
                    
                    pbar.set_postfix(step_metrics)
                    
                    # Log metrics and samples
                    self.metrics_tracker.log_training_step(step_metrics, step, epoch, 
                                                         inputs=sample_inputs, 
                                                         outputs=sample_outputs)
                        
                except Exception as e:
                    self.logger.error(f"Error in training step {step}: {str(e)}")
                    continue
        
        # Compute epoch metrics
        metrics['avg_loss'] = metrics['loss'] / metrics['steps']
        metrics['time'] = time.time() - start_time
        
        # Calculate final classification metrics
        if self.task == 'classification' and 'predictions' in metrics:
            from sklearn.metrics import precision_recall_fscore_support, accuracy_score
            
            y_true = metrics.pop('labels')
            y_pred = metrics.pop('predictions')
            
            # Calculate all metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            # Calculate precision, recall, f1 for each averaging method
            for average in ['micro', 'macro', 'weighted']:
                p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)
                metrics[f'precision_{average}'] = p
                metrics[f'recall_{average}'] = r
                metrics[f'f1_{average}'] = f1
        
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
        

        
        best_metric = float('inf')
        metrics = {}
        
        for epoch in range(num_epochs):
            # Train for one epoch
            epoch_metrics = self.train_epoch(train_dataloader, optimizer, scheduler, epoch)
            metrics[f'epoch_{epoch}'] = epoch_metrics
            
            # Evaluate if needed
            if self.evaluator and eval_steps and (epoch + 1) % eval_steps == 0:
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
        if self.evaluator:
            try:
                final_metrics = self.evaluator.evaluate('test')
                metrics['final_test'] = final_metrics
            except Exception as e:
                self.logger.error(f"Final evaluation failed: {str(e)}")
                self.logger.warning("Training completed but evaluation could not be performed.")
        
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
        checkpoint_path = self.weights_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save metrics
        metrics_path = self.metrics_dir / f'metrics_epoch_{epoch}.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics or {}, f, indent=2)
        
        # Save best model if needed
        if is_best:
            best_path = self.weights_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")
            
            # Save best metrics
            best_metrics_path = self.metrics_dir / 'best_metrics.json'
            with open(best_metrics_path, 'w') as f:
                json.dump(metrics or {}, f, indent=2)
        
        # Cleanup old checkpoints (keep only last 3)
        checkpoints = sorted(self.weights_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > 3:
            for checkpoint in checkpoints[:-3]:
                checkpoint.unlink()
                # Also remove corresponding metrics
                metrics_file = self.metrics_dir / f'metrics_epoch_{checkpoint.stem.split("_")[-1]}.json'
                if metrics_file.exists():
                    metrics_file.unlink()
                self.logger.info(f"Removed old checkpoint and metrics: {checkpoint}")
