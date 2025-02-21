"""
Metrics tracking module for comprehensive performance monitoring.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime
import logging

class MetricsTracker:
    """Tracks and exports training and evaluation metrics."""
    
    def __init__(self, experiment_name: str, output_dir: Path):
        """
        Initialize metrics tracker.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save metrics
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.training_metrics = []
        self.eval_metrics = []
        self.epoch_metrics = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def log_training_step(self, metrics: Dict[str, Any], step: int, epoch: int):
        """Log metrics for a training step."""
        metrics_entry = {
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.training_metrics.append(metrics_entry)
    
    def log_eval_step(self, metrics: Dict[str, Any], step: int, epoch: int):
        """Log metrics for an evaluation step."""
        metrics_entry = {
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.eval_metrics.append(metrics_entry)
    
    def log_epoch(self, metrics: Dict[str, Any], epoch: int):
        """Log metrics for an entire epoch."""
        metrics_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.epoch_metrics.append(metrics_entry)
    
    def export_metrics(self):
        """Export all metrics to CSV files."""
        # Export training metrics
        if self.training_metrics:
            df = pd.DataFrame(self.training_metrics)
            df.to_csv(self.output_dir / 'training_metrics.csv', index=False)
            self.logger.info(f"Exported training metrics to {self.output_dir / 'training_metrics.csv'}")
        
        # Export evaluation metrics
        if self.eval_metrics:
            df = pd.DataFrame(self.eval_metrics)
            df.to_csv(self.output_dir / 'eval_metrics.csv', index=False)
            self.logger.info(f"Exported evaluation metrics to {self.output_dir / 'eval_metrics.csv'}")
        
        # Export epoch metrics
        if self.epoch_metrics:
            df = pd.DataFrame(self.epoch_metrics)
            df.to_csv(self.output_dir / 'epoch_metrics.csv', index=False)
            self.logger.info(f"Exported epoch metrics to {self.output_dir / 'epoch_metrics.csv'}")
        
        # Export summary as JSON
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'num_epochs': len(self.epoch_metrics),
            'total_steps': len(self.training_metrics),
            'final_metrics': self.epoch_metrics[-1] if self.epoch_metrics else None
        }
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
