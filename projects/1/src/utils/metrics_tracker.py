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
    
    def __init__(self, experiment_name: str, output_dir: Path, model_name: str = None, max_samples: int = 10):
        """
        Initialize metrics tracker.
        
        Args:
            experiment_name: Name of the experiment (e.g., 'baseline_eval', 'train', 'test')
            output_dir: Base directory to save metrics
            model_name: Name of the model being used
            max_samples: Maximum number of input/output samples to store per epoch
        """
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        
        # Initialize metrics storage
        self.training_metrics = []
        self.eval_metrics = []
        self.epoch_metrics = []
        self.samples = []  # Store input/output samples
        
        # Track best metrics
        self.best_metrics = {}
        
        # Initialize metric categories
        self.classification_metrics = {
            'accuracy': [],
            'precision': {'micro': [], 'macro': [], 'weighted': []},
            'recall': {'micro': [], 'macro': [], 'weighted': []},
            'f1': {'micro': [], 'macro': [], 'weighted': []}
        }
        
        # Track timing information
        self.start_time = datetime.now()
        self.epoch_times = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def log_training_step(self, metrics: Dict[str, Any], step: int, epoch: int, inputs: Optional[Dict] = None, outputs: Optional[Dict] = None):
        """
        Log metrics for a training step.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
            epoch: Current epoch
            inputs: Optional dictionary containing input samples
            outputs: Optional dictionary containing model outputs
        """
        metrics_entry = {
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.training_metrics.append(metrics_entry)
        
        # Store sample inputs/outputs if provided
        if inputs is not None and outputs is not None:
            sample = {
                'step': step,
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'inputs': inputs,
                'outputs': outputs
            }
            if len(self.samples) < self.max_samples:
                self.samples.append(sample)
    
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
        now = datetime.now()
        
        # Prepare metrics entry
        metrics_entry = {
            'epoch': epoch,
            'timestamp': now.isoformat(),
            'time_elapsed_seconds': (now - self.start_time).total_seconds(),
            'metrics': {}
        }
        
        # Organize metrics by category
        for key, value in metrics.items():
            if key.startswith(('precision_', 'recall_', 'f1_')):
                metric_type, avg_type = key.split('_')
                if metric_type not in metrics_entry['metrics']:
                    metrics_entry['metrics'][metric_type] = {}
                metrics_entry['metrics'][metric_type][avg_type] = value
            else:
                metrics_entry['metrics'][key] = value
        self.epoch_metrics.append(metrics_entry)
        
        # Track epoch timing
        if epoch not in self.epoch_times:
            self.epoch_times[epoch] = [now, now]  # [start_time, end_time]
        else:
            self.epoch_times[epoch][1] = now  # Update end_time
        
        # Update best metrics
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                if metric not in self.best_metrics or value > self.best_metrics[metric]['value']:
                    self.best_metrics[metric] = {
                        'value': value,
                        'epoch': epoch,
                        'timestamp': now.isoformat()
                    }
    
    def export_metrics(self):
        """Export all metrics and samples to files."""
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
        
        # Export classification metrics if available
        classification_metrics = {}
        for metric_name, values in self.classification_metrics.items():
            if isinstance(values, dict):
                # Handle metrics with multiple types (micro, macro, weighted)
                for avg_type, scores in values.items():
                    if scores:
                        classification_metrics[f'{metric_name}_{avg_type}'] = scores
            else:
                # Handle simple metrics like accuracy
                if values:
                    classification_metrics[metric_name] = values
        
        if classification_metrics:
            df = pd.DataFrame(classification_metrics)
            df.to_csv(self.output_dir / 'classification_metrics.csv', index=False)
            self.logger.info(f"Exported classification metrics to {self.output_dir / 'classification_metrics.csv'}")
        
        # Export samples
        if self.samples:
            samples_file = self.output_dir / 'samples.json'
            with open(samples_file, 'w') as f:
                json.dump(self.samples, f, indent=2)
            self.logger.info(f"Exported {len(self.samples)} samples to {samples_file}")
        
        # Calculate training insights
        total_time = (datetime.now() - self.start_time).total_seconds()
        epoch_times = {epoch: (end_time - start_time).total_seconds()
                      for epoch, (start_time, end_time) in self.epoch_times.items()}
        
        # Calculate detailed performance trends
        if self.training_metrics:
            df = pd.DataFrame(self.training_metrics)
            performance_trends = {
                'loss_trend': df.groupby('epoch')['loss'].mean().to_dict() if 'loss' in df else None,
                'accuracy_trend': df.groupby('epoch')['accuracy'].mean().to_dict() if 'accuracy' in df else None,
                'precision_trend': {
                    avg_type: df.groupby('epoch')[f'precision_{avg_type}'].mean().to_dict()
                    for avg_type in ['micro', 'macro', 'weighted']
                    if f'precision_{avg_type}' in df.columns
                },
                'recall_trend': {
                    avg_type: df.groupby('epoch')[f'recall_{avg_type}'].mean().to_dict()
                    for avg_type in ['micro', 'macro', 'weighted']
                    if f'recall_{avg_type}' in df.columns
                },
                'f1_trend': {
                    avg_type: df.groupby('epoch')[f'f1_{avg_type}'].mean().to_dict()
                    for avg_type in ['micro', 'macro', 'weighted']
                    if f'f1_{avg_type}' in df.columns
                }
            }
        else:
            performance_trends = {}
        
        # Export comprehensive summary as JSON
        summary = {
            'experiment_name': self.experiment_name,
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'training_summary': {
                'num_epochs': len(self.epoch_metrics),
                'total_steps': len(self.training_metrics),
                'total_time_seconds': total_time,
                'average_epoch_time': sum(epoch_times.values()) / len(epoch_times) if epoch_times else 0,
                'epoch_times': epoch_times,
                'num_samples_stored': len(self.samples)
            },
            'best_metrics': self.best_metrics,
            'performance_trends': performance_trends,
            'final_metrics': self.epoch_metrics[-1] if self.epoch_metrics else None,
            'classification_metrics_summary': {
                metric: {
                    'min': min(values) if values else None,
                    'max': max(values) if values else None,
                    'final': values[-1] if values else None
                }
                for metric, values in classification_metrics.items()
            } if classification_metrics else None
        }
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
