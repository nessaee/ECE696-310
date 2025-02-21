"""
Visualization script for training and evaluation metrics.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import Optional, List

def setup_style():
    """Configure plot styling."""
    plt.style.use('bmh')  # Use built-in style that's similar to seaborn
    plt.rcParams.update({
        'figure.figsize': [12, 6],
        'figure.dpi': 100,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'axes.titlesize': 14,
        'axes.labelsize': 12
    })

def plot_training_metrics(metrics_dir: Path, save_dir: Optional[Path] = None):
    """Plot training metrics over time."""
    # Load training metrics
    df = pd.read_csv(metrics_dir / 'training_metrics.csv')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot loss
    sns.lineplot(data=df, x='step', y='loss', ax=axes[0])
    axes[0].set_title('Training Loss over Steps')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    
    # Plot learning rate
    if 'learning_rate' in df.columns:
        sns.lineplot(data=df, x='step', y='learning_rate', ax=axes[1])
        axes[1].set_title('Learning Rate over Steps')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Learning Rate')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / 'training_metrics.png')
        plt.close()
    else:
        plt.show()

def plot_eval_metrics(metrics_dir: Path, metrics: Optional[List[str]] = None, save_dir: Optional[Path] = None):
    """Plot evaluation metrics over time."""
    # Load evaluation metrics
    df = pd.read_csv(metrics_dir / 'eval_metrics.csv')
    
    if metrics is None:
        # Exclude non-numeric columns
        metrics = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        metrics = [m for m in metrics if m not in ['step', 'epoch']]
    
    # Create subplots for each metric
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
    if n_metrics == 1:
        axes = [axes]
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            sns.lineplot(data=df, x='step', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric} over Steps')
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel(metric)
    
    # Remove empty subplots
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / 'eval_metrics.png')
        plt.close()
    else:
        plt.show()

def plot_epoch_metrics(metrics_dir: Path, metrics: Optional[List[str]] = None, save_dir: Optional[Path] = None):
    """Plot epoch-level metrics."""
    # Load epoch metrics
    df = pd.read_csv(metrics_dir / 'epoch_metrics.csv')
    
    if metrics is None:
        # Exclude non-numeric columns
        metrics = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        metrics = [m for m in metrics if m not in ['epoch']]
    
    # Create subplots for each metric
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
    if n_metrics == 1:
        axes = [axes]
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            sns.lineplot(data=df, x='epoch', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric} over Epochs')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric)
    
    # Remove empty subplots
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / 'epoch_metrics.png')
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot training and evaluation metrics')
    parser.add_argument('metrics_dir', type=str, help='Directory containing metrics CSV files')
    parser.add_argument('--save-dir', type=str, help='Directory to save plots (optional)')
    parser.add_argument('--metrics', nargs='+', help='Specific metrics to plot (optional)')
    
    args = parser.parse_args()
    metrics_dir = Path(args.metrics_dir)
    save_dir = Path(args.save_dir) if args.save_dir else None
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    setup_style()
    
    # Plot all available metrics
    if (metrics_dir / 'training_metrics.csv').exists():
        plot_training_metrics(metrics_dir, save_dir)
    
    if (metrics_dir / 'eval_metrics.csv').exists():
        plot_eval_metrics(metrics_dir, args.metrics, save_dir)
    
    if (metrics_dir / 'epoch_metrics.csv').exists():
        plot_epoch_metrics(metrics_dir, args.metrics, save_dir)

if __name__ == '__main__':
    main()
