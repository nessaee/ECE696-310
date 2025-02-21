"""
Script to analyze and visualize training metrics from JSON data.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Any
import re

def setup_plotting_style():
    """Configure plot styling."""
    plt.style.use('bmh')
    plt.rcParams.update({
        'figure.figsize': [12, 8],
        'figure.dpi': 100,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'axes.titlesize': 14,
        'axes.labelsize': 12
    })

def extract_metrics_series(data: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Extract time series metrics from the JSON data.
    """
    num_epochs = data['metadata']['num_epochs']
    series = {
        'loss': [],
        'avg_loss': [],
        'steps': [],
        'time': [],
        'eval_accuracy': [],
        'eval_f1_score': []
    }
    
    for epoch in range(num_epochs):
        # Training metrics
        epoch_data = data[f'epoch_{epoch}']
        series['loss'].append(epoch_data['loss'])
        series['avg_loss'].append(epoch_data['avg_loss'])
        series['steps'].append(epoch_data['steps'])
        series['time'].append(epoch_data['time'])
        
        # Evaluation metrics
        eval_data = data[f'epoch_{epoch}_eval']
        series['eval_accuracy'].append(eval_data['accuracy'])
        series['eval_f1_score'].append(eval_data['f1_score'])
    
    return series

def parse_classification_report(report_str: str) -> Dict[str, Dict[str, float]]:
    """
    Parse the classification report string into a structured format.
    """
    lines = report_str.strip().split('\n')[2:]  # Skip header
    metrics = {}
    
    for line in lines:
        if line.strip():
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 5:  # Valid metric line
                class_name = parts[0]
                metrics[class_name] = {
                    'precision': float(parts[1]),
                    'recall': float(parts[2]),
                    'f1_score': float(parts[3]),
                    'support': int(parts[4])
                }
    
    return metrics

def plot_training_metrics(metrics: Dict[str, List[float]], save_dir: Path):
    """
    Create visualizations of training progress.
    """
    epochs = list(range(len(metrics['loss'])))
    
    # Figure 1: Training Metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Progress', fontsize=16, y=1.02)
    
    # Plot 1: Loss Metrics
    ax = axes[0, 0]
    ax.plot(epochs, metrics['loss'], marker='o', label='Total Loss')
    ax.plot(epochs, metrics['avg_loss'], marker='s', label='Average Loss')
    ax.set_title('Loss per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    
    # Plot 2: Evaluation Metrics
    ax = axes[0, 1]
    ax.plot(epochs, metrics['eval_accuracy'], marker='o', label='Accuracy')
    ax.plot(epochs, metrics['eval_f1_score'], marker='s', label='F1 Score')
    ax.set_title('Evaluation Metrics per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.legend()
    
    # Plot 3: Training Time
    ax = axes[1, 0]
    ax.plot(epochs, [t/60 for t in metrics['time']], marker='o')  # Convert to minutes
    ax.set_title('Training Time per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time (minutes)')
    
    # Plot 4: Steps per Epoch
    ax = axes[1, 1]
    ax.plot(epochs, metrics['steps'], marker='o')
    ax.set_title('Steps per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Number of Steps')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_metrics.png')
    plt.close()
    
    # Figure 2: Per-Class Performance
    final_report = parse_classification_report(data['final_test']['classification_report'])
    classes = [k for k in final_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    metrics_to_plot = ['precision', 'recall', 'f1_score']
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics_to_plot):
        values = [final_report[cls][metric] for cls in classes]
        ax.bar(x + i*width, values, width, label=metric.capitalize())
    
    ax.set_ylabel('Score')
    ax.set_title('Final Test Performance by Class')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'final_test_metrics.png')
    plt.close()

def create_summary(data: Dict[str, Any], metrics: Dict[str, List[float]], save_dir: Path):
    """
    Create a summary of the training run.
    """
    summary = {
        'configuration': {
            'model': data['metadata']['model'],
            'dataset': data['metadata']['dataset'],
            'num_epochs': data['metadata']['num_epochs'],
            'learning_rate': data['metadata']['learning_rate'],
            'eval_steps': data['metadata']['eval_steps'],
            'save_steps': data['metadata']['save_steps'],
            'use_wandb': data['metadata']['use_wandb']
        },
        'training_progress': {
            'initial_loss': metrics['loss'][0],
            'final_loss': metrics['loss'][-1],
            'initial_avg_loss': metrics['avg_loss'][0],
            'final_avg_loss': metrics['avg_loss'][-1],
            'total_steps': sum(metrics['steps']),
            'total_time_minutes': sum(metrics['time']) / 60,
            'best_eval_accuracy': max(metrics['eval_accuracy']),
            'best_eval_f1_score': max(metrics['eval_f1_score'])
        },
        'final_test_results': {
            'accuracy': data['final_test']['accuracy'],
            'f1_score': data['final_test']['f1_score']
        }
    }
    
    # Save as JSON
    with open(save_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create markdown summary
    markdown = f"""# Training Summary

## Configuration
- Model: {summary['configuration']['model']}
- Dataset: {summary['configuration']['dataset']}
- Number of Epochs: {summary['configuration']['num_epochs']}
- Learning Rate: {summary['configuration']['learning_rate'] or 'default'}
- Evaluation Steps: {summary['configuration']['eval_steps']}
- Save Steps: {summary['configuration']['save_steps']}
- WandB Logging: {'Enabled' if summary['configuration']['use_wandb'] else 'Disabled'}

## Training Progress
- Initial Loss: {summary['training_progress']['initial_loss']:.4f}
- Final Loss: {summary['training_progress']['final_loss']:.4f}
- Initial Average Loss: {summary['training_progress']['initial_avg_loss']:.4f}
- Final Average Loss: {summary['training_progress']['final_avg_loss']:.4f}
- Total Steps: {summary['training_progress']['total_steps']}
- Total Training Time: {summary['training_progress']['total_time_minutes']:.2f} minutes
- Best Evaluation Accuracy: {summary['training_progress']['best_eval_accuracy']:.4f}
- Best Evaluation F1 Score: {summary['training_progress']['best_eval_f1_score']:.4f}

## Final Test Results
- Accuracy: {summary['final_test_results']['accuracy']:.4f}
- F1 Score: {summary['final_test_results']['f1_score']:.4f}

## Classification Report
```
{data['final_test']['classification_report']}
```
"""
    
    with open(save_dir / 'training_summary.md', 'w') as f:
        f.write(markdown)

def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize JSON metrics')
    parser.add_argument('json_file', type=str, help='Path to metrics JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save visualizations (default: same directory as JSON)')
    
    args = parser.parse_args()
    json_path = Path(args.json_file)
    output_dir = Path(args.output_dir) if args.output_dir else json_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process data
    global data  # Make accessible to helper functions
    with open(json_path) as f:
        data = json.load(f)
    
    metrics = extract_metrics_series(data)
    
    # Setup plotting
    setup_plotting_style()
    
    # Generate visualizations and summary
    plot_training_metrics(metrics, output_dir)
    create_summary(data, metrics, output_dir)
    
    print(f"\nAnalysis complete! Results saved in: {output_dir}")
    print("Generated files:")
    print(f"- {output_dir}/training_metrics.png")
    print(f"- {output_dir}/final_test_metrics.png")
    print(f"- {output_dir}/training_summary.json")
    print(f"- {output_dir}/training_summary.md")

if __name__ == '__main__':
    main()
