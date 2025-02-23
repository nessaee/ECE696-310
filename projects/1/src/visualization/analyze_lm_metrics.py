"""
Analyze and visualize language modeling metrics.
"""
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_lm_metrics(metrics: dict, output_dir: Path):
    """Plot language modeling metrics."""
    # Extract training and evaluation metrics
    epochs = []
    train_losses = []
    eval_losses = []
    eval_perplexities = []
    
    for key, value in metrics.items():
        if key.startswith('epoch_') and not key.endswith('_eval'):
            epoch = int(key.split('_')[1])
            epochs.append(epoch)
            train_losses.append(value['avg_loss'])
            
            # Get corresponding eval metrics
            eval_key = f'epoch_{epoch}_eval'
            if eval_key in metrics:
                eval_metrics = metrics[eval_key]
                eval_losses.append(eval_metrics['loss'])
                eval_perplexities.append(eval_metrics['perplexity'])
    
    # Sort by epoch
    epochs, train_losses, eval_losses, eval_perplexities = zip(*sorted(
        zip(epochs, train_losses, eval_losses, eval_perplexities)
    ))
    
    # Set style
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
    
    # Figure 1: Training and Validation Loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', marker='o')
    plt.plot(epochs, eval_losses, 'r-', label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Validation Perplexity
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, eval_perplexities, 'g-', label='Validation Perplexity', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'perplexity_curve.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Save final test metrics
    if 'final_test' in metrics:
        final_metrics = metrics['final_test']
        with open(output_dir / 'final_test_metrics.txt', 'w') as f:
            f.write(f"Final Test Metrics:\n")
            f.write(f"Loss: {final_metrics['loss']:.4f}\n")
            f.write(f"Perplexity: {final_metrics['perplexity']:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze language modeling metrics')
    parser.add_argument('metrics_file', type=str, help='Path to metrics JSON file')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    # Load metrics
    metrics_path = Path(args.metrics_file)
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plot_lm_metrics(metrics, output_dir)

if __name__ == '__main__':
    main()
