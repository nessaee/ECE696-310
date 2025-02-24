"""
Script to analyze and visualize training metrics from CSV data.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
import argparse
from typing import Dict, List, Any

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

def extract_epoch_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and reshape epoch-specific metrics into a more analyzable format.
    """
    epoch_metrics = []
    
    # Get number of epochs from column names
    epoch_cols = [col for col in df.columns if col.startswith('epoch_')]
    num_epochs = len(set(int(col.split('_')[1]) for col in epoch_cols))
    
    for epoch in range(num_epochs):
        metrics = {}
        metrics['epoch'] = epoch
        
        # Extract metrics for this epoch
        for col in df.columns:
            if col.startswith(f'epoch_{epoch}_'):
                metric_name = '_'.join(col.split('_')[2:])  # Remove 'epoch_N_' prefix
                if metric_name not in ['classification_report']:  # Skip text reports
                    metrics[metric_name] = df[col].iloc[0]
        
        epoch_metrics.append(metrics)
    
    return pd.DataFrame(epoch_metrics)

def plot_training_progress(epoch_df: pd.DataFrame, save_dir: Path):
    """
    Plot various training metrics over epochs.
    """
    # Determine task type based on available metrics
    is_classification = 'eval_accuracy' in epoch_df.columns
    is_language_modeling = 'eval_perplexity' in epoch_df.columns
    
    # Create figure with appropriate number of subplots
    if is_classification:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
    else:  # Language modeling
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()
    
    fig.suptitle('Training Progress', fontsize=16, y=1.02)
    
    # Plot 1: Training Loss (common for both tasks)
    ax = axes[0]
    ax.plot(epoch_df['epoch'], epoch_df['avg_loss'], marker='o', label='Average Loss')
    ax.set_title('Training Loss per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    
    # Plot task-specific metrics
    if is_classification:
        # Plot 2: Classification Metrics
        ax = axes[1]
        ax.plot(epoch_df['epoch'], epoch_df['eval_accuracy'], marker='o', label='Accuracy')
        if 'eval_f1_score' in epoch_df.columns:
            ax.plot(epoch_df['epoch'], epoch_df['eval_f1_score'], marker='s', label='F1 Score')
        ax.set_title('Evaluation Metrics per Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.legend()
        
        # Plot 3: Training Time
        ax = axes[2]
        ax.plot(epoch_df['epoch'], epoch_df['time'] / 60, marker='o')  # Convert to minutes
        ax.set_title('Training Time per Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (minutes)')
        
        # Plot 4: Steps per Epoch
        ax = axes[3]
        ax.plot(epoch_df['epoch'], epoch_df['steps'], marker='o')
        ax.set_title('Steps per Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Number of Steps')
        
    elif is_language_modeling:
        # Plot 2: Perplexity
        ax = axes[1]
        ax.plot(epoch_df['epoch'], epoch_df['eval_perplexity'], marker='o', label='Validation')
        ax.set_title('Perplexity per Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Perplexity')
        ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_dir / 'training_progress.png', dpi=100, bbox_inches='tight')
    plt.close()

def create_summary_table(df: pd.DataFrame, epoch_df: pd.DataFrame, save_dir: Path):
    """
    Create and save a summary of the training run.
    """
    # Determine task type based on available metrics
    is_classification = 'final_test_accuracy' in df.columns
    is_language_modeling = 'final_test_perplexity' in df.columns
    
    summary = {
        'model': df['model'].iloc[0],
        'dataset': df['dataset'].iloc[0],
        'num_epochs': int(df['num_epochs'].iloc[0]),
        'learning_rate': float(df['learning_rate'].iloc[0]) if pd.notnull(df['learning_rate'].iloc[0]) else None,
        'training_progress': {
            'initial_loss': float(epoch_df['avg_loss'].iloc[0]),
            'final_loss': float(epoch_df['avg_loss'].iloc[-1]),
            'total_training_time_minutes': float(epoch_df['time'].sum() / 60),
            'total_steps': int(epoch_df['steps'].sum())
        }
    }
    
    # Add task-specific metrics
    if is_classification:
        summary['final_test_metrics'] = {
            'accuracy': float(df['final_test_accuracy'].iloc[0]),
            'f1_score': float(df['final_test_f1_score'].iloc[0])
        }
        summary['training_progress'].update({
            'best_eval_accuracy': float(epoch_df['eval_accuracy'].max()),
            'best_eval_f1_score': float(epoch_df['eval_f1_score'].max()) if 'eval_f1_score' in epoch_df.columns else None
        })
    elif is_language_modeling:
        summary['final_test_metrics'] = {
            'loss': float(df['final_test_loss'].iloc[0]),
            'perplexity': float(df['final_test_perplexity'].iloc[0])
        }
        summary['training_progress'].update({
            'best_eval_perplexity': float(epoch_df['eval_perplexity'].min())
        })
    
    # Save as JSON
    with open(save_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create markdown content
    markdown = f"""# Training Summary

## Configuration
- Model: {summary['model']}
- Dataset: {summary['dataset']}
- Number of Epochs: {summary['num_epochs']}
- Learning Rate: {summary['learning_rate']}

## Final Test Results
"""
    
    # Add task-specific metrics to markdown
    if is_classification:
        markdown += f"""- Accuracy: {summary['final_test_metrics']['accuracy']:.4f}
- F1 Score: {summary['final_test_metrics']['f1_score']:.4f}
"""
    elif is_language_modeling:
        markdown += f"""- Loss: {summary['final_test_metrics']['loss']:.4f}
- Perplexity: {summary['final_test_metrics']['perplexity']:.4f}
"""
    
    markdown += f"""
## Training Progress
- Initial Loss: {summary['training_progress']['initial_loss']:.4f}
- Final Loss: {summary['training_progress']['final_loss']:.4f}
"""
    
    if is_classification:
        markdown += f"""- Best Evaluation Accuracy: {summary['training_progress']['best_eval_accuracy']:.4f}
- Best Evaluation F1 Score: {summary['training_progress']['best_eval_f1_score']:.4f}
"""
    elif is_language_modeling:
        markdown += f"""- Best Evaluation Perplexity: {summary['training_progress']['best_eval_perplexity']:.4f}
"""
    
    markdown += f"""- Total Training Time: {summary['training_progress']['total_training_time_minutes']:.2f} minutes
- Total Steps: {summary['training_progress']['total_steps']}
"""
    
    with open(save_dir / 'training_summary.md', 'w') as f:
        f.write(markdown)

def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize training metrics')
    parser.add_argument('csv_file', type=str, help='Path to metrics CSV file')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Directory to save visualizations (default: same directory as CSV)')
    
    args = parser.parse_args()
    csv_path = Path(args.csv_file)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read and process data
    df = pd.read_csv(csv_path)
    epoch_df = extract_epoch_metrics(df)
    
    # Setup plotting
    setup_plotting_style()
    
    # Generate visualizations and summary
    plot_training_progress(epoch_df, output_dir)
    create_summary_table(df, epoch_df, output_dir)
    
    print(f"\nAnalysis complete! Results saved in: {output_dir}")
    print("Generated files:")
    print(f"- {output_dir}/training_progress.png")
    print(f"- {output_dir}/training_summary.json")
    print(f"- {output_dir}/training_summary.md")

if __name__ == '__main__':
    main()
