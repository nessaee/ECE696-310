import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import numpy as np
from pathlib import Path
import argparse
import json

def save_summary_stats(training_df: pd.DataFrame, 
                      baseline_df: pd.DataFrame,
                      output_dir: Path) -> None:
    """Save summary statistics for the entire experiment."""
    # Filter out datasets with NaN values
    valid_datasets = [dataset for dataset in training_df['dataset'].unique() 
                     if not training_df[training_df['dataset'] == dataset]['accuracy'].isna().any()]
    
    training_filtered = training_df[training_df['dataset'].isin(valid_datasets)]
    baseline_filtered = baseline_df[baseline_df['dataset'].isin(valid_datasets)]
    
    summary = {
        'datasets': valid_datasets,
        'total_epochs': len(training_filtered['epoch'].unique()),
        'metrics': {
            'training': {
                'final': training_filtered.groupby('dataset').last()[['accuracy', 'precision', 'recall', 'f1']].to_dict(),
                'best': training_filtered.groupby('dataset').max()[['accuracy', 'precision', 'recall', 'f1']].to_dict()
            },
            'baseline': baseline_filtered.groupby('dataset')[['accuracy', 'precision', 'recall', 'f1']].mean().to_dict()
        }
    }
    
    output_path = output_dir / 'experiment_summary.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved experiment summary to {output_path}")

def load_and_process_data(training_path: Path, baseline_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and process the training and baseline data from CSV files.
    
    Args:
        training_path: Path to the training progression CSV
        baseline_path: Path to the baseline performance CSV
    
    Returns:
        Tuple of processed training and baseline DataFrames
    """
    # Load the data
    training_df = pd.read_csv(training_path)
    baseline_df = pd.read_csv(baseline_path)
    
    # Group training data by dataset and epoch, calculating mean metrics
    training_grouped = training_df.groupby(['dataset', 'epoch']).agg({
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean'
    }).reset_index()
    
    return training_grouped, baseline_df

def create_performance_visualization(training_df: pd.DataFrame, 
                                  baseline_df: pd.DataFrame,
                                  dataset: str,
                                  output_dir: Path) -> None:
    """
    Create and save performance comparison visualizations.
    
    Args:
        training_df: Processed training data
        baseline_df: Baseline performance data
        dataset: Name of the dataset to visualize
    """
    # Filter data for the specified dataset
    dataset_training = training_df[training_df['dataset'] == dataset]
    dataset_baseline = baseline_df[baseline_df['dataset'] == dataset]
    
    # Set up the plot style with modern aesthetics
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Colors for different metrics
    colors = ['#2563eb', '#16a34a', '#dc2626', '#9333ea']
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Plot 1: Line plot for training progression
    for metric, color in zip(metrics, colors):
        # Add baseline as horizontal line
        baseline_value = dataset_baseline[metric].values[0]
        ax1.axhline(y=baseline_value, color=color, linestyle='--', alpha=0.5,
                    label=f'Baseline {metric}')
        
        # Add training progression
        ax1.plot(dataset_training['epoch'], dataset_training[metric],
                marker='o', color=color, label=f'Training {metric}')
    
    ax1.set_title(f'Model Performance Progression - {dataset}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Metric Value')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bar plot comparison
    bar_width = 0.15
    epochs = dataset_training['epoch'].unique()
    n_groups = len(epochs) + 1  # +1 for baseline
    
    # Set up x positions for grouped bars
    x = np.arange(n_groups)
    
    for i, metric in enumerate(metrics):
        # Baseline bar (first group)
        ax2.bar(x[0] + i*bar_width - (len(metrics)-1)*bar_width/2,
                dataset_baseline[metric].values[0],
                bar_width, label=metric.capitalize(),
                color=colors[i], alpha=0.8)
        
        # Training bars for each epoch
        for j, epoch in enumerate(epochs, start=1):
            epoch_data = dataset_training[dataset_training['epoch'] == epoch]
            ax2.bar(x[j] + i*bar_width - (len(metrics)-1)*bar_width/2,
                    epoch_data[metric].values[0],
                    bar_width, color=colors[i])
    
    ax2.set_title(f'Performance Metrics Comparison - {dataset}')
    ax2.set_ylabel('Metric Value')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Baseline'] + [f'Epoch {e}' for e in epochs])
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'performance_comparison_{dataset}.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved performance visualization to {output_path}")
    plt.close()

def create_performance_table(training_df: pd.DataFrame,
                           baseline_df: pd.DataFrame,
                           dataset: str,
                           output_dir: Path) -> pd.DataFrame:
    """
    Create a performance comparison table.
    
    Args:
        training_df: Processed training data
        baseline_df: Baseline performance data
        dataset: Name of the dataset to analyze
        
    Returns:
        DataFrame containing the performance comparison table
    """
    # Filter data for the specified dataset
    dataset_training = training_df[training_df['dataset'] == dataset]
    dataset_baseline = baseline_df[baseline_df['dataset'] == dataset]
    
    # Create comparison table
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    comparison_data = {
        'Baseline': dataset_baseline[metrics].iloc[0].to_dict()
    }
    
    # Add data for each epoch
    for epoch in dataset_training['epoch'].unique():
        epoch_data = dataset_training[dataset_training['epoch'] == epoch]
        comparison_data[f'Epoch {epoch}'] = epoch_data[metrics].iloc[0].to_dict()
    
    # Create and format the comparison table
    comparison_df = pd.DataFrame(comparison_data).T
    comparison_df = comparison_df.round(4)
    
    # Calculate improvement over baseline
    baseline_values = comparison_df.loc['Baseline']
    improvement_df = ((comparison_df - baseline_values) / baseline_values * 100).round(2)
    improvement_df.columns = [f'{col}_improvement_%' for col in improvement_df.columns]
    
    # Combine metrics and improvements
    final_df = pd.concat([comparison_df, improvement_df], axis=1)
    final_df = final_df.sort_index()
    
    return final_df

def main():
    """
    Main function to execute the performance analysis.
    """
    parser = argparse.ArgumentParser(description='Analyze model performance across training runs')
    parser.add_argument('base_dir', type=str, help='Base directory containing analysis results')
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    analysis_dir = base_dir / 'analysis'
    
    if not analysis_dir.exists():
        print(f"Error: Analysis directory not found at {analysis_dir}")
        return
    
    # Load and process data
    training_df, baseline_df = load_and_process_data(
        analysis_dir / 'training_progression.csv',
        analysis_dir / 'baseline_performance.csv'
    )
    
    # Create detailed analysis directory
    detailed_analysis_dir = analysis_dir / 'detailed_analysis'
    detailed_analysis_dir.mkdir(exist_ok=True)
    
    # Process each dataset
    for dataset in training_df['dataset'].unique():
        print(f"\nAnalyzing dataset: {dataset}")
        
        # Create visualizations
        create_performance_visualization(training_df, baseline_df, dataset, detailed_analysis_dir)
        
        # Create and display performance table
        comparison_table = create_performance_table(training_df, baseline_df, dataset, detailed_analysis_dir)
        print("\nPerformance Comparison Table:")
        print(comparison_table.to_string())
        
        # Save table to CSV
        output_path = detailed_analysis_dir / f'performance_comparison_{dataset}.csv'
        comparison_table.to_csv(output_path)
        print(f"Saved performance comparison to {output_path}")
    
    # Save overall experiment summary
    save_summary_stats(training_df, baseline_df, analysis_dir)  
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved experiment summary to {output_path}")

def main():
    """
    Main function to execute the performance analysis.
    """
    parser = argparse.ArgumentParser(description='Analyze model performance across training runs')
    parser.add_argument('base_dir', type=str, help='Base directory containing analysis results')
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    analysis_dir = base_dir / 'analysis'
    
    if not analysis_dir.exists():
        print(f"Error: Analysis directory not found at {analysis_dir}")
        return
    
    # Load and process data
    training_path = analysis_dir / 'training_progression.csv'
    baseline_path = analysis_dir / 'baseline_performance.csv'
    
    if not training_path.exists():
        print(f"Error: Training progression file not found at {training_path}")
        return
    if not baseline_path.exists():
        print(f"Error: Baseline performance file not found at {baseline_path}")
        return
        
    training_df, baseline_df = load_and_process_data(training_path, baseline_path)
    
    # Create detailed analysis directory
    detailed_analysis_dir = analysis_dir / 'detailed_analysis'
    detailed_analysis_dir.mkdir(exist_ok=True)
    
    # Filter out datasets with NaN values
    valid_datasets = [dataset for dataset in training_df['dataset'].unique() 
                     if not training_df[training_df['dataset'] == dataset]['accuracy'].isna().any()]
    
    if not valid_datasets:
        print("No valid datasets found with complete metrics")
        return
    
    # Process each valid dataset
    for dataset in valid_datasets:
        print(f"\nAnalyzing dataset: {dataset}")
        
        # Create visualizations
        create_performance_visualization(training_df, baseline_df, dataset, detailed_analysis_dir)
        
        # Create and display performance table
        comparison_table = create_performance_table(training_df, baseline_df, dataset, detailed_analysis_dir)
        print("\nPerformance Comparison Table:")
        print(comparison_table.to_string())
        
        # Save table to CSV
        output_path = detailed_analysis_dir / f'performance_comparison_{dataset}.csv'
        comparison_table.to_csv(output_path)
        print(f"Saved performance comparison to {output_path}")
    
    # Save overall experiment summary
    save_summary_stats(training_df, baseline_df, analysis_dir)

if __name__ == "__main__":
    main()