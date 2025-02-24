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
        output_dir: Directory to save the output files
    """
    # Filter data for the specified dataset
    dataset_training = training_df[training_df['dataset'] == dataset]
    dataset_baseline = baseline_df[baseline_df['dataset'] == dataset]
    
    # Set up the plot style with modern aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    # Colors for different metrics
    colors = ['#2563eb', '#16a34a', '#dc2626', '#9333ea']
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Create training progression plot
    plt.figure(figsize=(10, 6))
    for metric, color in zip(metrics, colors):
        # Add baseline as horizontal line
        baseline_value = dataset_baseline[metric].values[0]
        plt.axhline(y=baseline_value, color=color, linestyle='--', alpha=0.5,
                    label=f'Baseline {metric}')
        
        # Add training progression
        plt.plot(dataset_training['epoch'], dataset_training[metric],
                marker='o', color=color, label=f'Training {metric}')
    
    plt.title(f'Training Progression - {dataset}')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save training progression plot
    progression_path = output_dir / f'training_progression_{dataset}.png'
    plt.savefig(progression_path, bbox_inches='tight', dpi=300)
    print(f"Saved training progression plot to {progression_path}")
    plt.close()
    
    # Create metrics comparison plot
    plt.figure(figsize=(10, 6))
    bar_width = 0.15
    epochs = dataset_training['epoch'].unique()
    n_groups = len(epochs) + 1  # +1 for baseline
    
    # Set up x positions for grouped bars
    x = np.arange(n_groups)
    
    for i, metric in enumerate(metrics):
        # Baseline bar (first group)
        plt.bar(x[0] + i*bar_width - (len(metrics)-1)*bar_width/2,
                dataset_baseline[metric].values[0],
                bar_width, label=metric.capitalize(),
                color=colors[i], alpha=0.8)
        
        # Training bars for each epoch
        for j, epoch in enumerate(epochs, start=1):
            epoch_data = dataset_training[dataset_training['epoch'] == epoch]
            plt.bar(x[j] + i*bar_width - (len(metrics)-1)*bar_width/2,
                    epoch_data[metric].values[0],
                    bar_width, color=colors[i])
    
    plt.title(f'Metrics Comparison - {dataset}')
    plt.ylabel('Metric Value')
    plt.xticks(x, ['Baseline'] + [f'Epoch {e}' for e in epochs])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save metrics comparison plot
    comparison_path = output_dir / f'metrics_comparison_{dataset}.png'
    plt.savefig(comparison_path, bbox_inches='tight', dpi=300)
    print(f"Saved metrics comparison plot to {comparison_path}")
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
    # Analysis directory is created by config.py
    
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
    # Analysis directory is created by config.py
    
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