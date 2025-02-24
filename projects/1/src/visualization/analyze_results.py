"""
Script to compile and analyze experiment results.
"""
import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

def load_metrics_file(file_path: Path) -> Dict:
    """Load metrics from a JSON file."""
    with open(file_path) as f:
        return json.load(f)

def extract_run_info(path: Path) -> Dict:
    """Extract run number and type from path."""
    try:
        # Find the run directory (run_N)
        run_part = next(p for p in path.parts if p.startswith('run_'))
        run_num = int(run_part.split('_')[1])  # run_N -> N
        
        # Find if it's train or baseline
        if (path / 'train').exists():
            run_type = 'train'
        elif (path / 'baseline').exists():
            run_type = 'baseline'
        else:
            return None
            
        return {'run_num': run_num, 'type': run_type}
    except (StopIteration, IndexError, ValueError):
        return None

def load_config(path: Path) -> Dict:
    """Load config from a run directory."""
    config_path = path / 'analysis' / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}

def process_metrics_directory(metrics_dir: Path, run_info: Dict) -> List[Dict]:
    """Process all metrics files in a directory."""
    results = []
    
    try:
        # Process metrics files
        for metrics_file in metrics_dir.glob('metrics_epoch_*.json'):
            try:
                epoch = int(metrics_file.stem.split('_')[-1])
                metrics = load_metrics_file(metrics_file)
                
                # Create base result dictionary
                result = {
                    'epoch': epoch,
                    'run_num': run_info['run_num'],
                }
                
                # Add all metrics from the file
                if isinstance(metrics, dict):
                    # Flatten metrics if needed
                    if 'metrics' in metrics and isinstance(metrics['metrics'], dict):
                        metrics = metrics['metrics']
                    
                    # Add all metrics
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            result[key] = value
                        elif isinstance(value, dict):
                            # Handle nested metrics
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, (int, float)):
                                    result[f"{key}_{subkey}"] = subvalue
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {metrics_file}: {e}")
                continue
                
    except Exception as e:
        print(f"Error processing directory {metrics_dir}: {e}")
    
    return results

def compile_results(base_dir: Path) -> pd.DataFrame:
    """Compile all results into a DataFrame."""
    all_results = []
    
    # Process each model directory
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        print(f"Processing {model_name}...")
        
        # Process each run directory
        for run_dir in model_dir.glob('run_*'):
            if not run_dir.is_dir():
                continue
                
            try:
                # Extract run number
                run_num = int(run_dir.name.split('_')[1])
                
                # Process metrics
                metrics_dir = run_dir / 'train' / 'metrics'
                if metrics_dir.exists():
                    results = process_metrics_directory(metrics_dir, {'run_num': run_num})
                    if results:
                        print(f"Found {len(results)} metrics files in {run_dir}")
                        all_results.extend(results)
                    else:
                        print(f"No metrics found in {run_dir}")
                else:
                    print(f"No metrics directory found in {run_dir}")
                    
            except Exception as e:
                print(f"Error processing {run_dir}: {e}")
                continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by run_num and epoch
    if not df.empty:
        df = df.sort_values(['run_num', 'epoch'])
        print(f"\nFound data for {df['run_num'].nunique()} runs across {len(df)} epochs")
        print("\nColumns found:")
        print(df.columns.tolist())
        print("\nSample data:")
        print(df.head())
    else:
        print("\nNo data found!")
    
    return df

def plot_metrics(df: pd.DataFrame, output_dir: Path):
    """Create visualization plots."""
    print("\nDataFrame Info:")
    print("Columns:", df.columns.tolist())
    print("\nSample data:")
    print(df.head())
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set basic style
    plt.style.use('default')
    
    # Configure plot style manually
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'grid.color': '#dddddd',
        'axes.linewidth': 1.0,
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
    })
    
    # Set color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    sns.set_palette(sns.color_palette(colors))
    
    # Plot training curves
    metrics_to_plot = ['loss', 'accuracy', 'f1_weighted']
    available_metrics = [m for m in metrics_to_plot if m in df.columns]
    
    print(f"\nPlotting metrics: {available_metrics}")
    
    for metric in available_metrics:
        plt.figure(figsize=(10, 6))
        
        # Plot by run number
        sns.lineplot(data=df, x='epoch', y=metric, hue='run_num', marker='o')
        
        plt.title(f'{metric.replace("_", " ").title()} vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend(title='Run Number', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric}_vs_epoch.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save summary statistics
    try:
        summary_metrics = {}
        for metric in available_metrics:
            summary_metrics[metric] = ['mean', 'std', 'min', 'max']
        
        if summary_metrics:
            summary = df.groupby('run_num').agg(summary_metrics).round(4)
            summary.to_csv(output_dir / 'summary_statistics.csv')
            print("\nSummary statistics:")
            print(summary)
    except Exception as e:
        print(f"\nError creating summary statistics: {e}")


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('results_dir', type=str, help='Path to results directory')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    print(f"Compiling results from {results_dir}...")
    df = compile_results(results_dir)
    
    # Save full results
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'all_results.csv', index=False)
    print(f"Saved full results to {output_dir / 'all_results.csv'}")
    
    # Create visualizations
    print("Creating visualizations...")
    plot_metrics(df, output_dir)
    print(f"Saved visualizations to {output_dir}")
    
    # Print summary
    print("\nSummary of results:")
    print(f"Total runs: {df['run_num'].nunique()}")
    print(f"Models: {', '.join(df['model'].unique())}")
    print(f"Datasets: {', '.join(df['dataset'].unique())}")
    
    if 'accuracy' in df.columns:
        print(f"\nBest accuracy: {df['accuracy'].max():.4f}")
    if 'f1_weighted' in df.columns:
        print(f"Best F1 (weighted): {df['f1_weighted'].max():.4f}")

if __name__ == '__main__':
    main()
