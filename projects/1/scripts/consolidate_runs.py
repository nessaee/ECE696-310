#!/usr/bin/env python3

import shutil
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, List, Any
from datetime import datetime

def parse_metrics(metrics_file: Path) -> Dict[str, Any]:
    """Parse metrics from a JSON file."""
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error parsing {metrics_file}: {e}")
        return {}

def parse_config(config_file: Path) -> Dict[str, Any]:
    """Parse configuration from a JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error parsing {config_file}: {e}")
        return {}

def get_dataset_from_log(run_dir: Path) -> str:
    """Extract dataset name from training log filename."""
    log_files = list(run_dir.glob("train_*.log"))
    if not log_files:
        return "unknown"
    
    log_name = log_files[0].name
    # Expected format: train_distilgpt2_dataset_timestamp.log
    parts = log_name.split('_')
    if len(parts) >= 3:
        return parts[2]
    return "unknown"

def create_training_progression_df(train_metrics: Dict[str, List[Dict]], configs: Dict[str, Dict]) -> pd.DataFrame:
    """Create a DataFrame tracking training progression across epochs for all runs."""
    progression_data = []
    
    for run_id, metrics_list in train_metrics.items():
        config = configs.get(run_id, {})
        dataset = config.get('dataset', 'unknown')
        
        for epoch, metrics in enumerate(metrics_list):
            data_point = {
                'run_id': run_id,
                'dataset': dataset,
                'epoch': epoch,
                'loss': metrics.get('loss', float('nan')),
                'avg_loss': metrics.get('avg_loss', float('nan')),
                'steps': metrics.get('steps', 0),
                'time': metrics.get('time', float('nan')),
                'accuracy': metrics.get('accuracy', float('nan')),
                'precision': metrics.get('precision_weighted', float('nan')),
                'recall': metrics.get('recall_weighted', float('nan')),
                'f1': metrics.get('f1_weighted', float('nan')),
                'timestamp': config.get('timestamp', 'unknown')
            }
            progression_data.append(data_point)
    
    return pd.DataFrame(progression_data)

def parse_classification_report(report_str: str) -> Dict[str, float]:
    """Parse the classification report string to extract metrics."""
    metrics = {}
    try:
        # Split the report into lines and process the last two lines for macro and weighted averages
        lines = report_str.strip().split('\n')
        macro_line = lines[-2].split()
        weighted_line = lines[-1].split()
        
        metrics.update({
            'macro_precision': float(macro_line[2]),
            'macro_recall': float(macro_line[3]),
            'macro_f1': float(macro_line[4]),
            'precision': float(weighted_line[2]),
            'recall': float(weighted_line[3]),
            'f1': float(weighted_line[4])
        })
    except Exception as e:
        print(f"Error parsing classification report: {e}")
    return metrics

def create_baseline_performance_df(test_metrics: Dict[str, Dict], configs: Dict[str, Dict]) -> pd.DataFrame:
    """Create a DataFrame of baseline performance metrics."""
    baseline_data = []
    
    for run_id, metrics in test_metrics.items():
        config = configs.get(run_id, {})
        
        # Extract metrics from classification report
        report_metrics = parse_classification_report(metrics.get('classification_report', ''))
        
        data_point = {
            'run_id': run_id,
            'dataset': config.get('dataset', 'unknown'),
            'timestamp': config.get('timestamp', 'unknown'),
            'accuracy': metrics.get('accuracy', float('nan')),
            'f1_score': metrics.get('f1_score', float('nan')),
            **report_metrics  # Include parsed classification report metrics
        }
        baseline_data.append(data_point)
    
    return pd.DataFrame(baseline_data)

def create_summary_report(base_dir: Path, train_metrics: Dict[str, List[Dict]], test_metrics: Dict[str, Dict],
                         configs: Dict[str, Dict]) -> None:
    """Create a summary report of all runs."""
    summary_data = []
    
    # Process each run that has either training metrics, test metrics, or config
    all_run_ids = sorted(set(train_metrics.keys()) | set(test_metrics.keys()) | set(configs.keys()))
    
    if not all_run_ids:
        print("No data found to analyze!")
        return
        
    for run_id in all_run_ids:
        config = configs.get(run_id, {})
        train_data = train_metrics.get(run_id, [])
        test_data = test_metrics.get(run_id, {})
        
        run_summary = {
            'run_id': run_id,
            'dataset': config.get('dataset', 'unknown'),
            'model': config.get('model_name', 'unknown'),
            'evaluation_type': config.get('evaluation_type', 'unknown'),
            'timestamp': config.get('timestamp', 'unknown'),
            'final_train_loss': train_data[-1].get('loss', float('nan')) if train_data else float('nan'),
            'test_loss': test_data.get('loss', float('nan')),
            'test_accuracy': test_data.get('accuracy', float('nan'))
        }
        summary_data.append(run_summary)
    
    if not summary_data:
        print("No data found to analyze!")
        return
        
    df = pd.DataFrame(summary_data)
    
    # Save detailed summary as CSV
    summary_file = base_dir / 'analysis' / 'performance_summary.csv'
    df.to_csv(summary_file, index=False)
    print(f"Saved performance summary to {summary_file}")
    
    # Create and save baseline performance data
    if test_metrics:
        baseline_df = create_baseline_performance_df(test_metrics, configs)
        baseline_file = base_dir / 'analysis' / 'baseline_performance.csv'
        baseline_df.to_csv(baseline_file, index=False)
        print(f"Saved baseline performance to {baseline_file}")
        
        # Generate baseline performance visualizations
        plt.figure(figsize=(12, 6))
        metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall', 'f1']
        
        for dataset in baseline_df['dataset'].unique():
            dataset_metrics = baseline_df[baseline_df['dataset'] == dataset][metrics_to_plot]
            means = dataset_metrics.mean()
            stds = dataset_metrics.std()
            
            x = np.arange(len(metrics_to_plot))
            plt.bar(x + 0.2 * (list(baseline_df['dataset'].unique()).index(dataset)),
                    means, yerr=stds, width=0.2, label=dataset, capsize=5)
        
        plt.title('Baseline Performance Metrics by Dataset')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.xticks(x + 0.2, metrics_to_plot, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        baseline_plot_file = base_dir / 'analysis' / 'baseline_performance.png'
        plt.savefig(baseline_plot_file)
        plt.close()
        print(f"Saved baseline performance plot to {baseline_plot_file}")
    
    # Create and save training progression data
    if train_metrics:
        progression_df = create_training_progression_df(train_metrics, configs)
        progression_file = base_dir / 'analysis' / 'training_progression.csv'
        progression_df.to_csv(progression_file, index=False)
        print(f"Saved training progression to {progression_file}")
        
        # Generate training curves plots
        metrics_to_plot = [
            ('loss', 'Training Loss'),
            ('accuracy', 'Accuracy'),
            ('f1', 'F1 Score'),
            ('time', 'Time per Epoch')
        ]
        
        for metric, title in metrics_to_plot:
            plt.figure(figsize=(12, 6))
            for dataset in progression_df['dataset'].unique():
                dataset_df = progression_df[progression_df['dataset'] == dataset]
                sns.lineplot(data=dataset_df, x='epoch', y=metric, label=f'{dataset}')
            
            plt.title(f'{title} Progression by Dataset')
            plt.xlabel('Epoch')
            plt.ylabel(title)
            plt.legend()
            
            plot_file = base_dir / 'analysis' / f'training_{metric}_curves.png'
            plt.savefig(plot_file)
            plt.close()
            print(f"Saved {title} curves plot to {plot_file}")
            
        # Generate correlation heatmap for metrics
        plt.figure(figsize=(10, 8))
        metric_cols = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'time']
        correlation = progression_df[metric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Metric Correlations')
        
        heatmap_file = base_dir / 'analysis' / 'metric_correlations.png'
        plt.savefig(heatmap_file)
        plt.close()
        print(f"Saved metric correlations heatmap to {heatmap_file}")
        print(f"Saved training curves plot to {plot_file}")
    
    # Generate comprehensive statistics
    stats_file = base_dir / 'analysis' / 'performance_analysis.txt'
    with open(stats_file, 'w') as f:
        f.write("=== Performance Analysis Report ===\n\n")
        
        # Per dataset statistics
        f.write("=== Dataset Statistics ===\n")
        datasets = df['dataset'].unique()
        if len(datasets) == 0 or (len(datasets) == 1 and datasets[0] == 'unknown'):
            f.write("\nNo dataset information found in configs\n")
            return
            
        for dataset in datasets:
            dataset_df = df[df['dataset'] == dataset]
            baseline_df = dataset_df[dataset_df['evaluation_type'] == 'baseline_eval']
            test_df = dataset_df[dataset_df['evaluation_type'] == 'test']
            
            f.write(f"\nDataset: {dataset}\n")
            f.write(f"Total runs: {len(dataset_df)}\n")
            f.write(f"Baseline runs: {len(baseline_df)}\n")
            f.write(f"Test runs: {len(test_df)}\n\n")
            
            if not baseline_df.empty:
                f.write("Baseline Metrics:\n")
                f.write(f"  Loss: {baseline_df['test_loss'].mean():.4f} ± {baseline_df['test_loss'].std():.4f}\n")
                f.write(f"  Accuracy: {baseline_df['test_accuracy'].mean():.4f} ± {baseline_df['test_accuracy'].std():.4f}\n")
            
            if not test_df.empty:
                f.write("\nTest Metrics:\n")
                f.write(f"  Loss: {test_df['test_loss'].mean():.4f} ± {test_df['test_loss'].std():.4f}\n")
                f.write(f"  Accuracy: {test_df['test_accuracy'].mean():.4f} ± {test_df['test_accuracy'].std():.4f}\n")
            
            if not test_df.empty and not baseline_df.empty:
                loss_improvement = baseline_df['test_loss'].mean() - test_df['test_loss'].mean()
                acc_improvement = test_df['test_accuracy'].mean() - baseline_df['test_accuracy'].mean()
                f.write("\nImprovements:\n")
                f.write(f"  Loss reduction: {loss_improvement:.4f}\n")
                f.write(f"  Accuracy gain: {acc_improvement:.4f}\n")
        
        # Training progression analysis
        f.write("\n=== Training Progression ===\n")
        for run_id, metrics in train_metrics.items():
            if metrics:
                f.write(f"\nRun {run_id}:\n")
                losses = [m.get('loss', float('nan')) for m in metrics]
                f.write(f"  Initial loss: {losses[0]:.4f}\n")
                f.write(f"  Final loss: {losses[-1]:.4f}\n")
                f.write(f"  Improvement: {losses[0] - losses[-1]:.4f}\n")
            dataset_df = df[df['dataset'] == dataset]
            f.write(f"\nDataset: {dataset}\n")
            f.write(f"Number of runs: {len(dataset_df)}\n")
            f.write(f"Average test loss: {dataset_df['test_loss'].mean():.4f} ± {dataset_df['test_loss'].std():.4f}\n")
            f.write(f"Average test accuracy: {dataset_df['test_accuracy'].mean():.4f} ± {dataset_df['test_accuracy'].std():.4f}\n")

def has_data(directory: Path) -> bool:
    """Check if a directory contains any data files."""
    if not directory.exists() or not directory.is_dir():
        return False
    
    # Check for any .json, .txt, .log, or .pt files
    for ext in ['*.json', '*.txt', '*.log', '*.pt']:
        if any(directory.rglob(ext)):
            return True
    return False

def analyze_runs(base_dir: Path) -> None:
    """Analyze all runs and generate performance reports."""
    print(f"Analyzing runs in {base_dir}...")
    
    train_metrics = {}
    test_metrics = {}
    configs = {}
    
    # Skip the analysis directory if it exists
    if (base_dir / "analysis").exists():
        shutil.rmtree(base_dir / "analysis")
    
    # Process each run directory
    for run_dir in sorted(base_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
            
        # Skip if neither train nor baseline has data
        train_dir = run_dir / "train"
        baseline_dir = run_dir / "baseline"
        if not has_data(train_dir) and not has_data(baseline_dir):
            continue
            
        run_id = run_dir.name
        print(f"Processing {run_id}...")
        
        # Process training data
        train_dir = run_dir / "train"
        if train_dir.exists():
            # Get training metrics
            metrics_dir = train_dir / "metrics"
            if metrics_dir.exists():
                metrics_files = sorted(metrics_dir.glob("metrics_epoch_*.json"))
                if metrics_files:
                    train_metrics[run_id] = []
                    for f in metrics_files:
                        metric_data = parse_metrics(f)
                        if metric_data:  # Only add if we got valid data
                            train_metrics[run_id].append(metric_data)
            
            # Get training config
            train_config = train_dir / "analysis" / "config.json"
            if train_config.exists() and run_id not in configs:
                config_data = parse_config(train_config)
                if config_data:  # Only add if we got valid data
                    configs[run_id] = config_data
        
        # Process baseline data
        baseline_dir = run_dir / "baseline"
        if baseline_dir.exists():
            # Get test metrics
            test_metrics_file = baseline_dir / "metrics" / "test_metrics.json"
            if test_metrics_file.exists():
                metric_data = parse_metrics(test_metrics_file)
                if metric_data:  # Only add if we got valid data
                    test_metrics[run_id] = metric_data
            
            # Get baseline config if we don't have a training config
            if run_id not in configs:
                baseline_config = baseline_dir / "analysis" / "config.json"
                if baseline_config.exists():
                    config_data = parse_config(baseline_config)
                    if config_data:  # Only add if we got valid data
                        configs[run_id] = config_data
    
    # Create analysis directory and its subdirectories
    analysis_dir = base_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Create train and baseline subdirectories in analysis
    for subdir in ["train", "baseline"]:
        for subtype in ["metrics", "weights", "analysis"]:
            (analysis_dir / subdir / subtype).mkdir(parents=True, exist_ok=True)
    
    # Generate summary report
    if train_metrics or test_metrics or configs:
        create_summary_report(base_dir, train_metrics, test_metrics, configs)
    else:
        print("No data found to analyze!")

def consolidate_runs(base_dir: Path):
    """Consolidate all training and baseline subdirectories under their respective directories."""
    print(f"\nConsolidating directory structure...")
    
    # First analyze the runs
    analyze_runs(base_dir)
    
    # Process each model directory
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name == "analysis":
            continue
            
        # Create train and baseline subdirectories
        train_dir = model_dir / "train"
        baseline_dir = model_dir / "baseline"
        
        for subdir in ["metrics", "weights", "analysis"]:
            (train_dir / subdir).mkdir(parents=True, exist_ok=True)
            (baseline_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Track moved runs to clean up later
        moved_runs = []
        
        # Process each run directory
        for run_dir in model_dir.glob("run_*"):
            if not run_dir.is_dir():
                continue
                
            try:
                run_num = run_dir.name
                
                # Process training subdirectories
                src_train = run_dir / "train"
                if src_train.exists() and src_train.is_dir():
                    for subdir in ["metrics", "weights", "analysis"]:
                        src_subdir = src_train / subdir
                        if src_subdir.exists() and src_subdir.is_dir():
                            dst_subdir = train_dir / subdir / run_num
                            if dst_subdir.exists():
                                shutil.rmtree(dst_subdir)
                            shutil.copytree(src_subdir, dst_subdir)
                
                # Process baseline subdirectories
                src_baseline = run_dir / "baseline"
                if src_baseline.exists() and src_baseline.is_dir():
                    for subdir in ["metrics", "weights", "analysis"]:
                        src_subdir = src_baseline / subdir
                        if src_subdir.exists() and src_subdir.is_dir():
                            dst_subdir = baseline_dir / subdir / run_num
                            if dst_subdir.exists():
                                shutil.rmtree(dst_subdir)
                            shutil.copytree(src_subdir, dst_subdir)
                
                moved_runs.append(run_dir)
                    
            except Exception as e:
                print(f"Error processing {run_dir}: {e}")
                continue
        
        # Clean up original run directories
        for run_dir in moved_runs:
            shutil.rmtree(run_dir)
    
    print("\nDirectory consolidation complete!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: consolidate_runs.py <results_dir>")
        sys.exit(1)
        
    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        sys.exit(1)
        
    consolidate_runs(results_dir)
