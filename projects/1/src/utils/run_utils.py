"""
Utilities for managing experiment runs.
"""
import os
from pathlib import Path
import re

def get_next_run_number(model_dir: Path) -> int:
    """
    Get the next run number by scanning existing run directories.
    
    Args:
        model_dir: Path to model's base directory
        
    Returns:
        Next run number to use
    """
    if not model_dir.exists():
        return 1
        
    # Look for run_* directories
    run_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if not run_dirs:
        return 1
        
    # Extract run numbers and find max
    run_numbers = []
    for d in run_dirs:
        match = re.match(r'run_(\d+)', d.name)
        if match:
            run_numbers.append(int(match.group(1)))
            
    return max(run_numbers) + 1 if run_numbers else 1
