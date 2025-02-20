"""
Configuration for logging across the project.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(experiment_name: str = None) -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        experiment_name (str, optional): Name of the experiment for log file naming.
            If None, uses timestamp.
            
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Generate log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{experiment_name}_{timestamp}" if experiment_name else timestamp
    log_file = log_dir / f"{log_name}.log"
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger("llm_finetuning")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
