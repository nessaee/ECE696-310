#!/usr/bin/env python3
"""
Script to download Llama models for offline use.
"""
import os
import argparse
import logging
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download Llama models for offline use")
    
    parser.add_argument(
        "--token", 
        type=str, 
        default=None,
        help="Hugging Face token for model access"
    )
    parser.add_argument(
        "--download-dir", 
        type=str, 
        default=None,
        help="Directory to download and cache models"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the models on"
    )
    
    return parser.parse_args()

def download_llama_models_directly(token, device, download_dir=None):
    """Download Llama-3.2-1B-Instruct and Llama-Guard-3-8B models directly and save them locally.
    
    Args:
        token: Hugging Face token for model access
        device: Device to run the model on
        download_dir: Directory to download and cache models
    
    Returns:
        tuple: Paths to the saved model directories (llm_path, guard_path)
    """
    # Set download directory if provided
    if download_dir:
        os.environ['TRANSFORMERS_CACHE'] = str(download_dir)
        logger.info(f"Set Transformers cache directory to: {download_dir}")
        base_dir = Path(download_dir)
    else:
        base_dir = Path.home() / ".cache" / "huggingface"
        os.environ['TRANSFORMERS_CACHE'] = str(base_dir)
    
    # Create specific directories for each model
    llm_dir = base_dir / "llama-3.2-1b-instruct"
    guard_dir = base_dir / "llama-guard-3-8b"
    
    llm_dir.mkdir(parents=True, exist_ok=True)
    guard_dir.mkdir(parents=True, exist_ok=True)
    
    llm_saved = False
    guard_saved = False
    
    # Download Llama-3.2-1B-Instruct
    logger.info("Downloading Llama-3.2-1B-Instruct directly")
    try:
        # Load model directly
        logger.info("Loading Llama-3.2-1B-Instruct tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_auth_token=token)
        
        logger.info("Loading Llama-3.2-1B-Instruct model")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_auth_token=token
        )
        
        # Save model and tokenizer to local directory
        logger.info(f"Saving Llama-3.2-1B-Instruct to {llm_dir}")
        model.save_pretrained(str(llm_dir))
        tokenizer.save_pretrained(str(llm_dir))
        
        logger.info("Successfully downloaded and saved Llama-3.2-1B-Instruct locally")
        llm_saved = True
    except Exception as e:
        logger.error(f"Failed to download Llama-3.2-1B-Instruct: {e}")
    
    # Download Llama-Guard-3-8B
    logger.info("Downloading Llama-Guard-3-8B directly")
    try:
        # Load model directly
        logger.info("Loading Llama-Guard-3-8B tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-Guard-3-8B", use_auth_token=token)
        
        logger.info("Loading Llama-Guard-3-8B model")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-Guard-3-8B",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_auth_token=token
        )
        
        # Save model and tokenizer to local directory
        logger.info(f"Saving Llama-Guard-3-8B to {guard_dir}")
        model.save_pretrained(str(guard_dir))
        tokenizer.save_pretrained(str(guard_dir))
        
        logger.info("Successfully downloaded and saved Llama-Guard-3-8B locally")
        guard_saved = True
    except Exception as e:
        logger.error(f"Failed to download Llama-Guard-3-8B: {e}")
    
    return (str(llm_dir) if llm_saved else None, str(guard_dir) if guard_saved else None)

def main():
    """Main function."""
    args = parse_args()
    
    # Get token from args or environment
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        logger.error(
            "No Hugging Face token provided. This will cause authentication errors "
            "when downloading the models. Set the token using the --token argument "
            "or the HF_TOKEN environment variable."
        )
        return
    
    # Create download directory if provided
    if args.download_dir:
        download_dir = Path(args.download_dir).expanduser().absolute()
        download_dir.mkdir(parents=True, exist_ok=True)
        # Convert to string for later use
        download_dir = str(download_dir)
    else:
        download_dir = None
    
    # Download models directly using the provided code
    logger.info("=== Downloading Llama models directly ===")
    llm_path, guard_path = download_llama_models_directly(token, args.device, download_dir)
    
    # Get cache directory for information
    cache_dir = os.environ.get('TRANSFORMERS_CACHE', os.path.join(os.path.expanduser('~'), '.cache', 'huggingface'))
    logger.info(f"\nModels downloaded to cache directory: {cache_dir}")
    
    # Print paths to the saved models
    if llm_path:
        logger.info(f"Llama-3.2-1B-Instruct saved to: {llm_path}")
    if guard_path:
        logger.info(f"Llama-Guard-3-8B saved to: {guard_path}")
    
    # Create a config file to store the model paths
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama_config.json")
    config = {
        "llm_path": llm_path,
        "guard_path": guard_path,
        "download_dir": download_dir or cache_dir
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\nSaved model paths to config file: {config_path}")
    
    # Print instructions for using the directly downloaded models
    logger.info(
        "\nTo use the downloaded models in offline mode:\n"
        f"python run_llama_pipeline.py --offline "
        f"--llm-local-path {llm_path} "
        f"--moderation-local-path {guard_path} "
        "--prompt \"Your prompt here\""
    )

if __name__ == "__main__":
    main()
