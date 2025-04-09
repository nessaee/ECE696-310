#!/usr/bin/env python3
"""
Simple command-line interface for the Llama chat model.
Provides a more user-friendly experience with colored output and command history.
"""
import os
import sys
import json
import argparse
import readline
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from llama_chat import LlamaChat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
COLORS = {
    'reset': '\033[0m',
    'bold': '\033[1m',
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'magenta': '\033[95m',
    'cyan': '\033[96m',
}


def colorize(text: str, color: str) -> str:
    """Add color to terminal text.
    
    Args:
        text: Text to colorize
        color: Color name from COLORS dict
        
    Returns:
        Colorized text string
    """
    if color not in COLORS:
        return text
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "local_model_path": None,
        "device": "cuda",
        "offline_mode": False,
        "use_8bit": False,
        "use_4bit": False,
        "system_prompt": "You are a helpful, harmless, and honest AI assistant.",
        "temperature": 0.7,
        "max_length": 512,
        "top_p": 0.9,
        "top_k": 50,
        "max_history": 5
    }
    
    if not config_path or not os.path.exists(config_path):
        return default_config
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update default config with loaded values
        default_config.update(config)
        logger.info(f"Configuration loaded from {config_path}")
        return default_config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return default_config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")


def setup_readline_history(history_file: str = "~/.llama_chat_history") -> None:
    """Set up readline history for command recall.
    
    Args:
        history_file: Path to the history file
    """
    history_path = os.path.expanduser(history_file)
    try:
        if os.path.exists(history_path):
            readline.read_history_file(history_path)
        
        # Set maximum number of history items
        readline.set_history_length(1000)
    except Exception as e:
        logger.warning(f"Could not set up readline history: {e}")
    
    # Save history on exit
    import atexit
    atexit.register(lambda: readline.write_history_file(history_path))


def print_help() -> None:
    """Print help information for available commands."""
    help_text = """
Available Commands:
------------------
/help               - Show this help message
/exit, /quit, /bye  - Exit the chat
/reset              - Reset the conversation history
/save <filename>    - Save the conversation to a file
/load <filename>    - Load a conversation from a file
/config             - Show current configuration
/config save <file> - Save current configuration to a file
/config load <file> - Load configuration from a file
/system "<prompt>"  - Change the system prompt
/temp <value>       - Change the temperature (0.0-1.0)
/length <value>     - Change the max response length
"""
    print(colorize(help_text, "cyan"))


def main():
    """Run the chat interface."""
    parser = argparse.ArgumentParser(description="Interactive chat interface for Llama models")
    
    # Basic configuration
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to a configuration file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Name or path of the language model"
    )
    parser.add_argument(
        "--local-path", 
        type=str, 
        default=None,
        help="Local path to the language model directory"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        default=None,
        help="Hugging Face token for model access"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run the models on"
    )
    parser.add_argument(
        "--use-8bit", 
        action="store_true",
        help="Load models in 8-bit precision to reduce memory usage"
    )
    parser.add_argument(
        "--use-4bit", 
        action="store_true",
        help="Load models in 4-bit precision to reduce memory usage even further"
    )
    parser.add_argument(
        "--offline", 
        action="store_true",
        help="Run in offline mode, using only locally available models"
    )
    parser.add_argument(
        "--download-dir", 
        type=str, 
        default=None,
        help="Directory to download and cache models"
    )
    parser.add_argument(
        "--system-prompt", 
        type=str, 
        default=None,
        help="System prompt to use for the conversation"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments if provided
    if args.model:
        config["model_name"] = args.model
    if args.local_path:
        config["local_model_path"] = args.local_path
    if args.device:
        config["device"] = args.device
    if args.offline:
        config["offline_mode"] = True
    if args.use_8bit:
        config["use_8bit"] = True
    if args.use_4bit:
        config["use_4bit"] = True
    if args.system_prompt:
        config["system_prompt"] = args.system_prompt
    
    # Get token from args or environment
    token = args.token or os.environ.get("HF_TOKEN")
    
    # Set up command history
    setup_readline_history()
    
    # Initialize chat
    try:
        chat = LlamaChat(
            model_name=config["model_name"],
            local_model_path=config["local_model_path"],
            use_auth_token=token,
            device=config["device"],
            offline_mode=config["offline_mode"],
            download_dir=args.download_dir,
            use_8bit=config["use_8bit"],
            use_4bit=config["use_4bit"],
            system_prompt=config["system_prompt"],
            max_history=config["max_history"]
        )
    except Exception as e:
        print(colorize(f"Error initializing chat: {e}", "red"))
        sys.exit(1)
    
    # Welcome message
    print(colorize("\n=== Llama Chat Interface ===", "bold"))
    print(colorize("Type /help for available commands\n", "yellow"))
    print(colorize(f"Model: {config['model_name'] if not config['local_model_path'] else config['local_model_path']}", "green"))
    print(colorize(f"Device: {config['device']}", "green"))
    print(colorize(f"System prompt: \"{config['system_prompt']}\"", "green"))
    
    # Main chat loop
    while True:
        try:
            user_input = input(colorize("\nYou: ", "blue")).strip()
            
            # Check for commands (starting with /)
            if user_input.startswith("/"):
                cmd_parts = user_input[1:].split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                args = cmd_parts[1] if len(cmd_parts) > 1 else ""
                
                if cmd in ["exit", "quit", "bye"]:
                    print(colorize("Goodbye!", "yellow"))
                    break
                    
                elif cmd == "help":
                    print_help()
                    continue
                    
                elif cmd == "reset":
                    chat.reset_chat()
                    print(colorize("Chat history has been reset.", "yellow"))
                    continue
                    
                elif cmd == "save" and args:
                    chat.save_conversation(args)
                    print(colorize(f"Conversation saved to {args}", "yellow"))
                    continue
                    
                elif cmd == "load" and args:
                    if os.path.exists(args):
                        chat.load_conversation(args)
                        print(colorize(f"Conversation loaded from {args}", "yellow"))
                    else:
                        print(colorize(f"File not found: {args}", "red"))
                    continue
                    
                elif cmd == "config":
                    if not args:
                        # Show current config
                        print(colorize("Current configuration:", "cyan"))
                        for key, value in config.items():
                            print(colorize(f"  {key}: {value}", "cyan"))
                    elif args.startswith("save "):
                        filename = args[5:].strip()
                        save_config(config, filename)
                        print(colorize(f"Configuration saved to {filename}", "yellow"))
                    elif args.startswith("load "):
                        filename = args[5:].strip()
                        if os.path.exists(filename):
                            new_config = load_config(filename)
                            config.update(new_config)
                            print(colorize(f"Configuration loaded from {filename}", "yellow"))
                            # Reinitialize chat with new config
                            chat = LlamaChat(
                                model_name=config["model_name"],
                                local_model_path=config["local_model_path"],
                                use_auth_token=token,
                                device=config["device"],
                                offline_mode=config["offline_mode"],
                                download_dir=args.download_dir,
                                use_8bit=config["use_8bit"],
                                use_4bit=config["use_4bit"],
                                system_prompt=config["system_prompt"],
                                max_history=config["max_history"]
                            )
                        else:
                            print(colorize(f"File not found: {filename}", "red"))
                    continue
                    
                elif cmd == "system":
                    if args:
                        config["system_prompt"] = args.strip('"')
                        chat.system_prompt = config["system_prompt"]
                        print(colorize(f"System prompt updated: \"{config['system_prompt']}\"", "yellow"))
                    else:
                        print(colorize(f"Current system prompt: \"{chat.system_prompt}\"", "yellow"))
                    continue
                    
                elif cmd == "temp":
                    try:
                        if args:
                            temp = float(args)
                            if 0.0 <= temp <= 1.0:
                                config["temperature"] = temp
                                print(colorize(f"Temperature set to {temp}", "yellow"))
                            else:
                                print(colorize("Temperature must be between 0.0 and 1.0", "red"))
                        else:
                            print(colorize(f"Current temperature: {config['temperature']}", "yellow"))
                    except ValueError:
                        print(colorize("Invalid temperature value", "red"))
                    continue
                    
                elif cmd == "length":
                    try:
                        if args:
                            length = int(args)
                            if length > 0:
                                config["max_length"] = length
                                print(colorize(f"Max length set to {length}", "yellow"))
                            else:
                                print(colorize("Max length must be positive", "red"))
                        else:
                            print(colorize(f"Current max length: {config['max_length']}", "yellow"))
                    except ValueError:
                        print(colorize("Invalid length value", "red"))
                    continue
                
                else:
                    print(colorize(f"Unknown command: {cmd}. Type /help for available commands.", "red"))
                    continue
            
            # Process regular chat message
            if not user_input:
                continue
                
            print(colorize("Assistant is thinking...", "yellow"))
            result = chat.chat(
                user_input,
                temperature=config["temperature"],
                max_length=config["max_length"],
                top_p=config["top_p"],
                top_k=config["top_k"]
            )
            
            print(colorize("\nAssistant:", "green"), end=" ")
            print(result["response"])
            
        except KeyboardInterrupt:
            print(colorize("\nInterrupted. Type /exit to quit.", "yellow"))
            continue
        except Exception as e:
            print(colorize(f"\nError: {e}", "red"))
            continue


if __name__ == "__main__":
    main()
