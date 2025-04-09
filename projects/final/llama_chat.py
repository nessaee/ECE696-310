#!/usr/bin/env python3
"""
Implementation of a multi-turn conversation interface for Llama models.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from llama_models import Llama7BModel
from llm_interface import LLMInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LlamaChat:
    """Chat interface for Llama models with conversation history support."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        local_model_path: Optional[str] = None,
        use_auth_token: Optional[str] = None,
        device: str = "cuda",
        offline_mode: bool = False,
        download_dir: Optional[str] = None,
        use_8bit: bool = False,
        use_4bit: bool = False,
        max_history: int = 5,
        **kwargs
    ):
        """Initialize the Llama chat interface.
        
        Args:
            model_name: Name or path of the model
            local_model_path: Path to a local model directory
            use_auth_token: Hugging Face token for model access
            device: Device to run the model on (default: "cuda")
            offline_mode: Whether to operate in offline mode (default: False)
            download_dir: Directory to download and cache models
            use_8bit: Whether to load model in 8-bit precision to save memory
            use_4bit: Whether to load model in 4-bit precision to save memory
            max_history: Maximum number of conversation turns to keep in history
            **kwargs: Additional model-specific parameters
        """
        self.max_history = max_history
        self.history = []
        self.system_prompt = kwargs.get("system_prompt", "You are a helpful, harmless, and honest AI assistant.")
        
        # Initialize the model
        logger.info(f"Initializing Llama model from: {local_model_path or model_name}")
        self.llm = Llama7BModel(
            model_name=model_name,
            local_model_path=local_model_path,
            use_auth_token=use_auth_token,
            device=device,
            offline_mode=offline_mode,
            download_dir=download_dir,
            use_8bit=use_8bit,
            use_4bit=use_4bit,
            **{k: v for k, v in kwargs.items() if k != "system_prompt"}
        )
        
        logger.info("Chat interface initialized successfully")
    
    def _format_chat_prompt(self) -> str:
        """Format the conversation history into a prompt for the model.
        
        Returns:
            Formatted prompt string
        """
        # Start with the system prompt
        formatted_prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n\n"
        
        # Add conversation history
        for turn in self.history:
            role = turn["role"]
            content = turn["content"]
            formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n\n"
        
        # Add assistant prefix for the next response
        formatted_prompt += "<|im_start|>assistant\n"
        
        return formatted_prompt
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.
        
        Args:
            role: Message role (user or assistant)
            content: Message content
        """
        self.history.append({"role": role, "content": content})
        
        # Trim history if it exceeds max_history
        if len(self.history) > self.max_history * 2:  # *2 because each turn has user and assistant messages
            # Keep the most recent messages
            self.history = self.history[-self.max_history * 2:]
    
    def chat(self, user_message: str, **kwargs) -> Dict[str, Any]:
        """Process a user message and generate a response.
        
        Args:
            user_message: User input message
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Add user message to history
        self.add_message("user", user_message)
        
        # Format the prompt with conversation history
        prompt = self._format_chat_prompt()
        
        # Generate response
        generation_config = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_length": kwargs.get("max_length", 512),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50)
        }
        
        # Update with any additional kwargs
        generation_config.update({k: v for k, v in kwargs.items() 
                                if k not in ["temperature", "max_length", "top_p", "top_k"]})
        
        # Generate response
        logger.info("Generating response from language model")
        llm_result = self.llm.generate(prompt, **generation_config)
        
        # Extract the assistant's response
        response_text = llm_result["response"]
        
        # Clean up the response if needed
        if "<|im_end|>" in response_text:
            response_text = response_text.split("<|im_end|>")[0].strip()
        
        # Add assistant response to history
        self.add_message("assistant", response_text)
        
        # Prepare result
        result = {
            "response": response_text,
            "model_name": self.llm.model_name,
            "generation_config": generation_config
        }
        
        return result
    
    def reset_chat(self) -> None:
        """Reset the conversation history."""
        self.history = []
        logger.info("Chat history has been reset")
    
    def save_conversation(self, file_path: str) -> None:
        """Save the conversation history to a file.
        
        Args:
            file_path: Path to save the conversation history
        """
        with open(file_path, 'w') as f:
            json.dump({
                "system_prompt": self.system_prompt,
                "history": self.history
            }, f, indent=2)
        logger.info(f"Conversation saved to {file_path}")
    
    def load_conversation(self, file_path: str) -> None:
        """Load a conversation history from a file.
        
        Args:
            file_path: Path to the conversation history file
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.system_prompt = data.get("system_prompt", self.system_prompt)
            self.history = data.get("history", [])
        logger.info(f"Conversation loaded from {file_path}")


def main():
    """Simple demonstration of the LlamaChat class."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a simple chat demo with Llama model")
    
    # Model configuration
    parser.add_argument(
        "--model", 
        type=str, 
        default="meta-llama/Llama-3.2-1B-Instruct",
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
        default="cuda",
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
        default="You are a helpful, harmless, and honest AI assistant.",
        help="System prompt to use for the conversation"
    )
    
    args = parser.parse_args()
    
    # Get token from args or environment
    token = args.token or os.environ.get("HF_TOKEN")
    
    # Initialize chat
    chat = LlamaChat(
        model_name=args.model,
        local_model_path=args.local_path,
        use_auth_token=token,
        device=args.device,
        offline_mode=args.offline,
        download_dir=args.download_dir,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        system_prompt=args.system_prompt
    )
    
    print("\nWelcome to Llama Chat!")
    print("Type 'exit', 'quit', or 'bye' to end the conversation.")
    print("Type 'reset' to reset the conversation history.")
    print("Type 'save <filename>' to save the conversation.")
    print("Type 'load <filename>' to load a conversation.\n")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        # Check for special commands
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        elif user_input.lower() == "reset":
            chat.reset_chat()
            print("Chat history has been reset.")
            continue
        elif user_input.lower().startswith("save "):
            filename = user_input[5:].strip()
            chat.save_conversation(filename)
            print(f"Conversation saved to {filename}")
            continue
        elif user_input.lower().startswith("load "):
            filename = user_input[5:].strip()
            if os.path.exists(filename):
                chat.load_conversation(filename)
                print(f"Conversation loaded from {filename}")
            else:
                print(f"File not found: {filename}")
            continue
        
        # Process regular chat message
        result = chat.chat(user_input)
        print(f"\nAssistant: {result['response']}")


if __name__ == "__main__":
    main()
