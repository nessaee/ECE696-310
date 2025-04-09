"""
Implementation of Llama 7B and Llama Guard model interfaces with support for offline usage.
"""
import os
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from llm_interface import LLMInterface, ContentModerationInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Llama7BModel(LLMInterface):
    """Implementation of Llama 7B model interface with offline support."""
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-hf", 
                 device: str = "cuda", 
                 local_model_path: Optional[str] = None,
                 offline_mode: bool = False,
                 download_dir: Optional[str] = None,
                 **kwargs):
        """Initialize the Llama 7B model.
        
        Args:
            model_name: Name or path of the model (default: "meta-llama/Llama-2-7b-hf")
            device: Device to run the model on (default: "cuda")
            local_model_path: Path to a local model directory (overrides model_name if provided)
            offline_mode: Whether to operate in offline mode (default: False)
            download_dir: Directory to download and cache models (default: ~/.cache/huggingface)
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.device = device
        self.offline_mode = offline_mode
        self.use_auth_token = kwargs.get("use_auth_token", None)
        self.local_model_path = local_model_path
        
        # Set download directory if provided
        if download_dir:
            os.environ['TRANSFORMERS_CACHE'] = download_dir
            logger.info(f"Set Transformers cache directory to: {download_dir}")
        
        # Determine model path (local or remote)
        model_path = self.local_model_path if self.local_model_path else self.model_name
        
        # Load model and tokenizer
        try:
            logger.info(f"Loading model from: {model_path}")
            
            # Set up loading parameters based on mode
            load_kwargs = {
                'local_files_only': self.offline_mode,
            }
            
            # Only add auth token if not in offline mode and not using local path
            if not self.offline_mode and not self.local_model_path and self.use_auth_token:
                load_kwargs['use_auth_token'] = self.use_auth_token
            
            # First check if the model exists locally when in offline mode
            if self.offline_mode and not self.local_model_path:
                # Try to find the model in the cache directory
                cache_dir = os.environ.get('TRANSFORMERS_CACHE', os.path.join(os.path.expanduser('~'), '.cache', 'huggingface'))
                potential_model_dir = os.path.join(cache_dir, 'models--' + self.model_name.replace('/', '--'))
                
                if os.path.exists(potential_model_dir):
                    logger.info(f"Found model in cache: {potential_model_dir}")
                    model_path = potential_model_dir
                else:
                    raise ValueError(
                        f"Model {self.model_name} not found locally and offline_mode is enabled. "
                        f"Please provide a valid local_model_path or disable offline_mode."
                    )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                **load_kwargs
            )
            
            # Load model with memory optimization options
            use_8bit = kwargs.get("use_8bit", False)
            use_4bit = kwargs.get("use_4bit", False)
            
            # Set environment variable for memory management
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            # Determine quantization and loading options
            model_kwargs = {
                **load_kwargs,
                **{k: v for k, v in kwargs.items() if k not in ["use_auth_token", "use_8bit", "use_4bit"]}
            }
            
            if device == "cuda":
                if use_8bit:
                    # Load in 8-bit precision to save memory
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        quantization_config=quantization_config,
                        **model_kwargs
                    )
                elif use_4bit:
                    # Load in 4-bit precision to save even more memory
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        quantization_config=quantization_config,
                        **model_kwargs
                    )
                else:
                    # Standard loading with float16
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        **model_kwargs
                    )
            else:
                # CPU loading with float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    **model_kwargs
                )
            
            logger.info(f"Successfully loaded model and tokenizer from {model_path}")
            
        except Exception as e:
            if self.offline_mode:
                raise ValueError(
                    f"Error loading model in offline mode: {e}. "
                    f"Please ensure the model is available locally at {model_path}."
                )
            elif "401" in str(e):
                raise ValueError(
                    f"Authentication error: {e}. Please provide a valid Hugging Face token "
                    "using the 'use_auth_token' parameter."
                )
            elif "404" in str(e):
                raise ValueError(
                    f"Model not found: {e}. Please check if the model name is correct "
                    "or if you have access to this model."
                )
            else:
                raise e
    
    def generate(self, prompt: str, max_length: int = 512, **kwargs) -> Dict[str, Any]:
        """Generate a response for the given prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of the generated text
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing the response and metadata
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generation_config = {
            "max_length": max_length,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "num_return_sequences": 1,
            "do_sample": kwargs.get("do_sample", True),
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Update with any additional kwargs
        generation_config.update({k: v for k, v in kwargs.items() 
                                if k not in ["temperature", "top_p", "top_k", "do_sample"]})
        
        with torch.no_grad():
            output = self.model.generate(**inputs, **generation_config)
        
        response_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove the prompt from the response if it's included
        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):].strip()
        
        return {
            "response": response_text,
            "model_name": self.model_name,
            "prompt": prompt,
            "generation_config": generation_config
        }
    
    def batch_generate(self, prompts: List[str], max_length: int = 512, **kwargs) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts.
        
        Args:
            prompts: List of input text prompts
            max_length: Maximum length of the generated text
            **kwargs: Additional generation parameters
            
        Returns:
            List of dictionaries containing responses and metadata
        """
        return [self.generate(prompt, max_length, **kwargs) for prompt in prompts]


class LlamaGuardModel(ContentModerationInterface):
    """Implementation of Llama Guard model interface for content moderation with offline support."""
    
    def __init__(self, 
                 model_name: str = "meta-llama/LlamaGuard-7b", 
                 device: str = "cuda", 
                 local_model_path: Optional[str] = None,
                 offline_mode: bool = False,
                 download_dir: Optional[str] = None,
                 **kwargs):
        """Initialize the Llama Guard model.
        
        Args:
            model_name: Name or path of the model (default: "meta-llama/LlamaGuard-7b")
            device: Device to run the model on (default: "cuda")
            local_model_path: Path to a local model directory (overrides model_name if provided)
            offline_mode: Whether to operate in offline mode (default: False)
            download_dir: Directory to download and cache models (default: ~/.cache/huggingface)
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.device = device
        self.offline_mode = offline_mode
        self.use_auth_token = kwargs.get("use_auth_token", None)
        self.local_model_path = local_model_path
        
        # Set download directory if provided
        if download_dir:
            os.environ['TRANSFORMERS_CACHE'] = download_dir
            logger.info(f"Set Transformers cache directory to: {download_dir}")
        
        # Determine model path (local or remote)
        model_path = self.local_model_path if self.local_model_path else self.model_name
        
        # Load model and tokenizer
        try:
            logger.info(f"Loading moderation model from: {model_path}")
            
            # Set up loading parameters based on mode
            load_kwargs = {
                'local_files_only': self.offline_mode,
            }
            
            # Only add auth token if not in offline mode and not using local path
            if not self.offline_mode and not self.local_model_path and self.use_auth_token:
                load_kwargs['use_auth_token'] = self.use_auth_token
            
            # First check if the model exists locally when in offline mode
            if self.offline_mode and not self.local_model_path:
                # Try to find the model in the cache directory
                cache_dir = os.environ.get('TRANSFORMERS_CACHE', os.path.join(os.path.expanduser('~'), '.cache', 'huggingface'))
                potential_model_dir = os.path.join(cache_dir, 'models--' + self.model_name.replace('/', '--'))
                
                if os.path.exists(potential_model_dir):
                    logger.info(f"Found moderation model in cache: {potential_model_dir}")
                    model_path = potential_model_dir
                else:
                    raise ValueError(
                        f"Moderation model {self.model_name} not found locally and offline_mode is enabled. "
                        f"Please provide a valid local_model_path or disable offline_mode."
                    )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                **load_kwargs
            )
            
            # Load model with memory optimization options
            use_8bit = kwargs.get("use_8bit", False)
            use_4bit = kwargs.get("use_4bit", False)
            
            # Set environment variable for memory management
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            # Determine quantization and loading options
            model_kwargs = {
                **load_kwargs,
                **{k: v for k, v in kwargs.items() if k not in ["use_auth_token", "use_8bit", "use_4bit"]}
            }
            
            if device == "cuda":
                if use_8bit:
                    # Load in 8-bit precision to save memory
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        quantization_config=quantization_config,
                        **model_kwargs
                    )
                elif use_4bit:
                    # Load in 4-bit precision to save even more memory
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        quantization_config=quantization_config,
                        **model_kwargs
                    )
                else:
                    # Standard loading with float16
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        **model_kwargs
                    )
            else:
                # CPU loading with float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    **model_kwargs
                )
            
            # Set up the pipeline for easier use
            # When using device_map="auto", we should not specify device in the pipeline
            if device == "cuda":
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer
                    # No device parameter when using device_map="auto"
                )
            else:
                # For CPU, we can specify the device
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1
                )
            
            logger.info(f"Successfully loaded moderation model and tokenizer from {model_path}")
            
        except Exception as e:
            if self.offline_mode:
                raise ValueError(
                    f"Error loading moderation model in offline mode: {e}. "
                    f"Please ensure the model is available locally at {model_path}."
                )
            elif "401" in str(e):
                raise ValueError(
                    f"Authentication error: {e}. Please provide a valid Hugging Face token "
                    "using the 'use_auth_token' parameter."
                )
            elif "404" in str(e):
                raise ValueError(
                    f"Model not found: {e}. Please check if the model name is correct "
                    "or if you have access to this model."
                )
            else:
                raise e
        
        # Define harm categories based on Llama Guard documentation
        self.harm_categories = [
            "Violence and Hate",
            "Sexual Content",
            "Criminal Planning",
            "Guns and Illegal Weapons",
            "Regulated or Controlled Substances",
            "Self-Harm",
        ]
    
    def _format_moderation_prompt(self, content: str) -> str:
        """Format the content for moderation.
        
        Args:
            content: Text content to moderate
            
        Returns:
            Formatted prompt for the moderation model
        """
        prompt_template = f"""<|im_start|>user
Please assess whether the following content is harmful. Content: {content}<|im_end|>
<|im_start|>assistant
"""
        return prompt_template
    
    def _parse_moderation_response(self, response: str) -> Dict[str, Any]:
        """Parse the moderation response.
        
        Args:
            response: Raw response from the model
            
        Returns:
            Parsed moderation results
        """
        # Extract the assistant's response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[1].strip()
        
        # Check for harmful content detection
        is_harmful = "harmful" in response.lower() and not "not harmful" in response.lower()
        
        # Identify specific harm categories
        detected_categories = []
        for category in self.harm_categories:
            if category.lower() in response.lower():
                detected_categories.append(category)
        
        return {
            "is_harmful": is_harmful,
            "detected_categories": detected_categories,
            "full_response": response
        }
    
    def moderate(self, content: str, max_length: int = 256, **kwargs) -> Dict[str, Any]:
        """Moderate the given content.
        
        Args:
            content: Text content to moderate
            max_length: Maximum length of the generated text
            **kwargs: Additional moderation parameters
            
        Returns:
            Dictionary containing moderation results and metadata
        """
        prompt = self._format_moderation_prompt(content)
        
        generation_config = {
            "max_new_tokens": max_length,
            "temperature": kwargs.get("temperature", 0.1),  # Low temperature for more deterministic outputs
            "top_p": kwargs.get("top_p", 0.9),
            "do_sample": kwargs.get("do_sample", False),  # Deterministic by default
        }
        
        # Update with any additional kwargs
        generation_config.update({k: v for k, v in kwargs.items() 
                                if k not in ["temperature", "top_p", "do_sample"]})
        
        # Generate response
        response = self.pipe(prompt, **generation_config)[0]["generated_text"]
        
        # Parse the response
        moderation_result = self._parse_moderation_response(response)
        
        # Add metadata
        moderation_result.update({
            "model_name": self.model_name,
            "content": content,
            "generation_config": generation_config
        })
        
        return moderation_result
    
    def batch_moderate(self, contents: List[str], max_length: int = 256, **kwargs) -> List[Dict[str, Any]]:
        """Moderate multiple content items.
        
        Args:
            contents: List of text content to moderate
            max_length: Maximum length of the generated text
            **kwargs: Additional moderation parameters
            
        Returns:
            List of dictionaries containing moderation results and metadata
        """
        return [self.moderate(content, max_length, **kwargs) for content in contents]
