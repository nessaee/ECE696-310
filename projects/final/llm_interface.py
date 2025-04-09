"""
Base interfaces and abstract classes for LLM interactions.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any


class LLMInterface(ABC):
    """Abstract base class for language model interfaces."""
    
    @abstractmethod
    def __init__(self, model_name: str, **kwargs):
        """Initialize the model interface.
        
        Args:
            model_name: Name or path of the model
            **kwargs: Additional model-specific parameters
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response for the given prompt.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing the response and metadata
        """
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts.
        
        Args:
            prompts: List of input text prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of dictionaries containing responses and metadata
        """
        pass


class ContentModerationInterface(ABC):
    """Abstract base class for content moderation interfaces."""
    
    @abstractmethod
    def __init__(self, model_name: str, **kwargs):
        """Initialize the moderation interface.
        
        Args:
            model_name: Name or path of the moderation model
            **kwargs: Additional model-specific parameters
        """
        pass
    
    @abstractmethod
    def moderate(self, content: str, **kwargs) -> Dict[str, Any]:
        """Moderate the given content.
        
        Args:
            content: Text content to moderate
            **kwargs: Additional moderation parameters
            
        Returns:
            Dictionary containing moderation results and metadata
        """
        pass
    
    @abstractmethod
    def batch_moderate(self, contents: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Moderate multiple content items.
        
        Args:
            contents: List of text content to moderate
            **kwargs: Additional moderation parameters
            
        Returns:
            List of dictionaries containing moderation results and metadata
        """
        pass
