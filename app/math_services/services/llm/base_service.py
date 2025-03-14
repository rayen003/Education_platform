"""
Base LLM Service Interface.

This module defines the interface for LLM services used by the math agent.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseLLMService(ABC):
    """Base interface for LLM services."""
    
    @abstractmethod
    def generate_completion(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Generate a completion based on the provided prompts.
        
        Args:
            system_prompt: The system prompt to use
            user_prompt: The user prompt to use
            
        Returns:
            Response dictionary with content and metadata
        """
        pass
