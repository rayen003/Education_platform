"""
OpenAI LLM Service.

This module contains the OpenAI implementation of the LLM service interface.
"""

import os
import logging
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from .base_service import BaseLLMService

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class OpenAILLMService(BaseLLMService):
    """OpenAI implementation of the LLM service."""
    
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the OpenAI LLM service.
        
        Args:
            model: The OpenAI model to use
        """
        super().__init__()
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info(f"Initialized OpenAILLMService with model {model}")
    
    def generate_completion(self, system_prompt: str, user_prompt: str, 
                          temperature: float = 0.7, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a completion using the OpenAI API.
        
        Args:
            system_prompt: The system prompt to use
            user_prompt: The user prompt to use
            temperature: The temperature to use for generation
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            Response dictionary with content and metadata
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens if max_tokens else 1000
            )
            
            # Convert usage to a dictionary to avoid attribute access issues
            usage_dict = {}
            if hasattr(response, 'usage') and response.usage:
                usage_dict = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            
            return {
                "content": response.choices[0].message.content,
                "finish_reason": response.choices[0].finish_reason,
                "model": self.model,
                "usage": usage_dict
            }
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            return {
                "error": str(e),
                "content": None
            }
