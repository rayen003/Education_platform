"""
OpenAI LLM Service.

This module contains the OpenAI implementation of the LLM service interface.
"""

import os
import logging
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

from app.math.services.llm.base_service import BaseLLMService

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
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info(f"Initialized OpenAILLMService with model {model}")
    
    def generate_completion(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Generate a completion using the OpenAI API.
        
        Args:
            system_prompt: The system prompt to use
            user_prompt: The user prompt to use
            
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
                temperature=0.7,
                max_tokens=1000
            )
            
            return {
                "content": response.choices[0].message.content,
                "model": self.model,
                "tokens": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            return {
                "content": f"Error: {str(e)}",
                "model": self.model,
                "error": str(e)
            }
