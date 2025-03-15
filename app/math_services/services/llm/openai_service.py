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
    
    def __init__(self, model="gpt-4o-mini", mock_mode=False):
        """
        Initialize the OpenAI LLM service.
        
        Args:
            model: The OpenAI model to use
            mock_mode: Whether to use mock mode when API key isn't available
        """
        super().__init__()
        self.model = model
        self.mock_mode = mock_mode
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and not mock_mode:
            logger.warning("OPENAI_API_KEY not found. Switching to mock mode.")
            self.mock_mode = True
            
        if not self.mock_mode:
            self.client = OpenAI(api_key=api_key)
            logger.info(f"Initialized OpenAILLMService with model {model}")
        else:
            logger.info(f"Initialized OpenAILLMService in mock mode")
    
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
            if self.mock_mode:
                return self._generate_mock_response(user_prompt)
                
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
            return self._generate_mock_response(user_prompt)
    
    def _generate_mock_response(self, user_prompt: str) -> Dict[str, Any]:
        """Generate a mock response for testing purposes."""
        logger.info("Using mock LLM response")
        
        # Extract keywords from the prompt for simple pattern matching
        prompt_lower = user_prompt.lower()
        
        # For math feedback
        if "feedback" in prompt_lower and "answer" in prompt_lower:
            if "correct: true" in prompt_lower:
                content = "Great job! Your answer is correct. You've demonstrated a solid understanding of the concept."
            else:
                content = "Your answer needs some work. Try reviewing the problem carefully and checking your calculations."
        
        # For math hints
        elif "hint" in prompt_lower:
            if "solve" in prompt_lower and "equation" in prompt_lower:
                content = "1. First, isolate the variable by getting all terms with the variable on one side.\n2. Combine like terms to simplify.\n3. Divide both sides by the coefficient of the variable."
            else:
                content = "1. Start by identifying what the problem is asking for.\n2. Write down the relevant formulas.\n3. Substitute the given values into the formula."
        
        # For proximity scores
        elif "proximity score" in prompt_lower or "scale of 0-10" in prompt_lower:
            content = "7"
        
        # Default response
        else:
            content = "I'm a mock LLM response. In production, this would be generated by the OpenAI API."
        
        return {
            "content": content,
            "finish_reason": "stop",
            "model": "mock-model",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    def complete(self, system: str, user: str, temperature: float = 0.7, max_tokens: Optional[int] = None, output_format: str = None) -> str:
        """
        Complete a prompt and return just the content string.
        This is a convenience method that wraps generate_completion.
        
        Args:
            system: The system prompt
            user: The user prompt
            temperature: The temperature for generation
            max_tokens: The maximum number of tokens to generate
            output_format: Optional format hint (e.g., "json")
            
        Returns:
            The completion text
        """
        response = self.generate_completion(
            system_prompt=system,
            user_prompt=user,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.get("content", "")
