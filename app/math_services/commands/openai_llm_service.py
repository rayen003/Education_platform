"""
OpenAI LLM Service for math commands.

This module contains an LLM service implementation using OpenAI for generating
hints, feedback, and other responses for math problems.
"""

import os
import json
import logging
from typing import Any, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class OpenAILLMService:
    """OpenAI LLM service for generating responses."""
    
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the OpenAI LLM service.
        
        Args:
            model: The OpenAI model to use
        """
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.timeout = 15  # Increased timeout from 8 to 15 seconds
        logger.info(f"Initialized OpenAILLMService with model {model}")
    
    def generate_completion(self, system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a completion using the OpenAI API with timeout handling.
        
        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt
            **kwargs: Additional arguments to pass to the OpenAI API
            
        Returns:
            Dict containing the response content and other metadata
        """
        # Log the prompt for debugging
        logger.info(f"Generating completion with prompt: {user_prompt[:50]}...")
        
        # Set timeout from instance variable or kwargs
        timeout_seconds = kwargs.pop('timeout', self.timeout)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Set default parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1000,
            **kwargs
        }
        
        try:
            # Use a timeout for the API call
            import threading
            import queue
            
            response_queue = queue.Queue()
            
            def make_api_call():
                try:
                    response = self.client.chat.completions.create(**params)
                    response_queue.put({"success": True, "response": response})
                except Exception as e:
                    response_queue.put({"success": False, "error": str(e)})
            
            # Start API call in a separate thread
            api_thread = threading.Thread(target=make_api_call)
            api_thread.daemon = True
            api_thread.start()
            
            # Wait for the response with timeout
            try:
                result = response_queue.get(timeout=timeout_seconds)
                if not result["success"]:
                    logger.error(f"API call failed: {result['error']}")
                    return {"content": f"Error: {result['error']}", "error": True}
                
                response = result["response"]
                content = response.choices[0].message.content
                return {"content": content, "response": response}
            except queue.Empty:
                logger.error(f"API call timed out after {timeout_seconds} seconds")
                return {"content": "Error: API call timed out", "error": True, "timed_out": True}
            
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            return {"content": f"Error: {str(e)}", "error": True}
