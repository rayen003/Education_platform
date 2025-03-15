"""
Mock LLM Service.

This module contains a mock implementation of the LLM service interface
for testing and fallback purposes.
"""

import logging
from typing import Dict, Any

from app.math_services.services.llm.base_service import BaseLLMService

logger = logging.getLogger(__name__)

class MockLLMService(BaseLLMService):
    """Mock implementation of the LLM service for testing and fallback."""
    
    def generate_completion(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Generate a mock completion.
        
        Args:
            system_prompt: The system prompt to use
            user_prompt: The user prompt to use
            
        Returns:
            Mock response dictionary
        """
        logger.info("Using mock LLM service")
        
        # Determine the type of response based on the prompt
        if "hint" in system_prompt.lower():
            return self._handle_generate_hint(user_prompt)
        elif "feedback" in system_prompt.lower():
            return self._handle_generate_feedback(user_prompt)
        elif "solve" in system_prompt.lower():
            return self._handle_generate_solution(user_prompt)
        else:
            return {
                "content": "This is a mock response. Please use the real LLM service for better results.",
                "model": "mock-model"
            }
    
    def _handle_generate_hint(self, user_prompt: str) -> Dict[str, Any]:
        """Generate a mock hint."""
        # Extract hint count if available
        hint_count = 1
        if "hint count:" in user_prompt.lower():
            try:
                hint_count_line = [line for line in user_prompt.split('\n') if "hint count:" in line.lower()][0]
                hint_count = int(hint_count_line.split(':')[1].strip())
            except:
                pass
        
        # Generate a hint based on the count
        if hint_count == 1:
            hint = "Try breaking down the problem into smaller steps and identify the key variables and equations."
        elif hint_count == 2:
            hint = "Consider applying the relevant formula and check your calculations carefully."
        else:
            hint = "Look at the specific values in the problem and make sure you're using the correct units and formulas."
        
        return {
            "content": hint,
            "model": "mock-model"
        }
    
    def _handle_generate_feedback(self, user_prompt: str) -> Dict[str, Any]:
        """Generate mock feedback."""
        # Determine if the answer is correct
        is_correct = "correct" in user_prompt.lower() and not "incorrect" in user_prompt.lower()
        
        if is_correct:
            feedback = {
                "summary": "Good job! Your answer is correct.",
                "strengths": ["You applied the correct formula", "Your calculations are accurate"],
                "areas_for_improvement": [],
                "next_steps": ["Try more challenging problems."]
            }
        else:
            feedback = {
                "summary": "Your answer needs some work.",
                "strengths": ["You attempted to solve the problem"],
                "areas_for_improvement": ["Check your calculations", "Review the formula you used"],
                "next_steps": ["Try again with the provided hint"]
            }
            
            # Add solution explanation for final attempts
            if "final attempt" in user_prompt.lower() or "attempt: 3" in user_prompt.lower():
                feedback["solution_explanation"] = "The correct approach is to use the formula and apply it step by step."
        
        return {
            "content": str(feedback),
            "model": "mock-model"
        }
    
    def _handle_generate_solution(self, user_prompt: str) -> Dict[str, Any]:
        """Generate a mock solution."""
        return {
            "content": "x = 42",
            "model": "mock-model"
        }
