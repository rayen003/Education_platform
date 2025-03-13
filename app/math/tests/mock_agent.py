"""
Mock Agent for Testing.

This module contains a mock agent implementation for testing purposes.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MockAgent:
    """Mock agent for testing purposes."""
    
    def __init__(self):
        """
        Initialize the mock agent.
        """
        self.meta_agent = None  # Placeholder for the meta-agent, can be set in tests
        self.llm_service = None  # Placeholder for the LLM service, can be set in tests
        self.logger = logger
    
    def generate_completion(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Generate a completion using the LLM service.
        
        Args:
            system_prompt: The system prompt for the LLM
            user_prompt: The user prompt for the LLM
            
        Returns:
            The completion response
        """
        if self.llm_service:
            return self.llm_service.generate_completion(system_prompt, user_prompt)
        else:
            # Return a default response if no LLM service is set
            return {"content": "This is a mock response from the mock agent."}
    
    def record_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Record an event for tracking purposes.
        
        Args:
            event_type: The type of event
            event_data: The event data
        """
        logger.info(f"Mock agent recorded event: {event_type} - {event_data}")
