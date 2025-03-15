"""
Service Container for Dependency Injection.

This module provides a container for all the services used in the math
services, removing the need for circular dependencies between commands
and agents.
"""

from typing import Optional, Dict, Any
import logging
from app.math_services.services.llm.base_service import BaseLLMService
from app.math_services.services.llm.openai_service import OpenAILLMService
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MathFeedback:
    assessment: str = ""
    is_correct: bool = False
    proximity_score: float = 0.0
    detail: str = ""
    confidence: float = 0.0  # Add confidence level (0-1)

class ServiceContainer:
    """
    Container for all services used in the math services.
    This eliminates the need for commands to reference the agent directly.
    """
    
    def __init__(self, llm_service: Optional[BaseLLMService] = None, 
                meta_agent: Optional[Any] = None):
        """
        Initialize the service container.
        
        Args:
            llm_service: LLM service to use (will create one if None)
            meta_agent: Meta agent for verification (optional)
        """
        self.llm_service = llm_service or OpenAILLMService()
        self.meta_agent = meta_agent
        # We'll add more services as needed
        
        logger.info("Initialized ServiceContainer")
    
    def get_llm_service(self) -> BaseLLMService:
        """Get the LLM service."""
        return self.llm_service
    
    def get_meta_agent(self) -> Optional[Any]:
        """Get the meta agent for verification, if available."""
        return self.meta_agent
    
    def add_service(self, name: str, service: Any) -> None:
        """
        Add a custom service to the container.
        
        Args:
            name: Name of the service
            service: The service instance
        """
        setattr(self, name, service)
        logger.info(f"Added service: {name}")
    
    def get_service(self, name: str) -> Any:
        """
        Get a service by name.
        
        Args:
            name: Name of the service
            
        Returns:
            The service instance
        
        Raises:
            AttributeError: If the service doesn't exist
        """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise AttributeError(f"Service not found: {name}") 