"""
Base Command Interface.

This module provides the base command interface that all math service
commands should implement, using dependency injection.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

from app.math_services.models.state import MathState
from app.math_services.services.service_container import ServiceContainer

logger = logging.getLogger(__name__)

class BaseCommand(ABC):
    """Base command interface for all math service commands."""
    
    def __init__(self, services: ServiceContainer):
        """
        Initialize the command with required services.
        
        Args:
            services: Service container with all needed services
        """
        self.services = services
        self._init_services(services)
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def _init_services(self, service_container: ServiceContainer) -> None:
        """
        Initialize services from the container.
        Override this in subclasses if needed.
        
        Args:
            service_container: The service container
        """
        self.llm_service = service_container.get_llm_service()
        self.meta_agent = service_container.get_meta_agent()
    
    @abstractmethod
    def execute(self, state: MathState) -> MathState:
        """
        Execute the command on the given state.
        
        Args:
            state: The current state
            
        Returns:
            The updated state
        """
        pass
    
    def record_event(self, state: MathState, event_type: str, event_data: Dict[str, Any]) -> MathState:
        """
        Record an event for tracking and debugging.
        
        Args:
            state: The current state
            event_type: Type of event
            event_data: Data about the event
            
        Returns:
            The updated state with the event recorded
        """
        logger.info(f"{self.__class__.__name__} event: {event_type}, {event_data}")
        
        if not hasattr(state, 'events') or state.events is None:
            state.events = []
        
        event = {
            "type": event_type,
            "command": self.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
            "data": event_data
        }
        
        state.events.append(event)
        return state
    
    def log_error(self, error: Exception, state: Optional[MathState] = None) -> None:
        """
        Log an error that occurred during command execution.
        
        Args:
            error: The exception that occurred
            state: Optional state for context
        """
        logger.error(f"Error in {self.__class__.__name__}: {str(error)}", exc_info=True)
        
        if state:
            self.record_event(
                state,
                "error",
                {"error_type": type(error).__name__, "error_message": str(error)}
            ) 