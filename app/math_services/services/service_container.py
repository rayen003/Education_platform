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
from app.math_services.services.health_check import HealthCheck, generate_llm_service_check, generate_meta_agent_check, HealthStatus
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
        # Primary services
        self.llm_service = llm_service or OpenAILLMService()
        self.meta_agent = meta_agent
        
        # Health check system
        self.health_check = HealthCheck()
        self._setup_health_checks()
        
        # Service registry
        self._services = {
            "llm_service": self.llm_service,
            "meta_agent": self.meta_agent,
            "health_check": self.health_check
        }
        
        # Fallback services (used when primary services fail)
        self._fallbacks = {}
        
        logger.info("Initialized ServiceContainer")
    
    def _setup_health_checks(self):
        """Set up health checks for the primary services."""
        # Register LLM service health check
        if self.llm_service:
            self.health_check.register_service(
                "llm_service", 
                generate_llm_service_check(self.llm_service),
                interval_seconds=300  # Check every 5 minutes
            )
        
        # Register meta agent health check if available
        if self.meta_agent:
            self.health_check.register_service(
                "meta_agent", 
                generate_meta_agent_check(self.meta_agent),
                interval_seconds=600  # Check every 10 minutes
            )
        
        logger.info("Health checks configured")
    
    def get_llm_service(self) -> BaseLLMService:
        """Get the LLM service with fallback handling."""
        # Check health if it's been a while
        if self.health_check.services.get("llm_service", {}).get("status") != HealthStatus.OK:
            try:
                # Run a health check to see if service is OK
                result = self.health_check.check_service("llm_service")
                
                # If still not OK and we have a fallback, use it
                if result["status"] != HealthStatus.OK and "llm_service" in self._fallbacks:
                    logger.warning("Using fallback LLM service due to health check failure")
                    return self._fallbacks["llm_service"]
            except:
                # If health check fails, just continue with current service
                pass
                
        return self.llm_service
    
    def get_meta_agent(self) -> Optional[Any]:
        """Get the meta agent for verification with fallback handling."""
        # Check if we have a meta agent
        if not self.meta_agent:
            return None
            
        # Check health if it's been a while
        if self.health_check.services.get("meta_agent", {}).get("status") != HealthStatus.OK:
            try:
                # Run a health check to see if service is OK
                result = self.health_check.check_service("meta_agent")
                
                # If still not OK and we have a fallback, use it
                if result["status"] != HealthStatus.OK and "meta_agent" in self._fallbacks:
                    logger.warning("Using fallback meta agent due to health check failure")
                    return self._fallbacks["meta_agent"]
            except:
                # If health check fails, just continue with current service
                pass
                
        return self.meta_agent
    
    def register_fallback(self, service_name: str, fallback_service: Any) -> None:
        """
        Register a fallback service to use when the primary service fails.
        
        Args:
            service_name: Name of the primary service
            fallback_service: The fallback service to use
        """
        self._fallbacks[service_name] = fallback_service
        logger.info(f"Registered fallback for service: {service_name}")
    
    def add_service(self, name: str, service: Any) -> None:
        """
        Add a custom service to the container.
        
        Args:
            name: Name of the service
            service: The service instance
        """
        setattr(self, name, service)
        self._services[name] = service
        logger.info(f"Added service: {name}")
    
    def get_service(self, name: str) -> Any:
        """
        Get a service by name with error handling.
        
        Args:
            name: Name of the service
            
        Returns:
            The service instance
        
        Raises:
            AttributeError: If the service doesn't exist
        """
        # Check if we have a direct attribute first
        if hasattr(self, name):
            return getattr(self, name)
        
        # Next check in services dictionary
        if name in self._services:
            return self._services[name]
            
        # Check if we have a fallback
        if name in self._fallbacks:
            logger.warning(f"Service {name} not found, using fallback")
            return self._fallbacks[name]
        
        # Service doesn't exist
        logger.error(f"Service not found: {name}")
        raise AttributeError(f"Service not found: {name}")
    
    def get_service_health(self, service_name: str = None) -> Dict[str, Any]:
        """
        Get health information for a service or all services.
        
        Args:
            service_name: Optional specific service to check
            
        Returns:
            Health information dictionary
        """
        if service_name:
            try:
                return self.health_check.check_service(service_name)
            except KeyError:
                return {"status": HealthStatus.UNKNOWN, "error": f"Service not registered: {service_name}"}
        else:
            return self.health_check.get_health_report() 