"""
Health Check Module.

This module provides health check functionality for monitoring the status
of services and components in the math assessment system.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HealthStatus(str, Enum):
    """Health status values for services."""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"

class HealthCheck:
    """
    Health check system for monitoring service health.
    
    Tracks and reports on the health status of various services,
    providing centralized monitoring and reporting.
    """
    
    def __init__(self):
        """Initialize the health check system."""
        # Service registry: service_name -> service info
        self.services = {}
        
        # Health check functions registry
        self.check_functions = {}
        
        # Last check timestamps
        self.last_checks = {}
        
        logger.info("Initialized health check system")
    
    def register_service(self, 
                       service_name: str, 
                       check_function: Callable[[], Dict[str, Any]],
                       interval_seconds: int = 300) -> None:
        """
        Register a service for health checks.
        
        Args:
            service_name: Name of the service
            check_function: Function to call to check service health
            interval_seconds: How often to check the service (default: 5 min)
        """
        self.services[service_name] = {
            "name": service_name,
            "status": HealthStatus.UNKNOWN,
            "last_check": None,
            "check_interval": interval_seconds,
            "details": {"error": "Not checked yet"}
        }
        
        self.check_functions[service_name] = check_function
        logger.info(f"Registered service for health checks: {service_name}")
    
    def unregister_service(self, service_name: str) -> None:
        """
        Unregister a service from health checks.
        
        Args:
            service_name: Name of the service to unregister
        """
        if service_name in self.services:
            del self.services[service_name]
        
        if service_name in self.check_functions:
            del self.check_functions[service_name]
        
        if service_name in self.last_checks:
            del self.last_checks[service_name]
            
        logger.info(f"Unregistered service from health checks: {service_name}")
    
    def check_service(self, service_name: str, force: bool = False) -> Dict[str, Any]:
        """
        Check the health of a specific service.
        
        Args:
            service_name: Name of the service to check
            force: Whether to force a check even if interval hasn't passed
            
        Returns:
            Health check result
            
        Raises:
            KeyError: If service is not registered
        """
        if service_name not in self.services:
            logger.error(f"Service not registered: {service_name}")
            raise KeyError(f"Service not registered: {service_name}")
        
        # Get current service info
        service_info = self.services[service_name]
        check_interval = service_info.get("check_interval", 300)
        last_check = service_info.get("last_check")
        
        # Check if interval has passed or force check
        if not force and last_check:
            time_since_check = (datetime.now() - last_check).total_seconds()
            if time_since_check < check_interval:
                logger.debug(f"Skipping health check for {service_name}, checked {time_since_check:.0f}s ago")
                return service_info
        
        # Get check function
        check_function = self.check_functions.get(service_name)
        if not check_function:
            logger.error(f"No check function registered for service: {service_name}")
            service_info["status"] = HealthStatus.UNKNOWN
            service_info["details"] = {"error": "No check function registered"}
            return service_info
        
        # Perform health check
        try:
            # Track start time for performance monitoring
            start_time = time.time()
            
            # Call the check function
            check_result = check_function()
            
            # Update service info
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Extract status from result, default to UNKNOWN
            status = check_result.get("status", HealthStatus.UNKNOWN)
            
            # Update service info
            service_info["status"] = status
            service_info["last_check"] = datetime.now()
            service_info["details"] = check_result.get("details", {})
            service_info["duration_ms"] = duration_ms
            
            logger.info(f"Health check for {service_name}: {status} ({duration_ms}ms)")
            
        except Exception as e:
            # Handle errors in health check
            logger.error(f"Error checking health of {service_name}: {str(e)}")
            service_info["status"] = HealthStatus.ERROR
            service_info["last_check"] = datetime.now()
            service_info["details"] = {"error": str(e)}
        
        # Update services registry
        self.services[service_name] = service_info
        
        return service_info
    
    def check_all_services(self, force: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Check the health of all registered services.
        
        Args:
            force: Whether to force a check even if interval hasn't passed
            
        Returns:
            Dictionary mapping service names to health check results
        """
        results = {}
        
        for service_name in self.services.keys():
            try:
                results[service_name] = self.check_service(service_name, force)
            except Exception as e:
                logger.error(f"Error checking {service_name}: {e}")
                results[service_name] = {
                    "name": service_name,
                    "status": HealthStatus.ERROR,
                    "details": {"error": str(e)}
                }
        
        return results
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive health report for all services.
        
        Returns:
            Health report dictionary
        """
        # Check all services that need checking
        for service_name, service_info in self.services.items():
            # Check if service needs checking
            if service_info.get("last_check") is None:
                # Never checked before
                self.check_service(service_name)
            else:
                # Check interval
                last_check = service_info.get("last_check")
                check_interval = service_info.get("check_interval", 300)
                time_since_check = (datetime.now() - last_check).total_seconds()
                
                if time_since_check > check_interval:
                    self.check_service(service_name)
        
        # Determine overall status
        overall_status = HealthStatus.OK
        
        for service_info in self.services.values():
            status = service_info.get("status", HealthStatus.UNKNOWN)
            
            if status == HealthStatus.ERROR:
                overall_status = HealthStatus.ERROR
                break
            elif status == HealthStatus.WARNING and overall_status != HealthStatus.ERROR:
                overall_status = HealthStatus.WARNING
            elif status == HealthStatus.UNKNOWN and overall_status == HealthStatus.OK:
                overall_status = HealthStatus.UNKNOWN
        
        # Build the report
        report = {
            "timestamp": datetime.now().isoformat(),
            "status": overall_status,
            "services": self.services
        }
        
        return report

def generate_llm_service_check(llm_service: Any) -> Callable[[], Dict[str, Any]]:
    """
    Generate a health check function for an LLM service.
    
    Args:
        llm_service: The LLM service to check
        
    Returns:
        Health check function
    """
    def check_llm_service() -> Dict[str, Any]:
        """Check LLM service health."""
        try:
            # Simple test to check if service is responsive
            test_prompt = "Write 'OK' if you can read this message."
            system_prompt = "You are a helpful assistant."
            
            response = llm_service.generate_completion(
                system_prompt=system_prompt,
                user_prompt=test_prompt,
                max_tokens=10,
                temperature=0
            )
            
            # Check if we got a valid response
            if not response or "content" not in response:
                return {
                    "status": HealthStatus.ERROR,
                    "details": {
                        "error": "Invalid response from LLM service",
                        "response": str(response)
                    }
                }
            
            content = response.get("content", "").strip().lower()
            
            # Check for expected content
            if "ok" in content:
                return {
                    "status": HealthStatus.OK,
                    "details": {
                        "model": getattr(llm_service, "model_name", "unknown"),
                        "response_length": len(content)
                    }
                }
            else:
                return {
                    "status": HealthStatus.WARNING,
                    "details": {
                        "error": "Unexpected response content",
                        "content": content
                    }
                }
                
        except Exception as e:
            return {
                "status": HealthStatus.ERROR,
                "details": {"error": str(e)}
            }
    
    return check_llm_service

def generate_meta_agent_check(meta_agent: Any) -> Callable[[], Dict[str, Any]]:
    """
    Generate a health check function for a meta agent.
    
    Args:
        meta_agent: The meta agent to check
        
    Returns:
        Health check function
    """
    def check_meta_agent() -> Dict[str, Any]:
        """Check meta agent health."""
        try:
            # Check if meta agent has required attributes
            if not hasattr(meta_agent, "verify_solution"):
                return {
                    "status": HealthStatus.ERROR,
                    "details": {"error": "Meta agent missing verify_solution method"}
                }
            
            # Simple test to check if service is responsive
            test_question = "What is 2+2?"
            test_solution = "The sum of 2 and 2 is 4."
            
            verification = meta_agent.verify_solution(test_question, test_solution)
            
            # Check if we got a valid response
            if not verification or "is_valid" not in verification:
                return {
                    "status": HealthStatus.WARNING,
                    "details": {
                        "error": "Invalid response from meta agent",
                        "response": str(verification)
                    }
                }
            
            is_valid = verification.get("is_valid", False)
            
            if is_valid:
                return {
                    "status": HealthStatus.OK,
                    "details": {
                        "confidence": verification.get("confidence", 0),
                        "verification_type": verification.get("method", "unknown")
                    }
                }
            else:
                return {
                    "status": HealthStatus.WARNING,
                    "details": {
                        "error": "Meta agent rejected valid solution",
                        "feedback": verification.get("feedback", "No feedback provided")
                    }
                }
                
        except Exception as e:
            return {
                "status": HealthStatus.ERROR,
                "details": {"error": str(e)}
            }
    
    return check_meta_agent 