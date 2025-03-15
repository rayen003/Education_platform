"""
Math Hint Command.

This module contains the command for generating progressive hints
for math problems.
"""

import logging
import json
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.math_services.commands.base_command import BaseCommand
from app.math_services.models.state import MathState, InteractionMode
from app.math_services.services.llm.base_service import BaseLLMService
from app.math_services.services.service_container import ServiceContainer

logger = logging.getLogger(__name__)

class MathGenerateHintCommand(BaseCommand):
    """Command for generating progressive hints for math problems."""
    
    def __init__(self, service_container: ServiceContainer):
        """
        Initialize the hint command.
        
        Args:
            service_container: Container with required services
        """
        super().__init__(service_container)
    
    def _init_services(self, service_container: ServiceContainer) -> None:
        """
        Initialize services needed by this command.
        
        Args:
            service_container: The service container
        """
        self.service_container = service_container
        self.llm_service = service_container.get_llm_service()
        self.meta_agent = service_container.get_service("meta_agent")
        logger.info("Initialized MathGenerateHintCommand")

    def execute(self, state: MathState) -> MathState:
        """
        Generate a hint for a math problem.
        
        Args:
            state: The current state with question and student information
            
        Returns:
            Updated state with hint
        """
        logger.info("Generating hint")
        self.record_event(state, "hint_generation_start", {
            "question": state.question,
            "has_previous_hints": len(state.hints) > 0
        })
        
        try:
            # Log problem details
            logger.info(f"Problem: {state.question}")
            logger.info(f"Student answer: {state.student_answer}")
            logger.info(f"Previous hints count: {len(state.hints)}")
            
            # Generate the hint
            hint = self._generate_hint(
                problem=state.question,
                student_answer=state.student_answer,
                correct_answer=state.correct_answer,
                hint_count=len(state.hints),
                hints=state.hints,
                analysis=state.analysis.to_dict() if state.analysis else {}
            )
            
            # Assess confidence in the hint
            confidence = self._assess_hint_confidence(state, hint)
            
            # Verify the hint if meta-agent is available
            verification_result = None
            if self.meta_agent:
                try:
                    # Verify the hint
                    verification_result = self.meta_agent.verify_hint(
                        state.question,
                        hint
                    )
                    
                    # Log verification result
                    logger.info(f"Hint verification result: {verification_result}")
                    
                    # Update confidence based on verification
                    if verification_result:
                        verification_confidence = verification_result.get("confidence", 0.5)
                        # Weight the verification more heavily
                        confidence = (confidence + (verification_confidence * 2)) / 3
                        
                        # Adjust bounds based on verification
                        if verification_result.get("verified", False):
                            confidence = max(confidence, 0.8)
                        else:
                            confidence = min(confidence, 0.6)
                except Exception as e:
                    logger.warning(f"Hint verification failed: {str(e)}")
            
            # Add the hint to the state
            state.hints.append(hint)
            state.hint_count = len(state.hints)
            
            # Store confidence in context
            if not state.context:
                state.context = {}
            state.context["hint_confidence"] = confidence
            
            self.record_event(state, "hint_generation_complete", {
                "hint_length": len(hint),
                "verification": verification_result.get("verified", True) if verification_result else True,
                "confidence": confidence
            })
            
            return state
        
        except Exception as e:
            logger.error(f"Error generating hint: {str(e)}")
            self.log_error(e, state)
            
            # Create a fallback hint
            fallback_hint = "Try breaking down the problem into smaller steps and identify the key variables."
            state.hints.append(fallback_hint)
            state.hint_count = len(state.hints)
            
            # Set low confidence for fallback hint
            if not state.context:
                state.context = {}
            state.context["hint_confidence"] = 0.2
            
            return state
    
    def _assess_hint_confidence(self, state: MathState, hint: str) -> float:
        """
        Assess confidence in a generated hint.
        
        Args:
            state: Current math state
            hint: The generated hint
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Start with a base confidence
        confidence = 0.7
        
        # Adjust based on problem complexity
        word_count = len(state.question.split())
        if word_count > 50:  # Complex problem
            confidence -= 0.1
        elif word_count < 15:  # Simple problem
            confidence += 0.1
        
        # Adjust based on hint count - progressively lower confidence with more hints
        # as later hints are more specific and may be less reliable
        hint_count = len(state.hints)
        if hint_count > 2:
            confidence -= 0.1 * (hint_count - 2)
        
        # Check if we have analysis results to inform our hint
        if state.analysis:
            if state.analysis.is_correct:
                # Higher confidence for correct answer hints
                confidence += 0.1
            elif state.analysis.error_type:
                # We understand the error, so higher confidence
                confidence += 0.05
        
        # Assess hint quality characteristics
        if len(hint) < 20:  # Very short hint
            confidence -= 0.1
        elif len(hint) > 200:  # Very long hint
            confidence -= 0.05
            
        # Check for question marks in the hint, which suggest prompting rather than directing
        if "?" in hint:
            confidence += 0.05
            
        # Check for specific math terms in the hint
        math_terms = ["equation", "formula", "solve", "calculate", "value", "variable", "function", "graph"]
        term_count = sum(1 for term in math_terms if term in hint.lower())
        confidence += min(0.1, term_count * 0.02)  # Up to 0.1 bonus for relevant terms
        
        # Set bounds
        confidence = max(0.3, min(confidence, 0.95))
        
        return confidence
