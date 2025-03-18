"""
Hint Command Module.

This module provides the command for generating contextual hints for math problems
using Chain of Draft methodology for concise, targeted hints.
"""

import logging
import re
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.math_services.commands.base_command import BaseCommand
from app.math_services.models.state import MathState
from app.math_services.services.service_container import ServiceContainer

logger = logging.getLogger(__name__)

class MathGenerateHintCommand(BaseCommand):
    """Command for generating hints for math problems using Chain of Draft."""
    
    def __init__(self, service_container: ServiceContainer):
        """
        Initialize the command with services.
        
        Args:
            service_container: Container with required services
        """
        super().__init__(service_container)
        
        # Cache for hints to avoid regeneration
        self._hint_cache = {}
    
    def _init_services(self, service_container: ServiceContainer) -> None:
        """
        Initialize services required by this command.
        
        Args:
            service_container: The service container
        """
        self.llm_service = service_container.get_llm_service()
        self.meta_agent = service_container.get_meta_agent()
    
    def execute(self, state: MathState) -> MathState:
        """
        Generate a single contextual hint based on the current state.
        
        Args:
            state: Current math state
            
        Returns:
            Updated state with a new hint
        """
        logger.info(f"Generating hint for problem: '{state.question[:50]}...'")
        
        # Check if problem is solvable
        if not state.question:
            logger.warning("Cannot generate hint: No problem statement")
            return self.record_event(state, "error", {"message": "No problem statement"})
        
        # Get the hint count and existing hints
        hint_count = state.hint_count
        hints = state.hints
        
        # Generate new hint using Chain of Draft
        new_hint = self._generate_contextual_hint(
            state.question,
            state.student_answer,
            state.correct_answer,
            hint_count,
            hints,
            state.analysis.to_dict() if hasattr(state.analysis, "to_dict") else {}
        )
        
        # Add the hint to the state
        state.hints.append(new_hint)
        state.hint_count = len(state.hints)
        
        # Limit to max 5 hints to manage memory
        if len(state.hints) > 5:
            state.hints = state.hints[-5:]
            state.hint_count = len(state.hints)
        
        # Record the event
        self.record_event(state, "hint_generated", {
            "hint_number": state.hint_count,
            "hint": new_hint[:50] + "..." if len(new_hint) > 50 else new_hint
        })
        
        return state
    
    def _generate_contextual_hint(self, question: str, student_answer: str, correct_answer: Optional[str], 
                           hint_count: int, hints: List[str], analysis: Dict[str, Any]) -> str:
        """
        Generate a contextual hint using Chain of Draft.
        
        Args:
            question: The math problem
            student_answer: Student's current answer
            correct_answer: Correct answer if available
            hint_count: Number of hints already given
            hints: Previous hints
            analysis: Error analysis data
            
        Returns:
            A new contextual hint
        """
        # Define the hint progression strategy based on hint count
        hint_strategies = [
            "general_approach",      # First hint: General approach
            "concept_reminder",      # Second hint: Remind of relevant concept
            "error_specific",        # Third hint: Address specific error
            "step_guidance",         # Fourth hint: Guide through next step
            "detailed_direction"     # Fifth+ hint: More detailed direction
        ]
        
        # Get the appropriate strategy
        strategy_index = min(hint_count, len(hint_strategies) - 1)
        current_strategy = hint_strategies[strategy_index]
        
        # Format previous hints for context
        previous_hints_context = ""
        if hints:
            previous_hints_context = "Previous hints:\n"
            for i, hint in enumerate(hints):
                previous_hints_context += f"{i+1}. {hint}\n"
        
        # Extract error information if available
        error_context = ""
        if "error_type" in analysis and analysis["error_type"]:
            error_context = f"Error type: {analysis['error_type']}\n"
            if "misconception" in analysis and analysis["misconception"]:
                error_context += f"Possible misconception: {analysis['misconception']}\n"
        
        try:
            # Chain of Draft approach
            system_prompt = f"""
            You are an expert math tutor using Chain of Draft to create a hint.
            
            The student is on hint #{hint_count + 1}. Use the '{current_strategy}' strategy.
            
            Chain of Draft process:
            1. Identify the key concept needed (5 words max)
            2. Determine what specific help is needed (5 words max)
            3. Decide on the most valuable insight (5 words max)
            4. Compose a concise, helpful hint (1-2 sentences)
            
            Show ONLY your final hint from step 4 in your response.
            Build on previous hints and focus on progressing the student's understanding without solving the problem for them.
            """
            
            user_prompt = f"""
            Problem: {question}
            
            Student's current answer: {student_answer}
            
            {error_context}
            {previous_hints_context}
            
            Based on the information above, provide a single, helpful hint using the '{current_strategy}' strategy.
            The hint should naturally build on any previous hints and help the student make progress.
            """
            
            # Generate the hint
            response = self.llm_service.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
                max_tokens=100  # Limit tokens for concise hints
            )
            
            hint = response.get("content", "").strip()
            
            # Clean up the hint (remove prefixes like "Hint: " if present)
            hint = re.sub(r"^(Hint|Hint \d+|Here's a hint):\s*", "", hint, flags=re.IGNORECASE)
            
            return hint.strip()
        except Exception as e:
            logger.error(f"Error generating hint: {e}")
            return "I couldn't generate a proper hint. Try approaching the problem step by step."
    
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
        
        # Adjust based on hint quality factors
        if len(hint) < 10:
            confidence -= 0.2  # Too short
        elif len(hint) > 200:
            confidence -= 0.1  # Too verbose
            
        # Check if hint mentions key concepts from the problem
        problem_words = set(re.findall(r'\b\w+\b', state.question.lower()))
        hint_words = set(re.findall(r'\b\w+\b', hint.lower()))
        concept_overlap = len(problem_words.intersection(hint_words)) / len(problem_words) if problem_words else 0
        
        if concept_overlap > 0.3:
            confidence += 0.1
        
        # Cap confidence at reasonable bounds
        return max(0.1, min(0.95, confidence))

