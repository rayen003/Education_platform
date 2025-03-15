"""
Command for solving a math problem symbolically using the LLM.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

from app.math_services.commands.base_command import BaseCommand
from app.math_services.models.state import MathState
from app.math_services.services.service_container import ServiceContainer

logger = logging.getLogger(__name__)

class MathSolveSymbolicallyCommand(BaseCommand):
    """Command for solving math problems symbolically using the LLM."""
    
    def __init__(self, service_container: ServiceContainer):
        """
        Initialize the command with services.
        
        Args:
            service_container: Container with required services
        """
        super().__init__(service_container)
        
    def _init_services(self, service_container: ServiceContainer):
        """
        Initialize services required by this command.
        
        Args:
            service_container: The service container
        """
        self.llm_service = service_container.get_llm_service()
        self.meta_agent = service_container.get_meta_agent()
        
    def execute(self, state: MathState) -> MathState:
        """
        Solve the math problem symbolically.
        
        Args:
            state: Current MathState
            
        Returns:
            Updated MathState with the solution
        """
        # Log the start of execution
        logger.info("Beginning symbolic math solution")
        self.record_event(state, "solve_start", {
            "question": state.question,
        })
        
        # Check if we have a question to solve
        if not state.question:
            logger.error("No question found in state")
            state.correct_answer = "Error: No question provided"
            return state
        
        try:
            # Create the prompt for the LLM
            prompt = self._create_solve_prompt(state.question)
            
            # Call the LLM service to solve the problem
            logger.info(f"Calling LLM service with prompt length: {len(prompt)}")
            response = self.llm_service.complete(
                system=self._get_system_message(),
                user=prompt
            )
            
            # Process the response
            solution = self._extract_solution(response)
            
            # Store the solution in the state
            state.correct_answer = solution
            
            # Log successful completion
            logger.info(f"Successfully solved problem, answer: {solution[:100]}...")
            self.record_event(state, "solve_complete", {
                "answer_length": len(solution) if solution else 0,
            })
                        
            # Verify the solution if meta agent is available
            if self.meta_agent:
                try:
                    verification = self.meta_agent.verify_solution(
                        question=state.question,
                        solution=solution
                    )
                    logger.info(f"Solution verification result: {verification}")
                    
                    # Store verification results
                    state.verification_results = verification
                    
                except Exception as verify_error:
                    logger.error(f"Error verifying solution: {str(verify_error)}")
                    state.verification_results = {
                        "verified": False,
                        "error": str(verify_error)
                    }
            
            return state
            
        except Exception as e:
            logger.error(f"Error in symbolic solution: {str(e)}")
            self.log_error(e, state)
            
            # Set a fallback message in case of error
            state.correct_answer = "Error: Unable to solve this problem"
            return state
    
    def _create_solve_prompt(self, question: str) -> str:
        """
        Create a prompt for solving the math problem.
        
        Args:
            question: The math question
            
        Returns:
            A formatted prompt
        """
        prompt = f"""
Please solve the following math problem step by step:

{question}

Provide a detailed solution showing all work. Include:
1. The key concepts and formulas needed
2. Each step of the calculation
3. The final answer (clearly marked)

Format your answer as a clear, step-by-step solution that a student could follow.
"""
        return prompt.strip()
    
    def _get_system_message(self) -> str:
        """
        Get the system message for the LLM.
        
        Returns:
            The system message defining the role and rules
        """
        return """
You are an expert mathematics tutor. Your task is to solve math problems step by step, 
showing all work clearly. Provide complete solutions that demonstrate proper 
mathematical reasoning and notation.

When solving problems:
- Identify the key concepts and formulas needed
- Break down the solution into clear steps
- Show all algebraic manipulations
- Provide the final answer clearly marked

Use proper mathematical notation and formatting.
"""
    
    def _extract_solution(self, response: str) -> str:
        """
        Extract the solution from the LLM response.
        
        Args:
            response: The raw LLM response
            
        Returns:
            The extracted solution
        """
        # For this simple implementation, we'll return the entire response
        # In a real implementation, you might want to parse JSON or extract specific parts
        return response.strip()
