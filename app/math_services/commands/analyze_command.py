"""
Command for analyzing a student's mathematical calculation.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.math_services.commands.base_command import BaseCommand
from app.math_services.models.state import MathState, MathAnalysis
from app.math_services.services.service_container import ServiceContainer

logger = logging.getLogger(__name__)

class MathAnalyzeCalculationCommand(BaseCommand):
    """Command for analyzing a student's mathematical calculation."""
    
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
        
    def execute(self, state: MathState) -> MathState:
        """
        Analyze a student's answer to determine correctness and identify errors.
        
        Args:
            state: The current state with question and student answer
            
        Returns:
            Updated state with analysis
        """
        logger.info("Beginning calculation analysis")
        self.record_event(state, "analyze_start", {
            "question": state.question,
            "student_answer": state.student_answer,
            "correct_answer": state.correct_answer
        })
        
        try:
            # Create the prompt for analysis
            prompt = self._create_analysis_prompt(
                state.question, 
                state.student_answer, 
                state.correct_answer
            )
            
            # Call the LLM service
            logger.info("Calling LLM service for calculation analysis")
            response = self.llm_service.generate_completion(
                self._get_system_message(), 
                prompt
            )
            
            # Parse the response
            analysis_result = self._parse_analysis_response(response["content"])
            
            # Set confidence based on analysis factors
            confidence = self._assess_confidence(
                state.question,
                state.student_answer,
                state.correct_answer,
                analysis_result
            )
            
            # Create the analysis object
            analysis = MathAnalysis(
                is_correct=analysis_result.get("is_correct", False),
                error_type=analysis_result.get("error_type"),
                misconception=analysis_result.get("misconception"),
                calculation_steps=analysis_result.get("steps", []),
                verification_result=analysis_result.get("verification", {})
            )
            
            # Add confidence to context for use by other commands
            if not state.context:
                state.context = {}
            state.context["analysis_confidence"] = confidence
            
            # Log the result
            logger.info(f"Successfully analyzed calculation, correct: {analysis.is_correct}")
            
            if analysis.error_type:
                logger.info(f"Identified error type: {analysis.error_type}")
                
            if analysis.misconception:
                logger.info(f"Identified misconception: {analysis.misconception}")
            
            # Update the state
            state.analysis = analysis
            
            self.record_event(state, "analyze_complete", {
                "is_correct": analysis.is_correct,
                "error_type": analysis.error_type,
                "confidence": confidence
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error analyzing calculation: {str(e)}")
            self.log_error(e, state)
            
            # Create a default analysis
            default_analysis = MathAnalysis(
                is_correct=False,
                error_type="unknown",
                misconception=None
            )
            state.analysis = default_analysis
            
            # Add low confidence to context
            if not state.context:
                state.context = {}
            state.context["analysis_confidence"] = 0.3
            
            return state
    
    def _create_analysis_prompt(self, question: str, student_answer: str, correct_answer: Optional[str] = None) -> str:
        """
        Create a prompt for analyzing the calculation.
        
        Args:
            question: The math question
            student_answer: The student's answer
            correct_answer: The correct answer (if available)
            
        Returns:
            A formatted prompt
        """
        # Build the prompt with the available information
        prompt = f"""
Please analyze the following math problem and student's answer:

PROBLEM:
{question}

STUDENT'S ANSWER:
{student_answer}
"""

        # Add the correct answer if available
        if correct_answer:
            prompt += f"""
CORRECT ANSWER:
{correct_answer}
"""

        # Add instructions for the analysis
        prompt += """
Analyze the student's work and provide a detailed assessment in JSON format with the following structure:
{
  "is_correct": true/false,
  "error_type": null or ["calculation", "conceptual", "procedural", "algebraic", "arithmetic"],
  "misconception": null or "description of the student's misconception",
  "calculation_steps": [
    {"step": "description of step 1", "is_correct": true/false, "error": null or "description of error"},
    ...
  ],
  "verification": {
    "matches_correct_answer": true/false,
    "notes": "any additional verification notes"
  }
}

If the student's answer is correct, set is_correct to true, error_type to null, and misconception to null.
If the student's answer is incorrect, identify the type of error and any misconceptions.
"""
        return prompt.strip()
    
    def _get_system_message(self) -> str:
        """
        Get the system message for the LLM.
        
        Returns:
            The system message defining the role and rules
        """
        return """
You are an expert mathematics tutor specializing in analyzing student work.
Your task is to carefully analyze a student's mathematical calculation and identify
any errors or misconceptions. Your analysis should be detailed, accurate, and helpful.

Carefully examine each step of the student's work and identify:
1. Whether the final answer is correct
2. What type of error was made (if any)
3. What misconception might be present (if any)
4. A breakdown of each calculation step and whether it's correct

Provide your analysis in valid JSON format as specified in the prompt.
"""
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the analysis response from the LLM.
        
        Args:
            response: The raw LLM response
            
        Returns:
            The parsed analysis data
        """
        try:
            # Try to parse the response as JSON
            analysis_data = json.loads(response)
            return analysis_data
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from the text
            logger.warning("Failed to parse LLM response as JSON, attempting extraction")
            
            # Look for JSON-like content within the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                try:
                    json_str = response[json_start:json_end]
                    analysis_data = json.loads(json_str)
                    return analysis_data
                except json.JSONDecodeError:
                    logger.error("Failed to extract JSON from LLM response")
            
            # If extraction fails, return a default structure
            return {
                "is_correct": False,
                "error_type": "unknown",
                "misconception": "Failed to parse analysis from LLM response",
                "calculation_steps": [],
                "verification": {
                    "matches_correct_answer": False,
                    "notes": "Error in analysis response parsing"
                }
            }
    
    def _assess_confidence(self, question: str, student_answer: str, 
                         correct_answer: Optional[str], analysis_result: Dict[str, Any]) -> float:
        """
        Assess the confidence in our analysis.
        
        Args:
            question: The math problem
            student_answer: Student's submitted answer
            correct_answer: The correct answer (if available)
            analysis_result: The analysis results
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Start with a base confidence
        confidence = 0.7
        
        # Problem complexity affects confidence
        word_count = len(question.split())
        if word_count > 50:  # Complex problem
            confidence -= 0.1
        elif word_count < 15:  # Simple problem
            confidence += 0.1
            
        # Clear correct answer increases confidence
        if correct_answer:
            confidence += 0.1
            
        # Answer format affects confidence
        if len(student_answer) < 3:  # Very short answer
            confidence -= 0.05
        
        # Definite analysis results increase confidence
        if analysis_result.get("is_correct") is not None:
            confidence += 0.05
            
        if analysis_result.get("error_type"):
            confidence += 0.05
            
        # Add randomness for very high confidence to avoid appearing too certain
        if confidence > 0.9:
            confidence = min(0.95, confidence)
            
        # Set bounds
        confidence = max(0.3, min(confidence, 0.95))
        
        return confidence
