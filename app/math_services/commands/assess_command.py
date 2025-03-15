"""
Math Assess Command.

This module contains the command for assessing the proximity of student answers.
"""

import logging
import json
import traceback
from typing import Dict, Any, List, Optional
import sympy
from sympy import symbols, sympify, solve, Eq, simplify
from sympy.parsing.sympy_parser import parse_expr

from app.math_services.services.llm.base_service import BaseLLMService

logger = logging.getLogger(__name__)

# Utility functions (moved from utilities.math_utils)
def normalize_answer(answer: str) -> str:
    """
    Normalize a mathematical answer for comparison.
    
    Args:
        answer: The answer to normalize
        
    Returns:
        Normalized answer string
    """
    # Convert to lowercase
    normalized = answer.lower()
    
    # Remove whitespace
    normalized = normalized.replace(" ", "")
    
    # Remove dollar signs (LaTeX)
    normalized = normalized.replace('$', '')
    
    # Replace common variations
    normalized = normalized.replace('x=', '')
    normalized = normalized.replace('y=', '')
    normalized = normalized.replace('z=', '')
    
    return normalized

def calculate_string_similarity(str1: str, str2: str) -> float:
    """
    Calculate similarity between two strings using SequenceMatcher.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Similarity score between 0 and 1
    """
    from difflib import SequenceMatcher
    return SequenceMatcher(None, str1, str2).ratio()

def calculate_proximity(student_answer: str, correct_answer: str, tolerance: float = 0.01) -> Dict[str, Any]:
    """
    Calculate the proximity of a student's answer to the correct answer.
    
    Args:
        student_answer: The student's answer
        correct_answer: The correct answer
        tolerance: Tolerance for numerical comparison
        
    Returns:
        Dictionary with proximity assessment results
    """
    # Normalize both answers
    norm_student = normalize_answer(student_answer)
    norm_correct = normalize_answer(correct_answer)
    
    # Initialize result
    result = {
        "normalized_student_answer": norm_student,
        "normalized_correct_answer": norm_correct,
        "similarity": 0.0,
        "is_correct": False
    }
    
    # Check for exact string match
    if norm_student == norm_correct:
        result["similarity"] = 1.0
        result["is_correct"] = True
        return result
    
    # Try symbolic comparison
    try:
        student_expr = parse_expr(norm_student)
        correct_expr = parse_expr(norm_correct)
        
        # Check if they are symbolically equivalent
        if sympy.simplify(student_expr - correct_expr) == 0:
            result["similarity"] = 1.0
            result["is_correct"] = True
            return result
    except Exception:
        pass
    
    # Try numerical comparison for float values
    try:
        student_val = float(norm_student)
        correct_val = float(norm_correct)
        
        # Calculate relative error
        if correct_val != 0:
            rel_error = abs((student_val - correct_val) / correct_val)
            result["similarity"] = max(0, 1 - rel_error)
        else:
            # If correct value is 0, use absolute error
            abs_error = abs(student_val - correct_val)
            result["similarity"] = 1 / (1 + abs_error)  # Approaches 1 as error approaches 0
        
        # Check if they are close enough
        if abs(student_val - correct_val) <= tolerance:
            result["is_correct"] = True
            return result
    except ValueError:
        pass
    
    # Calculate string similarity as a last resort
    similarity = calculate_string_similarity(norm_student, norm_correct)
    result["similarity"] = similarity
    
    # Consider it correct if similarity is very high
    if similarity > 0.9:  # 90% similarity threshold
        result["is_correct"] = True
    
    return result

class MathAssessProximityCommand:
    """Command for assessing the proximity of student answers."""
    
    def __init__(self, agent):
        """
        Initialize the assess command.
        
        Args:
            agent: The agent instance that will use this command
        """
        self.agent = agent
        self.llm_service = agent.llm_service
        self.meta_agent = agent.meta_agent
        logger.info("Initialized MathAssessProximityCommand")
    
    def _execute_core(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess how close the student's answer is to the correct one.
        
        Args:
            state: The current state dictionary
            
        Returns:
            Updated state with proximity assessment
        """
        # Get the required information from the state
        student_answer = state.get("student_answer", "")
        correct_answer = state.get("correct_answer", "")
        
        if not student_answer or not correct_answer:
            logger.error("Missing student answer or correct answer")
            return {
                "status": "error",
                "error": "Missing student answer or correct answer"
            }
        
        try:
            # Calculate proximity between student answer and correct answer
            proximity_result = calculate_proximity(student_answer, correct_answer)
            logger.info(f"Answer similarity: {proximity_result['similarity']:.2f}")
            
            # Determine if the answer is correct
            is_correct = proximity_result['is_correct']
            
            # Always set needs_hint to True if the answer is incorrect, regardless of errors
            if not is_correct:
                state["needs_hint"] = True
            
            # Store assessment in state
            if "analysis" not in state:
                state["analysis"] = {}
            state["analysis"]["proximity_assessment"] = {
                "similarity": proximity_result['similarity'],
                "is_correct": is_correct,
                "needs_hint": state.get("needs_hint", False),
                "normalized_student_answer": proximity_result['normalized_student_answer'],
                "normalized_correct_answer": proximity_result['normalized_correct_answer']
            }
            
            # Update state with assessment results
            state["is_correct"] = is_correct
            
            # Record success event
            return {
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in proximity assessment: {e}")
            logger.debug(traceback.format_exc())
            
            state["is_correct"] = False
            
            # Record error event
            return {
                "status": "error",
                "error": f"Failed to assess proximity: {str(e)}"
            }
