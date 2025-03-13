"""
Math Analyze Command.

This module contains the command for analyzing student calculations.
"""

import logging
import json
import traceback
import re
from typing import Dict, Any, List, Optional
import sympy
from sympy import symbols, sympify, solve, Eq, simplify
from sympy.parsing.sympy_parser import parse_expr

from app.math.services.llm.base_service import BaseLLMService

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
    normalized = re.sub(r'\s+', '', normalized)
    
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

def is_answer_correct(student_answer: str, correct_answer: str, tolerance: float = 0.01) -> bool:
    """
    Check if a student's answer is correct.
    
    Args:
        student_answer: The student's answer
        correct_answer: The correct answer
        tolerance: Tolerance for numerical comparison
        
    Returns:
        True if the answer is correct, False otherwise
    """
    # Normalize both answers
    norm_student = normalize_answer(student_answer)
    norm_correct = normalize_answer(correct_answer)
    
    # Check for exact string match
    if norm_student == norm_correct:
        return True
    
    # Try symbolic comparison
    try:
        student_expr = parse_expr(norm_student)
        correct_expr = parse_expr(norm_correct)
        
        # Check if they are symbolically equivalent
        if sympy.simplify(student_expr - correct_expr) == 0:
            return True
    except Exception:
        pass
    
    # Try numerical comparison for float values
    try:
        student_val = float(norm_student)
        correct_val = float(norm_correct)
        
        # Check if they are close enough
        if abs(student_val - correct_val) <= tolerance:
            return True
    except ValueError:
        pass
    
    # Check string similarity as a last resort
    similarity = calculate_string_similarity(norm_student, norm_correct)
    if similarity > 0.9:  # 90% similarity threshold
        return True
    
    return False

class MathAnalyzeCalculationCommand:
    """Command for analyzing a student's calculation."""
    
    def __init__(self, agent):
        """
        Initialize the analyze command.
        
        Args:
            agent: The agent instance that will use this command
        """
        self.agent = agent
        self.llm_service = agent.llm_service
        self.meta_agent = agent.meta_agent
        logger.info("Initialized MathAnalyzeCalculationCommand")
    
    def _execute_core(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a student's calculation.
        
        Args:
            state: The current state dictionary
            
        Returns:
            Updated state with analysis
        """
        # Get the required information from the state
        question = state.get("question", "")
        student_answer = state.get("student_answer", "")
        parsed_question = state.get("analysis", {}).get("parsed_question", {})
        solution = state.get("solution", {})
        
        if not question or not student_answer:
            logger.error("Missing question or student answer")
            return self._record_event(state, {
                "status": "error",
                "error": "Missing question or student answer"
            })
        
        if not parsed_question:
            logger.error("No parsed question information available")
            return self._record_event(state, {
                "status": "error",
                "error": "No parsed question information available"
            })
        
        try:
            # Determine the problem type
            problem_type = parsed_question.get("type", "symbolic")
            logger.info(f"Analyzing calculation for problem type: {problem_type}")
            
            # Analyze based on problem type
            if problem_type == "symbolic":
                return self._analyze_symbolic_calculation(state, parsed_question, solution)
            else:
                return self._analyze_word_problem_calculation(state, parsed_question, solution)
                
        except Exception as e:
            logger.error(f"Error analyzing calculation: {e}")
            logger.debug(traceback.format_exc())
            
            # Record error event
            return self._record_event(state, {
                "status": "error",
                "error": f"Failed to analyze calculation: {str(e)}"
            })
    
    def _analyze_symbolic_calculation(self, state: Dict[str, Any], parsed_question: Dict[str, Any], solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a symbolic calculation.
        
        Args:
            state: The current state dictionary
            parsed_question: The parsed question information
            solution: The solution information
            
        Returns:
            Updated state with analysis
        """
        # Get the required information
        student_answer = state.get("student_answer", "")
        correct_answer = state.get("correct_answer", "")
        equations = parsed_question.get("equations", [])
        variables = parsed_question.get("variables", [])
        
        try:
            # Use LLM to analyze the calculation if available
            if self.agent and self.agent.llm_service:
                # Prepare the prompt
                system_prompt = """Analyze the student's answer to the symbolic math problem.
                
                Compare the student's answer with the correct answer and identify:
                - Whether the answer is correct
                - Any errors or misconceptions
                - The approach used by the student
                
                Return the analysis as a JSON object with the following structure:
                {
                    "is_correct": true/false,
                    "errors": ["Error 1", "Error 2", ...],
                    "approach": "Description of the student's approach",
                    "misconceptions": ["Misconception 1", "Misconception 2", ...],
                    "observations": ["Observation 1", "Observation 2", ...]
                }
                """
                
                user_prompt = f"""Problem: {', '.join(equations)}
                Variables: {', '.join(variables)}
                Correct answer: {correct_answer}
                Student answer: {student_answer}
                """
                
                # Generate completion
                response = self.agent.llm_service.generate_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
                
                # Extract the analysis
                content = response.get("content", "")
                
                try:
                    # Try to parse as JSON
                    try:
                        analysis_data = json.loads(content)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try to extract JSON from markdown code blocks
                        if "```json" in content:
                            json_str = content.split("```json")[1].split("```")[0].strip()
                            analysis_data = json.loads(json_str)
                        elif "```" in content:
                            json_str = content.split("```")[1].split("```")[0].strip()
                            analysis_data = json.loads(json_str)
                        else:
                            # For mock service, create default analysis
                            is_correct = is_answer_correct(student_answer, correct_answer)
                            analysis_data = {
                                "is_correct": is_correct,
                                "errors": [] if is_correct else ["Incorrect answer"],
                                "approach": "Unable to determine from the answer",
                                "misconceptions": [],
                                "observations": ["Student answer differs from correct answer"] if not is_correct else []
                            }
                    
                    # Store the analysis in the state
                    if "analysis" not in state:
                        state["analysis"] = {}
                    
                    state["analysis"]["calculation"] = analysis_data
                    
                    # Update state with analysis results
                    state["is_correct"] = analysis_data.get("is_correct", False)
                    state["errors"] = analysis_data.get("errors", [])
                    state["observations"] = analysis_data.get("observations", [])
                    
                    # Determine if the student needs a hint
                    state["needs_hint"] = not analysis_data.get("is_correct", False)
                    
                    # Calculate a score based on correctness
                    score = 10.0 if analysis_data.get("is_correct", False) else 5.0
                    
                    # Store the score in the state
                    if "scores" not in state:
                        state["scores"] = {}
                    
                    state["scores"]["accuracy"] = score
                    
                    # Record success event
                    return self._record_event(state, {"status": "success"})
                    
                except Exception as e:
                    logger.error(f"Error parsing analysis: {e}")
                    logger.debug(traceback.format_exc())
                    
                    # Create default analysis
                    is_correct = is_answer_correct(student_answer, correct_answer)
                    analysis_data = {
                        "is_correct": is_correct,
                        "errors": [] if is_correct else ["Incorrect answer"],
                        "approach": "Unable to determine from the answer",
                        "misconceptions": [],
                        "observations": ["Student answer differs from correct answer"] if not is_correct else []
                    }
                    
                    # Store the analysis in the state
                    if "analysis" not in state:
                        state["analysis"] = {}
                    
                    state["analysis"]["calculation"] = analysis_data
                    
                    # Update state with analysis results
                    state["is_correct"] = analysis_data.get("is_correct", False)
                    state["errors"] = analysis_data.get("errors", [])
                    state["observations"] = analysis_data.get("observations", [])
                    
                    # Determine if the student needs a hint
                    state["needs_hint"] = not analysis_data.get("is_correct", False)
                    
                    # Calculate a score based on correctness
                    score = 10.0 if analysis_data.get("is_correct", False) else 5.0
                    
                    # Store the score in the state
                    if "scores" not in state:
                        state["scores"] = {}
                    
                    state["scores"]["accuracy"] = score
                    
                    # Record partial success
                    return self._record_event(state, {
                        "status": "partial_success",
                        "error": str(e)
                    })
            else:
                # Simple analysis without LLM
                is_correct = is_answer_correct(student_answer, correct_answer)
                analysis_data = {
                    "is_correct": is_correct,
                    "errors": [] if is_correct else ["Incorrect answer"],
                    "approach": "Unable to determine from the answer",
                    "misconceptions": [],
                    "observations": ["Student answer differs from correct answer"] if not is_correct else []
                }
                
                # Store the analysis in the state
                if "analysis" not in state:
                    state["analysis"] = {}
                
                state["analysis"]["calculation"] = analysis_data
                
                # Update state with analysis results
                state["is_correct"] = analysis_data.get("is_correct", False)
                state["errors"] = analysis_data.get("errors", [])
                state["observations"] = analysis_data.get("observations", [])
                
                # Determine if the student needs a hint
                state["needs_hint"] = not analysis_data.get("is_correct", False)
                
                # Calculate a score based on correctness
                score = 10.0 if analysis_data.get("is_correct", False) else 5.0
                
                # Store the score in the state
                if "scores" not in state:
                    state["scores"] = {}
                
                state["scores"]["accuracy"] = score
                
                # Record success event
                return self._record_event(state, {"status": "success"})
                
        except Exception as e:
            logger.error(f"Error in symbolic calculation analysis: {e}")
            logger.debug(traceback.format_exc())
            
            # Record error event
            return self._record_event(state, {
                "status": "error",
                "error": f"Failed to analyze symbolic calculation: {str(e)}"
            })
    
    def _analyze_word_problem_calculation(self, state: Dict[str, Any], parsed_question: Dict[str, Any], solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a word problem calculation.
        
        Args:
            state: The current state dictionary
            parsed_question: The parsed question information
            solution: The solution information
            
        Returns:
            Updated state with analysis
        """
        # Get the required information
        student_answer = state.get("student_answer", "")
        correct_answer = state.get("correct_answer", "")
        solution_steps = parsed_question.get("solution_steps", [])
        
        try:
            # Use LLM to analyze the calculation if available
            if self.agent and self.agent.llm_service:
                # Prepare the prompt
                system_prompt = """Analyze the student's answer to the word problem.
                
                Compare the student's answer with the correct answer and identify:
                - Whether the answer is correct
                - Any errors or misconceptions
                - The approach used by the student
                
                Return the analysis as a JSON object with the following structure:
                {
                    "is_correct": true/false,
                    "errors": ["Error 1", "Error 2", ...],
                    "approach": "Description of the student's approach",
                    "misconceptions": ["Misconception 1", "Misconception 2", ...],
                    "observations": ["Observation 1", "Observation 2", ...]
                }
                """
                
                user_prompt = f"""Problem: {state.get('question', '')}
                Solution steps: {solution_steps}
                Correct answer: {correct_answer}
                Student answer: {student_answer}
                """
                
                # Generate completion
                response = self.agent.llm_service.generate_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
                
                # Extract the analysis
                content = response.get("content", "")
                
                try:
                    # Try to parse as JSON
                    try:
                        analysis_data = json.loads(content)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try to extract JSON from markdown code blocks
                        if "```json" in content:
                            json_str = content.split("```json")[1].split("```")[0].strip()
                            analysis_data = json.loads(json_str)
                        elif "```" in content:
                            json_str = content.split("```")[1].split("```")[0].strip()
                            analysis_data = json.loads(json_str)
                        else:
                            # For mock service, create default analysis
                            is_correct = is_answer_correct(student_answer, correct_answer)
                            analysis_data = {
                                "is_correct": is_correct,
                                "errors": [] if is_correct else ["Incorrect answer"],
                                "approach": "Unable to determine from the answer",
                                "misconceptions": [],
                                "observations": ["Student answer differs from correct answer"] if not is_correct else []
                            }
                    
                    # Store the analysis in the state
                    if "analysis" not in state:
                        state["analysis"] = {}
                    
                    state["analysis"]["calculation"] = analysis_data
                    
                    # Update state with analysis results
                    state["is_correct"] = analysis_data.get("is_correct", False)
                    state["errors"] = analysis_data.get("errors", [])
                    state["observations"] = analysis_data.get("observations", [])
                    
                    # Determine if the student needs a hint
                    state["needs_hint"] = not analysis_data.get("is_correct", False)
                    
                    # Calculate a score based on correctness
                    score = 10.0 if analysis_data.get("is_correct", False) else 5.0
                    
                    # Store the score in the state
                    if "scores" not in state:
                        state["scores"] = {}
                    
                    state["scores"]["accuracy"] = score
                    
                    # Record success event
                    return self._record_event(state, {"status": "success"})
                    
                except Exception as e:
                    logger.error(f"Error parsing analysis: {e}")
                    logger.debug(traceback.format_exc())
                    
                    # Create default analysis
                    is_correct = is_answer_correct(student_answer, correct_answer)
                    analysis_data = {
                        "is_correct": is_correct,
                        "errors": [] if is_correct else ["Incorrect answer"],
                        "approach": "Unable to determine from the answer",
                        "misconceptions": [],
                        "observations": ["Student answer differs from correct answer"] if not is_correct else []
                    }
                    
                    # Store the analysis in the state
                    if "analysis" not in state:
                        state["analysis"] = {}
                    
                    state["analysis"]["calculation"] = analysis_data
                    
                    # Update state with analysis results
                    state["is_correct"] = analysis_data.get("is_correct", False)
                    state["errors"] = analysis_data.get("errors", [])
                    state["observations"] = analysis_data.get("observations", [])
                    
                    # Determine if the student needs a hint
                    state["needs_hint"] = not analysis_data.get("is_correct", False)
                    
                    # Calculate a score based on correctness
                    score = 10.0 if analysis_data.get("is_correct", False) else 5.0
                    
                    # Store the score in the state
                    if "scores" not in state:
                        state["scores"] = {}
                    
                    state["scores"]["accuracy"] = score
                    
                    # Record partial success
                    return self._record_event(state, {
                        "status": "partial_success",
                        "error": str(e)
                    })
            else:
                # Simple analysis without LLM
                is_correct = is_answer_correct(student_answer, correct_answer)
                analysis_data = {
                    "is_correct": is_correct,
                    "errors": [] if is_correct else ["Incorrect answer"],
                    "approach": "Unable to determine from the answer",
                    "misconceptions": [],
                    "observations": ["Student answer differs from correct answer"] if not is_correct else []
                }
                
                # Store the analysis in the state
                if "analysis" not in state:
                    state["analysis"] = {}
                
                state["analysis"]["calculation"] = analysis_data
                
                # Update state with analysis results
                state["is_correct"] = analysis_data.get("is_correct", False)
                state["errors"] = analysis_data.get("errors", [])
                state["observations"] = analysis_data.get("observations", [])
                
                # Determine if the student needs a hint
                state["needs_hint"] = not analysis_data.get("is_correct", False)
                
                # Calculate a score based on correctness
                score = 10.0 if analysis_data.get("is_correct", False) else 5.0
                
                # Store the score in the state
                if "scores" not in state:
                    state["scores"] = {}
                
                state["scores"]["accuracy"] = score
                
                # Record success event
                return self._record_event(state, {"status": "success"})
                
        except Exception as e:
            logger.error(f"Error in word problem calculation analysis: {e}")
            logger.debug(traceback.format_exc())
            
            # Record error event
            return self._record_event(state, {
                "status": "error",
                "error": f"Failed to analyze word problem calculation: {str(e)}"
            })
