"""
Math Assessment Workflow using Prefect.

This module implements a Prefect-based workflow for math assessment with conditional flows
for handling incorrect answers, hints, and multiple attempts.
"""

import logging
from typing import Dict, Any, Optional
from prefect import task, flow
from prefect.task_runners import ConcurrentTaskRunner

# Import commands
from app.agents.agents.math_commands.parse_command import MathParseQuestionCommand
from app.agents.agents.math_commands.solve_command import MathSolveSymbolicallyCommand
from app.agents.agents.math_commands.analyze_command import MathAnalyzeCalculationCommand
from app.agents.agents.math_commands.assess_command import MathAssessProximityCommand
from app.agents.agents.math_commands.hint_command import MathGenerateHintCommand
from app.agents.agents.math_commands.feedback_command import MathGenerateFeedbackCommand

logger = logging.getLogger(__name__)

# Define tasks for each command
@task(name="parse_question")
def parse_question_task(state: Dict[str, Any], agent=None) -> Dict[str, Any]:
    """
    Task to parse a math question and identify variables and equations.
    
    Args:
        state: Current state with question
        agent: Optional agent instance to pass to the command
        
    Returns:
        Updated state with parsed question
    """
    logger.info("Executing parse_question_task")
    command = MathParseQuestionCommand(agent=agent)
    return command.execute(state)

@task(name="solve_symbolically")
def solve_symbolically_task(state: Dict[str, Any], agent=None) -> Dict[str, Any]:
    """
    Task to solve a math problem symbolically.
    
    Args:
        state: Current state with parsed question
        agent: Optional agent instance to pass to the command
        
    Returns:
        Updated state with solution
    """
    logger.info("Executing solve_symbolically_task")
    command = MathSolveSymbolicallyCommand(agent=agent)
    return command.execute(state)

@task(name="analyze_calculation")
def analyze_calculation_task(state: Dict[str, Any], agent=None) -> Dict[str, Any]:
    """
    Task to analyze the mathematical calculations in a student's answer.
    
    Args:
        state: Current state with student answer and correct answer
        agent: Optional agent instance to pass to the command
        
    Returns:
        Updated state with analysis
    """
    logger.info("Executing analyze_calculation_task")
    command = MathAnalyzeCalculationCommand(agent=agent)
    return command.execute(state)

@task(name="assess_proximity")
def assess_proximity_task(state: Dict[str, Any], agent=None) -> Dict[str, Any]:
    """
    Task to assess how close the student's answer is to the correct answer.
    
    Args:
        state: Current state with student answer and correct answer
        agent: Optional agent instance to pass to the command
        
    Returns:
        Updated state with proximity assessment
    """
    logger.info("Executing assess_proximity_task")
    command = MathAssessProximityCommand(agent=agent)
    return command.execute(state)

@task(name="generate_hint")
def generate_hint_task(state: Dict[str, Any], agent=None) -> Dict[str, Any]:
    """
    Task to generate a hint for the student based on their answer and the correct answer.
    
    Args:
        state: Current state with student answer, correct answer, and hint count
        agent: Optional agent instance to pass to the command
        
    Returns:
        Updated state with hint
    """
    logger.info("Executing generate_hint_task")
    command = MathGenerateHintCommand(agent=agent)
    return command.execute(state)

@task(name="generate_feedback")
def generate_feedback_task(state: Dict[str, Any], agent=None) -> Dict[str, Any]:
    """
    Task to generate detailed feedback based on the analysis results.
    
    Args:
        state: Current state with analysis results
        agent: Optional agent instance to pass to the command
        
    Returns:
        Updated state with feedback
    """
    logger.info("Executing generate_feedback_task")
    command = MathGenerateFeedbackCommand(agent=agent)
    return command.execute(state)

@flow(name="math_assessment_flow", task_runner=ConcurrentTaskRunner())
def math_assessment_flow(
    initial_state: Dict[str, Any], 
    agent=None, 
    max_attempts: int = 3,
    attempt: int = 1
) -> Dict[str, Any]:
    """
    Prefect flow for math assessment with conditional logic for hints and multiple attempts.
    
    Args:
        initial_state: Initial state with question and student answer
        agent: Optional agent instance to pass to commands
        max_attempts: Maximum number of attempts allowed
        attempt: Current attempt number
        
    Returns:
        Final state with assessment results
    """
    logger.info(f"Starting math assessment flow (attempt {attempt}/{max_attempts})")
    
    # Initialize state if needed
    state = initial_state.copy()
    if "hint_count" not in state:
        state["hint_count"] = 0
    if "hints" not in state:
        state["hints"] = []
    if "needs_hint" not in state:
        state["needs_hint"] = False
    if "attempts" not in state:
        state["attempts"] = attempt
    else:
        state["attempts"] = attempt
    
    # Core assessment flow
    state = parse_question_task(state, agent)
    state = solve_symbolically_task(state, agent)
    state = analyze_calculation_task(state, agent)
    state = assess_proximity_task(state, agent)
    
    # Check if answer is correct
    is_correct = state.get("is_correct", False)
    
    # If answer is correct or max attempts reached, generate feedback
    if is_correct or attempt >= max_attempts:
        state = generate_feedback_task(state, agent)
        return state
    
    # If answer is incorrect and more attempts are allowed, generate hint
    state = generate_hint_task(state, agent)
    
    # Return state with hint for user to try again
    # The actual retry will be handled by the caller
    return state

@flow(name="math_assessment_with_retry_flow")
def math_assessment_with_retry_flow(
    initial_state: Dict[str, Any],
    agent=None,
    max_attempts: int = 3
) -> Dict[str, Any]:
    """
    High-level flow that handles the retry logic for math assessment.
    
    Args:
        initial_state: Initial state with question and student answer
        agent: Optional agent instance to pass to commands
        max_attempts: Maximum number of attempts allowed
        
    Returns:
        Final state with assessment results
    """
    state = initial_state.copy()
    attempt = 1
    
    while attempt <= max_attempts:
        logger.info(f"Starting attempt {attempt}/{max_attempts}")
        
        # Run the assessment flow
        state = math_assessment_flow(state, agent, max_attempts, attempt)
        
        # Check if the answer is correct
        is_correct = state.get("is_correct", False)
        
        if is_correct:
            logger.info(f"Answer correct on attempt {attempt}")
            break
        
        # If this was the last attempt, we're done
        if attempt >= max_attempts:
            logger.info(f"Max attempts ({max_attempts}) reached")
            break
        
        # Increment attempt counter
        attempt += 1
        
        # In a real application, we would wait for user input here
        # For now, we'll just continue with the next attempt
        logger.info(f"Proceeding to attempt {attempt}")
    
    # Ensure feedback is generated for the final state
    if "feedback" not in state:
        state = generate_feedback_task(state, agent)
    
    return state
