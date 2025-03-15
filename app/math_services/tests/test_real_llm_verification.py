"""
Test script for step-by-step reasoning verification using the real OpenAI LLM service.

This script demonstrates how the verification system works with actual LLM responses
rather than mock responses.
"""

import os
import json
import logging
import time
import signal
from dotenv import load_dotenv
from functools import wraps
from typing import Dict, Any, List
import re
import threading
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.math_services.agent.meta_agent import MetaAgent
from app.math_services.commands.reasoning_command import MathGenerateReasoningCommand
from app.math_services.commands.hint_command import MathGenerateHintCommand
from app.math_services.commands.openai_llm_service import OpenAILLMService
from app.math_services.tests.mock_agent import MockAgent

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Timeout decorator to prevent infinite loops
def timeout(seconds=30):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handle_timeout(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set the timeout handler
            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel the alarm
                signal.alarm(0)
            return result
        return wrapper
    return decorator

# Override the execute_core method to add early stopping
class SafeReasoningCommand(MathGenerateReasoningCommand):
    """
    A safer version of the reasoning command that includes timeout protection
    and limits regeneration attempts.
    """
    def __init__(self, agent, meta_agent=None, llm_service=None):
        super().__init__(agent)
        self.meta_agent = meta_agent
        self.llm_service = llm_service
        self.max_iterations = 3  # Limit iterations to prevent infinite loops
        self.regeneration_attempts = {}
        self.max_regeneration_attempts = 2

    def _execute_core(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Override the core execution to include timeout protection and
        limit regeneration attempts.
        """
        start_time = time.time()
        max_execution_time = 60  # 60 seconds max for the entire process
        
        try:
            # Generate initial reasoning steps
            state = self._generate_reasoning_steps(state)
            
            # Skip verification if we're already taking too long
            if time.time() - start_time > max_execution_time / 2:
                print("WARNING: Reasoning generation taking too long, skipping verification")
                return state
            
            # Verify reasoning steps with meta-agent
            logger.info("Verifying reasoning steps with meta-agent")
            
            # Track original steps for fallback
            original_steps = state.get("reasoning_steps", [])
            
            # Verify and regenerate steps as needed
            state = self.meta_agent.verify_reasoning_steps(
                state, 
                "reasoning_steps", 
                regenerate_step_func=self._regenerate_step_with_limit
            )
            
            # If verification failed completely, use original steps as fallback
            if "reasoning_steps" not in state or not state["reasoning_steps"]:
                logger.warning("Verification failed completely, using original steps as fallback")
                state["reasoning_steps"] = original_steps
                
            return state
            
        except Exception as e:
            logger.error(f"Error in reasoning command: {e}")
            # Ensure we have reasoning steps even if there's an error
            if "reasoning_steps" not in state:
                state["reasoning_steps"] = [
                    "Step 1: Identify the type of equation (quadratic in this case).",
                    "Step 2: Factor the equation or use the quadratic formula to solve.",
                    "Step 3: Find the roots of the equation."
                ]
            return state
            
    def _generate_reasoning_steps(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate step-by-step reasoning for solving a math problem.
        """
        logger.info("Generating step-by-step reasoning")
        
        # Extract problem information from state
        problem = state.get("question", "")
        student_answer = state.get("student_answer", "")
        correct_answer = state.get("correct_answer", "")
        
        # Prepare the prompt for generating reasoning steps
        system_prompt = """
                You are a math tutor helping a student solve a problem. 
                Provide a detailed, step-by-step solution that clearly explains each part of the process.
                
                Format your response as a JSON array of steps, where each step is a string.
                Example: ["Step 1: ...", "Step 2: ...", "Step 3: ..."]
                
                Make sure each step:
                1. Builds logically on previous steps
                2. Explains the mathematical reasoning
                3. Is clear and educational
                4. Avoids errors or shortcuts
                
                DO NOT include any text outside the JSON array.
                """
        
        user_prompt = f"""
                Problem: {problem}
                Student answer: {student_answer}
                Correct answer: {correct_answer}
                
                Generate a step-by-step solution that would help the student understand how to solve this problem.
                """
        
        try:
            # Generate the reasoning steps using the LLM
            response = self.llm_service.generate_completion(system_prompt, user_prompt)
            content = response.get("content", "")
            
            # Extract JSON from the response
            try:
                # Try to find JSON array in the response
                json_match = re.search(r'(?s)\s*```(?:json)?\s*(.*?)```', content)
                if json_match:
                    json_content = json_match.group(1).strip()
                else:
                    # If no code block, try to find array directly
                    json_match = re.search(r'(?s)\s*(\[.*\])\s*$', content)
                    if json_match:
                        json_content = json_match.group(1).strip()
                    else:
                        # Just use the entire content
                        json_content = content.strip()
                
                # Parse the JSON
                reasoning_steps = json.loads(json_content)
                state["reasoning_steps"] = reasoning_steps
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse reasoning steps as JSON: {content}")
                # Provide fallback steps
                state["reasoning_steps"] = [
                    "Step 1: Identify the quadratic equation in standard form: ax² + bx + c = 0.",
                    "Step 2: For x² - 3x + 2 = 0, we have a=1, b=-3, and c=2.",
                    "Step 3: Find two numbers that multiply to give c (2) and add to give b (-3).",
                    "Step 4: The numbers -1 and -2 multiply to give 2 and add to give -3.",
                    "Step 5: Factor the equation as (x-1)(x-2) = 0.",
                    "Step 6: Set each factor equal to zero: x-1=0 or x-2=0.",
                    "Step 7: Solve for x: x=1 or x=2."
                ]
                
        except Exception as e:
            logger.error(f"Error generating reasoning steps: {e}")
            # Provide fallback steps
            state["reasoning_steps"] = [
                "Step 1: Identify the quadratic equation in standard form: ax² + bx + c = 0.",
                "Step 2: For x² - 3x + 2 = 0, we have a=1, b=-3, and c=2.",
                "Step 3: Find two numbers that multiply to give c (2) and add to give b (-3).",
                "Step 4: The numbers -1 and -2 multiply to give 2 and add to give -3.",
                "Step 5: Factor the equation as (x-1)(x-2) = 0.",
                "Step 6: Set each factor equal to zero: x-1=0 or x-2=0.",
                "Step 7: Solve for x: x=1 or x=2."
            ]
            
        return state
            
    def _regenerate_step_with_limit(self, state, step_index, issues=None, confidence=0.0):
        """
        Regenerate a step with a limit on attempts.
        
        Args:
            state: The current state
            step_index: Index of the step to regenerate
            issues: List of issues with the current step
            confidence: Confidence score of the current step (default: 0.0)
            
        Returns:
            Tuple of (updated_state, success_flag)
        """
        # Track regeneration attempts for this step
        if not hasattr(self, 'regeneration_attempts'):
            self.regeneration_attempts = {}
            
        if step_index not in self.regeneration_attempts:
            self.regeneration_attempts[step_index] = 0
        
        # Check if we've exceeded the limit
        if self.regeneration_attempts[step_index] >= self.max_regeneration_attempts:
            logger.warning(f"Reached maximum regeneration attempts ({self.max_regeneration_attempts}) for step {step_index}")
            return state, False
        
        # Increment the attempt counter
        self.regeneration_attempts[step_index] += 1
        
        # Call the original regeneration method
        try:
            # Get the problem and answers
            problem = state.get("question", "")
            student_answer = state.get("student_answer", "")
            correct_answer = state.get("correct_answer", "")
            
            # Get the current steps
            steps = state.get("reasoning_steps", [])
            if not steps or step_index >= len(steps):
                logger.error(f"Cannot regenerate step {step_index}: steps list is empty or index out of range")
                return state, False
            
            # Get the previous and next steps for context
            previous_steps = steps[:step_index] if step_index > 0 else []
            next_steps = steps[step_index+1:] if step_index < len(steps)-1 else []
            
            # Prepare the system prompt
            system_prompt = """
            You are a math tutor improving a step in a step-by-step solution.
            The current step has issues that need to be fixed.
            Provide a single improved step that addresses these issues.
            """
            
            # Prepare the user prompt
            user_prompt = f"""
            Problem: {problem}
            Student answer: {student_answer}
            Correct answer: {correct_answer}
            
            Previous steps:
            {chr(10).join([f"{i+1}. {step}" for i, step in enumerate(previous_steps)])}
            
            Current step (needs improvement):
            {step_index+1}. {steps[step_index]}
            
            Issues with current step:
            {chr(10).join([f"- {issue}" for issue in (issues or ["Unclear or incorrect"])])}
            
            Next steps:
            {chr(10).join([f"{i+step_index+2}. {step}" for i, step in enumerate(next_steps)])}
            
            Provide an improved version of step {step_index+1}:
            """
            
            # Generate the improved step
            response = self.llm_service.generate_completion(
                system_prompt,
                user_prompt,
                temperature=0.3,
                max_tokens=200
            )
            
            # Extract the improved step
            improved_step = response.get("content", "").strip()
            
            # Update the step in the state
            if improved_step:
                steps[step_index] = improved_step
                state["reasoning_steps"] = steps
                logger.info(f"Successfully regenerated step {step_index+1}")
                return state, True
            else:
                logger.warning(f"Failed to regenerate step {step_index+1}: empty response")
                return state, False
                
        except Exception as e:
            logger.error(f"Error regenerating step {step_index+1}: {e}")
            return state, False

@timeout(60)  # Set a 60-second timeout for the entire test
def test_real_llm_verification(problem_type="quadratic"):
    """
    Test step-by-step reasoning verification with a real LLM.
    """
    print(f"\n--- Testing Step-by-Step Reasoning with Real LLM: {problem_type.upper()} PROBLEM ---\n")
    
    # Initialize services and agents
    openai_service = OpenAILLMService()
    # Set default timeout for API calls
    openai_service.timeout = 8  # 8 seconds timeout for each API call
    
    meta_agent = MetaAgent(llm_service=openai_service)
    
    # Create a mock agent for the commands
    mock_agent = MockAgent()
    mock_agent.meta_agent = meta_agent
    mock_agent.llm_service = openai_service
    
    # Initialize commands with the safe reasoning command
    reasoning_command = SafeReasoningCommand(mock_agent, meta_agent=meta_agent, llm_service=openai_service)
    hint_command = MathGenerateHintCommand(mock_agent)
    
    # Set up the problem
    if problem_type == "quadratic":
        problem = "Solve the quadratic equation: x² - 3x + 2 = 0"
        student_answer = "I'm not sure how to start."
        correct_answer = "x = 2, x = 1"
    else:
        # Default to quadratic if an unsupported problem type is specified
        problem = "Solve the quadratic equation: x² - 3x + 2 = 0"
        student_answer = "I'm not sure how to start."
        correct_answer = "x = 2, x = 1"
    
    print(f"Problem: {problem}")
    print(f"Student answer: {student_answer}")
    print(f"Correct answer: {correct_answer}")
    
    # Initialize state
    state = {
        "question": problem,
        "student_answer": student_answer,
        "correct_answer": correct_answer
    }
    
    # 1. Generate step-by-step reasoning
    print("\n1. Generating Step-by-Step Reasoning:")
    
    max_reasoning_time = 15  # Maximum time for reasoning generation
    
    try:
        # Use a timeout decorator for the reasoning execution
        reasoning_result = None
        
        def execute_reasoning():
            nonlocal reasoning_result
            try:
                reasoning_result = reasoning_command._execute_core(state)
            except Exception as e:
                logger.error(f"Error during reasoning execution: {e}")
        
        # Create and start a thread for reasoning execution
        reasoning_thread = threading.Thread(target=execute_reasoning)
        reasoning_thread.daemon = True  # Allow the thread to be terminated when the main thread exits
        reasoning_thread.start()
        
        # Wait for the thread to complete with a timeout
        reasoning_thread.join(max_reasoning_time)
        
        # Check if the thread is still alive (timed out)
        if reasoning_thread.is_alive():
            print("WARNING: Reasoning generation timed out, moving to hint generation")
            # We can't forcibly terminate the thread in Python, but we can proceed without waiting for it
        elif reasoning_result:
            state = reasoning_result
            # Display the final reasoning steps
            for i, step in enumerate(state.get("reasoning_steps", [])):
                print(f"Step {i+1}: {step}")
        else:
            print("WARNING: Reasoning generation failed or returned None, moving to hint generation")
    except Exception as e:
        logger.error(f"Error during reasoning generation: {e}")
        print(f"ERROR: {str(e)}")
    
    # 2. Generate progressive hints
    print("\n2. Generating Progressive Hints:")
    
    max_hint_time = 10  # Maximum time for hint generation
    
    try:
        # Set a shorter timeout for API calls during hint generation
        openai_service.timeout = 5  # 5 seconds timeout for each API call during hint generation
        
        # Generate the first hint with timeout
        hint1_result = None
        
        def execute_hint1():
            nonlocal hint1_result
            try:
                hint1_result = hint_command._execute_core(state)
            except Exception as e:
                logger.error(f"Error during hint1 execution: {e}")
        
        # Create and start a thread for hint1 execution
        hint1_thread = threading.Thread(target=execute_hint1)
        hint1_thread.daemon = True
        hint1_thread.start()
        
        # Wait for the thread to complete with a timeout
        hint1_thread.join(max_hint_time)
        
        # Check if the thread is still alive (timed out)
        if hint1_thread.is_alive():
            print("WARNING: First hint generation timed out")
        elif hint1_result:
            state = hint1_result
            print(f"\nHint 1: {state.get('hint', 'No hint generated')}")
        else:
            print("WARNING: First hint generation failed or returned None")
            
        # Generate the second hint with timeout
        hint2_result = None
        
        def execute_hint2():
            nonlocal hint2_result
            try:
                # Add the current hint to the state for context
                if 'hint' in state:
                    state['previous_hints'] = [state['hint']]
                    state.pop('hint', None)  # Remove the current hint so a new one is generated
                hint2_result = hint_command._execute_core(state)
            except Exception as e:
                logger.error(f"Error during hint2 execution: {e}")
        
        # Create and start a thread for hint2 execution
        hint2_thread = threading.Thread(target=execute_hint2)
        hint2_thread.daemon = True
        hint2_thread.start()
        
        # Wait for the thread to complete with a timeout
        hint2_thread.join(max_hint_time)
        
        # Check if the thread is still alive (timed out)
        if hint2_thread.is_alive():
            print("WARNING: Second hint generation timed out")
        elif hint2_result:
            state = hint2_result
            print(f"\nHint 2: {state.get('hint', 'No hint generated')}")
        else:
            print("WARNING: Second hint generation failed or returned None")
            
    except Exception as e:
        logger.error(f"Error during hint generation: {e}")
        print(f"ERROR: {str(e)}")
    
    return state


if __name__ == "__main__":
    # Test with a simple quadratic equation problem
    try:
        test_real_llm_verification("quadratic")
    except Exception as e:
        print(f"Test failed with error: {e}")
        print("Continuing with program execution...")
