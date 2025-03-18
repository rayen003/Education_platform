"""
Meta-Agent for verifying outputs from other agents.

This module contains a meta-agent that verifies and potentially corrects
outputs from other agents to enhance robustness through iterative verification.
"""

import logging
import json
import re
import time
import signal
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from openai import OpenAI
import os
from dotenv import load_dotenv
import queue
import threading
from contextlib import contextmanager
from app.math_services.models.state import MathState
from app.math_services.services.llm.base_service import BaseLLMService
import concurrent.futures

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Define the timeout context manager
@contextmanager
def timeout(seconds):
    """
    Context manager for timing out operations.
    
    Args:
        seconds: Number of seconds to wait before timing out
        
    Raises:
        TimeoutError: If the operation times out
    """
    def handle_timeout(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the timeout handler
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Cancel the alarm
        signal.alarm(0)

class MetaAgent:
    """
    Meta-agent for verifying and correcting outputs from other agents.
    
    This agent implements an iterative verification process that:
    1. Assesses the mathematical correctness of outputs
    2. Evaluates the quality of reasoning steps
    3. Regenerates outputs when confidence thresholds aren't met
    """
    
    def __init__(self, model="gpt-4o-mini", confidence_threshold=0.8, max_iterations=3, llm_service=None):
        """
        Initialize the meta agent.
        
        Args:
            model: The OpenAI model to use
            confidence_threshold: Minimum confidence for verification
            max_iterations: Maximum regeneration iterations
            llm_service: Optional LLM service to use (will create one if None)
        """
        # Store parameters
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY") or "mock-key"
        self.client = OpenAI(api_key=api_key)
        
        # Use provided LLM service or create a new one
        if llm_service:
            self.llm_service = llm_service
        else:
            from app.math_services.services.llm.openai_service import OpenAILLMService
            self.llm_service = OpenAILLMService(model=model)
        
        logger.info(f"Initialized MetaAgent with model {model}")
    
    def verify_output(self, state: Dict[str, Any], output_type: str, regenerate_func: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Verify an output from another agent and correct if necessary.
        Uses an iterative approach to regenerate outputs until confidence threshold is met.
        
        Args:
            state: The current state dictionary
            output_type: Type of output to verify (e.g., "hint", "feedback", "solution")
            regenerate_func: Optional function to call for regenerating output if verification fails
            
        Returns:
            Updated state with verified output
        """
        # Iterative verification process
        iterations = 0
        max_iterations = self.max_iterations
        
        while iterations < max_iterations:
            # Extract the output to verify based on output_type
            output = None
            if output_type == "hint" and "hints" in state and state["hints"]:
                output = state["hints"][-1]
            elif output_type == "feedback" and "feedback" in state and state["feedback"]:
                output = state["feedback"][-1]
            elif output_type in state:
                if isinstance(state[output_type], list) and state[output_type]:
                    # If it's a list, verify the last item
                    output = state[output_type][-1]
                else:
                    # Otherwise verify the entire output
                    output = state[output_type]
            
            if output is None:
                logger.warning(f"No {output_type} found in state to verify")
                return state
            
            # Verify the output
            verification_result, confidence = self._verify_single_output(state, output_type, output)
            
            # Update the state with verification results
            state.update(verification_result)
            
            # Add iteration count to verification result
            if f"{output_type}_verification" in state:
                state[f"{output_type}_verification"]["iterations"] = iterations + 1
            
            # Check if we've reached the confidence threshold
            if confidence >= self.confidence_threshold:
                logger.info(f"{output_type.capitalize()} verification succeeded with confidence {confidence:.2f}")
                break
            
            # If we haven't reached the threshold and have a regenerate function, use it
            if regenerate_func and iterations < max_iterations - 1:
                logger.info(f"{output_type.capitalize()} verification failed with confidence {confidence:.2f}. Regenerating...")
                state = regenerate_func(state)
            else:
                logger.warning(f"{output_type.capitalize()} verification failed with confidence {confidence:.2f}, but no regeneration function provided or max iterations reached")
                break
            
            iterations += 1
        
        return state
    
    def _verify_single_output(self, state: Dict[str, Any], output_type: str, output: Any) -> Tuple[Dict[str, Any], float]:
        """
        Perform a single verification of an output.
        
        Args:
            state: The current state dictionary
            output_type: Type of output to verify
            output: The actual output to verify
            
        Returns:
            Tuple of (verification_result, confidence_score)
        """
        # Prepare verification prompt
        system_prompt = f"""
        You are a verification agent for educational content. Your task is to verify the {output_type} provided by another agent.
        
        Assess the {output_type} for:
        1. Mathematical correctness
        2. Quality of reasoning steps (if applicable)
        3. Clarity and educational value
        4. Appropriate formatting and tone
        
        Provide:
        1. A verification status (true/false)
        2. A confidence score (0-100) representing your certainty in the verification
        3. A list of specific issues identified, if any
        4. A corrected version of the output, if needed
        
        Format your response as a JSON object with these fields:
        {{
            "verified": true/false,
            "confidence": 0-100,
            "reasoning_quality": 0-100 or null if not applicable,
            "issues": ["issue1", "issue2", ...] or [] if none,
            "corrected_output": "corrected version" or null if not needed
        }}
        """
        
        # Construct the user prompt with context
        user_prompt = f"""
        Problem: {state.get('question', '')}
        Student answer: {state.get('student_answer', '')}
        Correct answer: {state.get('correct_answer', '')}
        
        {output_type.capitalize()} to verify:
        {json.dumps(output) if isinstance(output, dict) else output}
        
        Verify this {output_type} for correctness and provide your assessment as a JSON object.
        """
        
        # Call the LLM service
        try:
            response = self.llm_service.generate_completion(system_prompt, user_prompt)
            content = response.get("content", "")
            
            # Parse the JSON response
            try:
                verification_data = json.loads(content)
                
                # Extract the verification details
                verified = verification_data.get("verified", False)
                confidence = verification_data.get("confidence", 0) / 100.0  # Convert to 0-1 scale
                reasoning_quality = verification_data.get("reasoning_quality", None)
                issues = verification_data.get("issues", [])
                corrected_output = verification_data.get("corrected_output", None)
                
                # Create the verification result
                verification_result = {
                    f"{output_type}_verification": {
                        "verified": verified,
                        "confidence": confidence,
                        "reasoning_quality": reasoning_quality,
                        "issues": issues,
                        "corrected_output": corrected_output
                    }
                }
                
                # If not verified and a corrected output is provided, use it
                if not verified and corrected_output:
                    if output_type in state:
                        if isinstance(state[output_type], list):
                            # Replace the last item in the list
                            state[output_type][-1] = corrected_output
                        else:
                            # Replace the entire output
                            state[output_type] = corrected_output
                
                # Add the verification feedback to the state for potential regeneration
                state["verification_feedback"] = issues
                state["verification_confidence"] = confidence
                
                return verification_result, confidence
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing verification response: {e}")
                logger.error(f"Response content: {content}")
                
                # Return a default result with low confidence
                return {
                    f"{output_type}_verification": {
                        "verified": False,
                        "confidence": 0.0,
                        "reasoning_quality": None,
                        "issues": ["Error parsing verification response"],
                        "corrected_output": None
                    }
                }, 0.0
            
        except Exception as e:
            logger.error(f"Error verifying {output_type}: {e}")
            return {
                f"{output_type}_verification": {
                    "verified": False,
                    "confidence": 0.0,
                    "reasoning_quality": None,
                    "issues": [f"Verification error: {str(e)}"],
                    "corrected_output": None
                }
            }, 0.0
    
    def verify_reasoning_steps(self, state: Union[Dict[str, Any], MathState], steps_key: str, regenerate_step_func: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Verify a list of reasoning steps and regenerate problematic ones.
        
        Args:
            state: State object (MathState or dict) containing the steps
            steps_key: Key to access the steps (dot notation for nested access)
            regenerate_step_func: Function to call to regenerate a step
            
        Returns:
            Verification results dictionary
        """
        logger.info(f"Verifying reasoning steps with key: {steps_key}")
        
        # Handle MathState objects
        if isinstance(state, MathState):
            # Try to get steps from context
            if "reasoning" in state.context and "steps" in state.context["reasoning"]:
                steps = state.context["reasoning"]["steps"]
            # Then try direct access using dot notation
            elif "." in steps_key:
                steps = self._get_nested_value(state.context, steps_key)
            # Then try direct attribute
            elif hasattr(state, steps_key):
                steps = getattr(state, steps_key)
            else:
                # Finally try state.context directly
                steps = state.context.get(steps_key)
                
            # Convert to dict for later use if needed
            state_dict = state.to_dict()
        else:
            # For dict states, try the key directly
            state_dict = state
            steps = state_dict.get(steps_key)
            
            # If not found, try nested using dot notation
            if steps is None and "." in steps_key:
                steps = self._get_nested_value(state_dict, steps_key)
                
            # If still not found, try reasoning.steps as fallback
            if steps is None and "reasoning" in state_dict and "steps" in state_dict["reasoning"]:
                steps = state_dict["reasoning"]["steps"]
        
        # If steps not found, return early
        if not steps:
            logger.warning(f"No steps found at {steps_key}")
            return {"verified": False, "error": f"No steps found at {steps_key}"}
        
        # Initialize variables for verification
        verified_steps = []
        problematic_steps = []
        regenerated_steps = []
        all_verified = True
        
        # Verify each step
        for i, step in enumerate(steps):
            logger.info(f"Verifying step {i+1}/{len(steps)}")
            
            try:
                # Use a timeout to prevent hanging on verification
                with timeout(20):  # 20 second timeout per step
                    verification_result = self._verify_step(step)
                
                verified = verification_result.get("verified", False)
                confidence = verification_result.get("confidence", 0)
                
                # Store verification information
                step_result = {
                    "index": i,
                    "step": step,
                    "verified": verified,
                    "confidence": confidence,
                    "issues": verification_result.get("issues", [])
                }
                
                if verified and confidence >= self.confidence_threshold:
                    verified_steps.append(step_result)
                else:
                    all_verified = False
                    problematic_steps.append(step_result)
                    
                    # Regenerate the step if function provided
                    if regenerate_step_func:
                        logger.info(f"Regenerating step {i+1}")
                        
                        # Extract feedback for regeneration
                        feedback = verification_result.get("issues", ["Unclear or incorrect step"])
                        feedback_str = feedback[0] if feedback else "Unclear or incorrect step"
                        
                        # Call the regeneration function
                        try:
                            new_step = regenerate_step_func(i, feedback_str)
                            
                            # Update the step in the original list
                            steps[i] = new_step
                            
                            # Record the regeneration
                            regenerated_steps.append({
                                "index": i,
                                "original": step,
                                "regenerated": new_step,
                                "feedback": feedback
                            })
                            
                            logger.info(f"Successfully regenerated step {i+1}")
                        except Exception as regen_error:
                            logger.error(f"Error regenerating step {i+1}: {str(regen_error)}")
            except Exception as e:
                logger.error(f"Error verifying step {i+1}: {str(e)}")
                all_verified = False
                problematic_steps.append({
                    "index": i,
                    "step": step,
                    "verified": False,
                    "confidence": 0,
                    "issues": [f"Verification error: {str(e)}"]
                })
        
        # Prepare the verification summary
        verification_summary = {
            "verified": all_verified,
            "verified_steps": verified_steps,
            "problematic_steps": problematic_steps,
            "regenerated_steps": regenerated_steps,
            "all_steps": steps
        }
        
        # Update the state with verification results
        if isinstance(state, MathState):
            if "." in steps_key:
                parent_key, child_key = steps_key.rsplit(".", 1)
                parent = self._get_nested_value(state.context, parent_key)
                if parent and isinstance(parent, dict):
                    parent[child_key] = steps
                    parent["verification_results"] = verification_summary
            else:
                state.context[steps_key] = steps
                state.context["verification_results"] = verification_summary
        
        logger.info(f"Verification complete: {len(verified_steps)} verified, {len(problematic_steps)} problematic, {len(regenerated_steps)} regenerated")
        return verification_summary
    
    def _verify_step(self, step: str) -> Dict[str, Any]:
        """
        Verify a single step of reasoning.
        
        Args:
            step: The step to verify
            
        Returns:
            Verification result with status, confidence, and corrected step if needed
        """
        # Prepare verification prompt
        system_prompt = """
        You are a mathematical reasoning verifier. Your task is to verify the correctness and quality of a step in a mathematical solution.
        
        For the given step, determine if it is:
        1. Mathematically correct
        2. Logically follows from previous steps
        3. Clear and educational
        
        Return a JSON object with the following structure:
        {
            "verified": boolean,
            "confidence": number (0-100),
            "reasoning_quality": number (0-100),
            "issues": [list of specific issues found],
            "corrected_step": string (corrected version of the step if needed, otherwise null),
            "explanation": string (brief explanation of your assessment)
        }
        """
        
        # Construct user prompt
        user_prompt = f"""
        Step to verify: {step}
        
        Verify this step for mathematical correctness and provide your assessment as a JSON object.
        """
        
        try:
            response = self.llm_service.generate_completion(system_prompt, user_prompt)
            content = response.get("content", "")
            
            try:
                verification_data = json.loads(content)
                
                # Add the original step to the result
                verification_data["original_step"] = step
                
                return verification_data
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse step verification result as JSON: {content}")
                return {
                    "verified": False,
                    "confidence": 0,
                    "issues": ["Failed to parse verification result"],
                    "original_step": step
                }
            
        except Exception as e:
            logger.error(f"Error verifying step: {e}")
            return {
                "verified": False,
                "confidence": 0,
                "issues": [f"Verification error: {str(e)}"],
                "original_step": step
            }

    def verify_math_solution(self, problem: str, solution: str) -> Dict[str, Any]:
        """
        Specifically verify a mathematical solution for correctness.
        
        Args:
            problem: The mathematical problem
            solution: The proposed solution
            
        Returns:
            Verification result with status, confidence, and corrected solution if needed
        """
        system_prompt = """
        You are a mathematical verification agent. Your task is to verify if the provided solution to a math problem is correct.
        
        Check for:
        1. Correct application of mathematical principles
        2. Accurate calculations
        3. Valid reasoning steps
        4. Proper final answer
        
        Provide a detailed assessment with:
        1. A confidence score (0-100%) indicating your certainty in the solution's correctness
        2. An evaluation of reasoning quality (0-100%) based on the steps shown
        3. A list of specific mathematical errors, if any
        4. A corrected solution, if needed
        
        Format your response as a JSON object with these fields:
        {
            "verified": true/false,
            "confidence": 0-100,
            "reasoning_quality": 0-100,
            "issues": ["issue1", "issue2", ...] or [] if none,
            "corrected_solution": "corrected solution" or null if not needed
        }
        """
        
        user_prompt = f"""
        Problem: {problem}
        Proposed solution: {solution}
        
        Verify this solution for mathematical correctness and provide your assessment as a JSON object.
        """
        
        try:
            response = self.llm_service.generate_completion(system_prompt, user_prompt)
            content = response.get("content", "")
            
            try:
                verification_data = json.loads(content)
                
                # Add the original solution to the result
                verification_data["original_solution"] = solution
                
                return verification_data
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse math verification result as JSON: {content}")
                return {
                    "verified": False,
                    "confidence": 0,
                    "issues": ["Failed to parse verification result"],
                    "original_solution": solution
                }
            
        except Exception as e:
            logger.error(f"Error verifying math solution: {e}")
            return {
                "verified": False,
                "confidence": 0,
                "issues": [f"Verification error: {str(e)}"],
                "original_solution": solution
            }

    def verify_hint(self, problem: str, hint: str) -> Dict[str, Any]:
        """
        Verify the quality and correctness of a hint.
        
        Args:
            problem: The math problem
            hint: The hint to verify
            
        Returns:
            Dictionary with verification results including:
            - verified: Boolean indicating if the hint is verified
            - confidence: Confidence score (0-1)
            - reason: Reason for verification failure if not verified
        """
        try:
            # Prepare the system prompt
            system_prompt = """
            You are a math education expert evaluating the quality of a hint for a math problem.
            Assess whether the hint is:
            1. Mathematically correct
            2. Helpful without giving away the full solution
            3. Clear and understandable
            4. Appropriate for the problem difficulty
            
            Provide a confidence score from 0 to 1, where:
            - 0.0-0.3: Poor hint with serious issues
            - 0.4-0.7: Mediocre hint with some issues
            - 0.8-1.0: Good hint that meets all criteria
            
            Format your response as a JSON object with these fields:
            {
                "verified": boolean,
                "confidence": float,
                "reason": string (only if not verified)
            }
            """
            
            # Prepare the user prompt
            user_prompt = f"""
            Problem: {problem}
            
            Hint: {hint}
            
            Evaluate this hint and provide your assessment as JSON.
            """
            
            # Generate the verification using the LLM service
            response = self.llm_service.generate_completion(
                system_prompt,
                user_prompt,
                temperature=0.1,
                max_tokens=200
            )
            
            # Extract the JSON from the response
            content = response.get("content", "")
            
            # Handle the case where content is a string
            if isinstance(content, str):
                try:
                    # Try to parse JSON from the content
                    import json
                    import re
                    
                    # Look for JSON-like structure in the content
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        verification_data = json.loads(json_str)
                    else:
                        # If no JSON structure found, create a default response
                        verification_data = {
                            "verified": True,
                            "confidence": 0.8,
                            "reason": "Default verification (no JSON found in response)"
                        }
                except Exception as e:
                    # If JSON parsing fails, create a default response
                    verification_data = {
                        "verified": True,
                        "confidence": 0.7,
                        "reason": f"Default verification (JSON parsing error: {str(e)})"
                    }
            else:
                # If content is not a string (e.g., already a dict), use it directly
                verification_data = content
            
            # Ensure the verification data has the required fields
            if "verified" not in verification_data:
                verification_data["verified"] = True
            if "confidence" not in verification_data:
                verification_data["confidence"] = 0.7
            
            return verification_data
            
        except Exception as e:
            # Return a default verification result in case of errors
            logger.error(f"Error verifying hint: {e}")
            return {
                "verified": True,
                "confidence": 0.6,
                "reason": f"Verification error: {str(e)}"
            }
    
    def _extract_json(self, content: str) -> str:
        """
        Extract JSON from LLM response content using multiple patterns.
        Handles escape characters and cleans up the JSON string.
        """
        # Try various patterns to extract JSON
        patterns = [
            r'(?s)\s*```(?:json)?\s*(.*?)```',  # Markdown code block
            r'(?s)\s*(\{.*\})\s*$',  # JSON object at end
            r'(?s)\s*(\[.*\])\s*$',  # JSON array at end
            r'(?s)\s*"verified":',  # Look for JSON-like content
        ]
        
        json_content = None
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                json_content = match.group(1).strip() if pattern != r'(?s)\s*"verified":' else content.strip()
                break
        
        # If no pattern matched, use the entire content
        if json_content is None:
            json_content = content.strip()
        
        # Clean up escape sequences that might cause JSON parsing issues
        # Replace escaped backslashes with a temporary placeholder
        json_content = json_content.replace('\\\\', '___DOUBLE_BACKSLASH___')
        
        # Replace problematic escape sequences
        json_content = json_content.replace('\\(', '(')
        json_content = json_content.replace('\\)', ')')
        json_content = json_content.replace('\\[', '[')
        json_content = json_content.replace('\\]', ']')
        json_content = json_content.replace('\\"', '"')
        json_content = json_content.replace("\\'", "'")
        json_content = json_content.replace('\\n', '\n')
        json_content = json_content.replace('\\t', '\t')
        
        # Restore proper double backslashes
        json_content = json_content.replace('___DOUBLE_BACKSLASH___', '\\\\')
        
        # Handle any remaining invalid escape sequences
        json_content = re.sub(r'\\([^"\\/bfnrtu])', r'\1', json_content)
        
        return json_content

    def _normalize_verification_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and validate verification result.
        """
        result.setdefault("verified", False)
        result.setdefault("confidence", 0.0)
        result.setdefault("issues", ["Failed to determine issues with this step"])
        result.setdefault("corrected_step", None)
        result.setdefault("explanation", "No explanation provided")
        return result

    def _create_default_verification_result(self, message: str) -> Dict[str, Any]:
        """
        Create a default verification result with error message.
        """
        return {
            "verified": False,
            "confidence": 0.0,
            "issues": [message],
            "corrected_step": None,
            "explanation": "Verification failed"
        }

    def _apply_reasoning_fallback(self, state: Dict[str, Any], steps_key: str, verification_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply fallback mechanism when reasoning verification repeatedly fails.
        
        Args:
            state: The current state
            steps_key: Key for reasoning steps in state
            verification_data: Verification data collected so far
            
        Returns:
            Updated state with fallback reasoning
        """
        problem = state.get("question", "")
        correct_answer = state.get("correct_answer", "")
        
        # Identify which steps had issues
        problematic_steps = []
        for i, step_result in enumerate(verification_data["step_results"]):
            if not step_result.get("verified", False):
                problematic_steps.append(i)
        
        # Generate a complete fallback solution
        system_prompt = f"""
        You are an expert math tutor. The system has failed to generate a valid step-by-step solution after multiple attempts.
        
        Please provide a complete, correct step-by-step solution to the problem.
        
        Make sure each step is:
        1. Mathematically correct
        2. Clear and precise
        3. Logically connected to previous steps
        4. Directly relevant to solving the problem
        
        Format your response as a JSON array of steps, where each element is a string representing one logical step.
        """
        
        user_prompt = f"""
        Problem: {problem}
        Correct answer: {correct_answer}
        
        Generate a complete step-by-step solution:
        """
        
        try:
            response = self.llm_service.generate_completion(system_prompt, user_prompt)
            content = response.get("content", "")
            
            # Parse the JSON response
            try:
                fallback_steps = json.loads(content)
                if isinstance(fallback_steps, list) and all(isinstance(step, str) for step in fallback_steps):
                    # Replace the reasoning steps with fallback
                    state[steps_key] = fallback_steps
                    
                    # Mark as fallback in verification data
                    verification_data["used_fallback"] = True
                    verification_data["fallback_reason"] = f"Failed to verify steps after {verification_data['regeneration_attempts']} attempts"
                    
                    # Set overall verification to true since we're accepting the fallback
                    verification_data["overall_verified"] = True
                    verification_data["overall_confidence"] = 0.8  # Moderate confidence in fallback
                    
                    logger.info("Applied fallback reasoning steps")
                else:
                    logger.error(f"Fallback response not in expected format: {content}")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse fallback steps as JSON: {content}")
        except Exception as e:
            logger.error(f"Error generating fallback reasoning: {e}")
        
        return state

    def _get_nested_value(self, data: Dict[str, Any], key_path: str) -> Any:
        """
        Get a value from a nested dictionary using dot notation.
        
        Args:
            data: Dictionary to extract value from
            key_path: Path to the value using dot notation (e.g., "a.b.c")
            
        Returns:
            The value at the specified path, or None if not found
        """
        keys = key_path.split(".")
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
                
        return current

    def verify_solution(self, question: str, solution: str) -> Dict[str, Any]:
        """
        Verify a math solution and provide feedback.
        
        Args:
            question: The math question
            solution: The solution to verify
            
        Returns:
            Verification result
        """
        return self.verify_math_solution(question, solution)

    def verify_reasoning_steps_parallel(self, state, steps_key, max_workers=4):
        steps = self._get_steps_from_state(state, steps_key)
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all verification tasks and store futures
            future_to_step = {executor.submit(self._verify_step, step): (i, step) 
                             for i, step in enumerate(steps)}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_step):
                i, step = future_to_step[future]
                try:
                    result = future.result()
                    results.append((i, result))
                except Exception as e:
                    logger.error(f"Error verifying step {i}: {str(e)}")
                    results.append((i, self._create_default_verification_result(f"Verification error: {str(e)}")))
        
        # Sort results back into original order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
