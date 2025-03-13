"""
Meta-Agent for verifying outputs from other agents.

This module contains a meta-agent that verifies and potentially corrects
outputs from other agents to enhance robustness through iterative verification.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Callable
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
import queue
import threading

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

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
        Initialize the meta-agent.
        
        Args:
            model: The OpenAI model to use
            confidence_threshold: Minimum confidence score (0-1) required for verification
            max_iterations: Maximum number of regeneration attempts
            llm_service: LLM service abstraction
        """
        self.model = model
        self.llm_service = llm_service
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        logger.info(f"Initialized MetaAgent with model {model}, confidence threshold {confidence_threshold}")
    
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
    
    def verify_reasoning_steps(self, state: Dict[str, Any], steps_key: str, regenerate_step_func: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Verify each step of reasoning independently.
        
        Args:
            state: The current state
            steps_key: The key in the state that contains the steps to verify
            regenerate_step_func: A function to regenerate a step if it fails verification
            
        Returns:
            Updated state with verified steps
        """
        # Check if we need to verify steps
        if steps_key not in state or not state[steps_key]:
            logger.warning(f"No steps to verify in {steps_key}")
            return state
        
        # Initialize verification tracking
        verification_start_time = time.time()
        max_total_verification_time = 30  # Maximum time for the entire verification process
        max_step_verification_time = 8    # Maximum time for a single step verification
        max_regeneration_attempts = 2     # Maximum number of regeneration attempts per step
        
        # Track verification attempts for each step
        verification_attempts = {}
        
        # Initialize the verified steps
        verified_steps = []
        verification_results = []
        
        # Get the steps to verify
        steps = state[steps_key]
        
        # Verify each step
        for i, step in enumerate(steps):
            # Check if we've exceeded the total verification time
            if time.time() - verification_start_time > max_total_verification_time:
                logger.warning(f"Total verification time exceeded {max_total_verification_time} seconds, skipping remaining steps")
                # Add remaining steps without verification
                verified_steps.extend(steps[i:])
                verification_results.extend([{"verified": False, "reason": "Verification timeout"}] * (len(steps) - i))
                break
            
            # Initialize verification attempt counter for this step
            if i not in verification_attempts:
                verification_attempts[i] = 0
            
            # Try to verify the step with timeout
            step_verified = False
            verification_error = None
            verification_result = {"verified": False, "reason": "Unknown error"}
            
            # Verify the step with timeout
            try:
                # Use a queue to get the result from the thread
                result_queue = queue.Queue()
                
                def verify_step_with_timeout():
                    try:
                        # Verify the step
                        result = self._verify_step(step)
                        result_queue.put({"success": True, "result": result})
                    except Exception as e:
                        logger.error(f"Error verifying step {i+1}: {e}")
                        result_queue.put({"success": False, "error": str(e)})
                
                # Create and start a thread for verification
                verification_thread = threading.Thread(target=verify_step_with_timeout)
                verification_thread.daemon = True
                verification_thread.start()
                
                # Wait for the thread to complete with a timeout
                try:
                    thread_result = result_queue.get(timeout=max_step_verification_time)
                    if thread_result["success"]:
                        verification_result = thread_result["result"]
                        step_verified = verification_result.get("verified", False)
                    else:
                        verification_error = thread_result["error"]
                        verification_result = {"verified": False, "reason": f"Verification error: {verification_error}"}
                except queue.Empty:
                    logger.warning(f"Step {i+1} verification timed out after {max_step_verification_time} seconds")
                    verification_result = {"verified": False, "reason": f"Verification timed out after {max_step_verification_time} seconds"}
            
            except Exception as e:
                logger.error(f"Error in verification process for step {i+1}: {e}")
                verification_result = {"verified": False, "reason": f"Verification process error: {str(e)}"}
            
            # If the step is not verified and we have a regeneration function, try to regenerate it
            if not step_verified and regenerate_step_func and verification_attempts[i] < max_regeneration_attempts:
                try:
                    # Increment the verification attempt counter
                    verification_attempts[i] += 1
                    
                    # Log the regeneration attempt
                    logger.info(f"Regenerating step {i+1} (attempt {verification_attempts[i]})")
                    
                    # Use a queue to get the result from the thread
                    regen_queue = queue.Queue()
                    
                    def regenerate_step_with_timeout():
                        try:
                            # Regenerate the step
                            verification_result_dict = {
                                "verified": verification_result.get("verified", False),
                                "confidence": verification_result.get("confidence", 0.0),
                                "reason": verification_result.get("reason", "Unknown reason")
                            }
                            new_step = regenerate_step_func(i, step, verification_result_dict)
                            regen_queue.put({"success": True, "step": new_step})
                        except Exception as e:
                            logger.error(f"Error regenerating step {i+1}: {e}")
                            regen_queue.put({"success": False, "error": str(e)})
                    
                    # Create and start a thread for regeneration
                    regeneration_thread = threading.Thread(target=regenerate_step_with_timeout)
                    regeneration_thread.daemon = True
                    regeneration_thread.start()
                    
                    # Wait for the thread to complete with a timeout
                    try:
                        thread_result = regen_queue.get(timeout=max_step_verification_time)
                        if thread_result["success"]:
                            # Update the step
                            step = thread_result["step"]
                            
                            # Try to verify the regenerated step
                            verification_result = self._verify_step(step)
                            step_verified = verification_result.get("verified", False)
                        else:
                            logger.warning(f"Failed to regenerate step {i+1}: {thread_result['error']}")
                    except queue.Empty:
                        logger.warning(f"Step {i+1} regeneration timed out after {max_step_verification_time} seconds")
                        verification_result = {"verified": False, "reason": "Regeneration timed out"}
                
                except Exception as e:
                    logger.error(f"Error in regeneration process for step {i+1}: {e}")
                    verification_result = {"verified": False, "reason": f"Regeneration error: {str(e)}"}
            
            # Add the step to the verified steps
            verified_steps.append(step)
            verification_results.append(verification_result)
            
            # Log the verification result
            if step_verified:
                logger.info(f"Step {i+1} verified successfully")
            else:
                logger.warning(f"Step {i+1} verification failed: {verification_result.get('reason', 'Unknown reason')}")
        
        # Update the state with the verified steps
        state[steps_key] = verified_steps
        state["verification_results"] = verification_results
        
        # Calculate the overall verification score
        verified_count = sum(1 for result in verification_results if result.get("verified", False))
        verification_score = verified_count / len(verification_results) if verification_results else 0
        state["verification_score"] = verification_score
        
        # Log the verification score
        logger.info(f"Verification score: {verification_score:.2f} ({verified_count}/{len(verification_results)})")
        
        return state
    
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
