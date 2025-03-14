"""
Math Reasoning Command.

This module contains the command for generating step-by-step reasoning
for math problems with verification of each step.
"""

import logging
import json
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.math.services.llm.base_service import BaseLLMService

logger = logging.getLogger(__name__)

class MathGenerateReasoningCommand:
    """Command for generating verified step-by-step reasoning for math problems."""
    
    def __init__(self, agent):
        """
        Initialize the reasoning command.
        
        Args:
            agent: The agent instance that will use this command
        """
        self.agent = agent
        self.llm_service = agent.llm_service
        self.meta_agent = agent.meta_agent
        logger.info("Initialized MathGenerateReasoningCommand")

    def _execute_core(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate step-by-step reasoning for the problem.
        
        Args:
            state: Current state with problem and student answer
            
        Returns:
            Updated state with reasoning steps
        """
        logger.info("Generating step-by-step reasoning")
        
        # Get the problem and student answer from the state
        problem = state.get("question", "")
        student_answer = state.get("student_answer", "")
        correct_answer = state.get("correct_answer", "")
        
        # Generate the reasoning steps
        reasoning_steps = self._generate_reasoning_steps(problem, student_answer, correct_answer)
        
        # Add the reasoning steps to the state
        state["reasoning_steps"] = reasoning_steps
        
        # Verify the reasoning steps using the meta-agent if available
        if self.meta_agent:
            logger.info("Verifying reasoning steps with meta-agent")
            
            # Define a regeneration function for the meta-agent to use
            def regenerate_step(current_state):
                # Get the verification feedback and current step index
                feedback = current_state.get("step_verification_feedback", [])
                confidence = current_state.get("step_verification_confidence", 0)
                step_index = current_state.get("current_step_index", 0)
                
                logger.info(f"Regenerating step {step_index} with feedback: {feedback}, confidence: {confidence}")
                
                # Get the current reasoning steps
                steps = current_state.get("reasoning_steps", [])
                
                # Generate a new step with the feedback
                new_step = self._regenerate_single_step(
                    problem, 
                    student_answer, 
                    correct_answer,
                    step_index,
                    steps[:step_index],  # Previous steps
                    steps[step_index+1:] if step_index < len(steps) - 1 else [],  # Subsequent steps
                    verification_feedback=feedback,
                    previous_confidence=confidence
                )
                
                # Replace the problematic step
                if step_index < len(steps):
                    steps[step_index] = new_step
                
                # Update the state
                current_state["reasoning_steps"] = steps
                
                return current_state
            
            # Verify with the ability to regenerate
            state = self.meta_agent.verify_reasoning_steps(state, "reasoning_steps", regenerate_step)
        
        # Record success event
        return self._record_event(state, {"status": "success"})

    def _generate_reasoning_steps(self, problem: str, student_answer: str, correct_answer: str) -> List[str]:
        """
        Generate step-by-step reasoning for the problem.
        
        Args:
            problem: The math problem
            student_answer: The student's answer
            correct_answer: The correct answer
            
        Returns:
            List of reasoning steps
        """
        try:
            # Use LLM to generate reasoning steps
            if self.llm_service:
                # Create the system prompt
                system_prompt = """
                You are an expert math tutor providing step-by-step reasoning for a math problem.
                Break down the solution into clear, logical steps that a student can follow.
                
                Each step should:
                1. Be mathematically correct and precise
                2. Focus on a single logical operation or concept
                3. Build on previous steps
                4. Be clear and educational
                
                Format your response as a JSON array of strings, where each string is a single step in the solution.
                """
                
                # Construct the user prompt
                user_prompt = f"""
                Problem: {problem}
                Student answer: {student_answer}
                Correct answer: {correct_answer}
                
                Generate a step-by-step solution showing the reasoning process:
                """
                
                # Generate the reasoning steps
                response = self.llm_service.generate_completion(system_prompt, user_prompt)
                content = response.get("content", "")
                
                try:
                    # Parse the JSON response
                    reasoning_steps = json.loads(content)
                    
                    # Validate that we got a list of strings
                    if isinstance(reasoning_steps, list) and all(isinstance(step, str) for step in reasoning_steps):
                        return reasoning_steps
                    else:
                        logger.error(f"Invalid reasoning steps format: {content}")
                        return ["Error: Unable to generate proper reasoning steps."]
                        
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse reasoning steps as JSON: {content}")
                    # Extract steps from non-JSON response as fallback
                    steps = [line.strip() for line in content.split('\n') if line.strip()]
                    return steps
                
            else:
                return ["I'm sorry, I can't provide reasoning steps at this time."]
        except Exception as e:
            logger.error(f"Error generating reasoning steps: {e}")
            logger.error(traceback.format_exc())
            return ["I'm sorry, I encountered an error generating reasoning steps."]

    def _regenerate_single_step(self, problem: str, student_answer: str, correct_answer: str, 
                               step_index: int, previous_steps: List[str], subsequent_steps: List[str],
                               verification_feedback: List[str] = None, 
                               previous_confidence: float = None) -> str:
        """
        Regenerate a single reasoning step based on verification feedback.
        
        Args:
            problem: The math problem
            student_answer: The student's answer
            correct_answer: The correct answer
            step_index: Index of the step to regenerate
            previous_steps: Steps that come before the current step
            subsequent_steps: Steps that come after the current step
            verification_feedback: Feedback from verification
            previous_confidence: Confidence score from verification
            
        Returns:
            Regenerated step
        """
        try:
            # Use LLM to regenerate the step
            if self.llm_service:
                # Create the system prompt
                system_prompt = """
                You are an expert math tutor fixing a specific step in a mathematical solution.
                Your task is to regenerate a single step that was identified as problematic.
                
                The new step should:
                1. Be mathematically correct and precise
                2. Address all the issues identified in the verification feedback
                3. Maintain logical flow with previous and subsequent steps
                4. Be clear and educational
                
                Provide ONLY the corrected step as plain text, not as JSON.
                """
                
                # Add verification feedback if available
                if verification_feedback:
                    system_prompt += "\n\nThe previous step had these issues that need to be addressed:\n"
                    for i, feedback in enumerate(verification_feedback, 1):
                        system_prompt += f"{i}. {feedback}\n"
                    
                    if previous_confidence is not None:
                        system_prompt += f"\nThe verification confidence was only {previous_confidence:.2f}. Please improve the quality and correctness of this step."
                
                # Construct the user prompt
                user_prompt = f"""
                Problem: {problem}
                Student answer: {student_answer}
                Correct answer: {correct_answer}
                Step to regenerate: #{step_index + 1}
                
                Previous steps:
                """
                
                for i, prev_step in enumerate(previous_steps, 1):
                    user_prompt += f"Step {i}: {prev_step}\n"
                
                if subsequent_steps:
                    user_prompt += "\nSubsequent steps:\n"
                    for i, next_step in enumerate(subsequent_steps, len(previous_steps) + 2):
                        user_prompt += f"Step {i}: {next_step}\n"
                
                user_prompt += "\nGenerate the corrected step:"
                
                # Generate the corrected step
                response = self.llm_service.generate_completion(system_prompt, user_prompt)
                corrected_step = response.get("content", "").strip()
                
                return corrected_step
            else:
                return "I'm sorry, I can't regenerate this step at this time."
        except Exception as e:
            logger.error(f"Error regenerating step: {e}")
            logger.error(traceback.format_exc())
            return "I'm sorry, I encountered an error regenerating this step."

    def _regenerate_step_with_limit(self, step_index: int, step: str, verification_result: Dict[str, Any], confidence: float) -> str:
        """
        Regenerate a reasoning step with confidence limit.
        
        Args:
            step_index: Index of the step
            step: Current step content
            verification_result: Verification result dictionary
            confidence: Confidence score from verification
        """
        try:
            if confidence < self.confidence_threshold:
                return self._regenerate_single_step(step_index, step, verification_result, confidence)
            return step
        except Exception as e:
            logger.error(f"Error regenerating step {step_index+1}: {e}")
            raise

    def _record_event(self, state: Dict[str, Any], event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record an event for the reasoning generation.
        
        Args:
            state: The current state
            event_data: Data about the event
            
        Returns:
            The updated state
        """
        logger.info(f"Reasoning command event: {event_data}")
        
        # Add event to state if tracking events
        if "events" not in state:
            state["events"] = []
        
        event = {
            "type": "reasoning_generated",
            "timestamp": str(datetime.now()),
            "data": event_data
        }
        
        state["events"].append(event)
        
        return state
