"""
Math Reasoning Command.

This module contains the command for generating step-by-step reasoning
for math problem solutions.
"""

import logging
import json
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.math_services.commands.base_command import BaseCommand
from app.math_services.models.state import MathState
from app.math_services.services.service_container import ServiceContainer

logger = logging.getLogger(__name__)

class MathGenerateReasoningCommand(BaseCommand):
    """Command for generating step-by-step reasoning for math problem solutions."""
    
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
        Generate step-by-step reasoning for a math problem.
        
        Args:
            state: Current MathState
            
        Returns:
            Updated MathState with reasoning steps
        """
        # Log the start of execution
        logger.info("Beginning reasoning generation")
        self.record_event(state, "reasoning_start", {
            "question": state.question,
            "student_answer": state.student_answer
        })
        
        # Check if we have the required inputs
        if not state.question:
            logger.error("No question found in state")
            return state
            
        if not state.student_answer:
            logger.error("No student answer found in state")
            return state
        
        try:
            # Generate reasoning steps
            steps = self._generate_reasoning_steps(
                state.question, 
                state.student_answer, 
                state.correct_answer
            )
            
            # Store the steps in the state
            if "reasoning" not in state.context:
                state.context["reasoning"] = {}
                
            state.context["reasoning"]["steps"] = steps
            
            # Also store directly as a top-level attribute for meta agent verification
            # This is needed because the meta_agent expects steps at this location
            state.steps = steps
            
            # Log successful completion
            logger.info(f"Successfully generated {len(steps)} reasoning steps")
            self.record_event(state, "reasoning_complete", {
                "step_count": len(steps)
            })
            
            # If meta agent is available, verify the reasoning steps
            if self.meta_agent:
                try:
                    # Use a regenerate_step function that will be passed to the meta agent
                    def regenerate_step(step_index, feedback):
                        return self._regenerate_step(
                            state.question,
                            state.student_answer,
                            state.correct_answer,
                            step_index,
                            steps[:step_index],
                            steps[step_index+1:],
                            verification_feedback=[feedback],
                        )
                    
                    # Call the meta agent to verify the steps
                    logger.info("Verifying reasoning steps with meta agent")
                    verification_result = self.meta_agent.verify_reasoning_steps(
                        state=state.to_dict() if hasattr(state, 'to_dict') else state,
                        steps_key="steps",
                        regenerate_step_func=regenerate_step
                    )
                    
                    # Store the verification results
                    state.context["reasoning"]["verification"] = verification_result
                    
                    # Log verification result
                    logger.info(f"Reasoning verification complete: {len(verification_result.get('verified_steps', []))} steps verified")
                    self.record_event(state, "reasoning_verification_complete", {
                        "verified_count": len(verification_result.get("verified_steps", [])),
                        "regenerated_count": len(verification_result.get("regenerated_steps", []))
                    })
                    
                except Exception as verify_error:
                    logger.error(f"Error verifying reasoning steps: {str(verify_error)}")
                    state.context["reasoning"]["verification_error"] = str(verify_error)
                    self.log_error(verify_error, state)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in reasoning generation: {str(e)}")
            self.log_error(e, state)
            
            # Set a fallback message
            if "reasoning" not in state.context:
                state.context["reasoning"] = {}
                
            state.context["reasoning"]["error"] = str(e)
            state.context["reasoning"]["steps"] = ["Error generating reasoning steps."]
            
            return state
    
    def _generate_reasoning_steps(self, question: str, student_answer: str, correct_answer: str = None) -> List[str]:
        """
        Generate step-by-step reasoning for a math problem.
        
        Args:
            question: The math question
            student_answer: The student's answer
            correct_answer: The correct answer (if available)
            
        Returns:
            List of reasoning steps
        """
        logger.info("Generating reasoning steps")
        
        # Create the system prompt
        system_prompt = """
        You are an expert math tutor. Break down the solution to the math problem into 
        clear, sequential steps that explain the reasoning. Focus on being educational
        and showing the complete logical flow.
        
        For each step:
        1. Be concise but complete
        2. Show all work clearly
        3. Explain the mathematical concepts being applied
        
        Return your reasoning as a JSON array of steps, where each step is a string.
        Example: ["Step 1: Identify the equation type...", "Step 2: Isolate the variable..."]
        """
        
        # Build the user prompt with the available information
        user_prompt = f"""
        Problem: {question}
        Student's answer: {student_answer}
        """
        
        # Add the correct answer if available
        if correct_answer:
            user_prompt += f"Correct answer: {correct_answer}\n"
        
        user_prompt += """
        Please provide a step-by-step solution with clear reasoning.
        Format your response as a JSON array of strings.
        """
        
        # Call the LLM service to generate the reasoning
        response = self.llm_service.generate_completion(
            system_prompt,
            user_prompt
        )
        
        # Extract the steps from the response
        content = response.get("content", "")
        
        try:
            # Try to parse as JSON directly
            steps = json.loads(content)
            if not isinstance(steps, list):
                raise ValueError("Response is not a list of steps")
                
            logger.info(f"Successfully parsed {len(steps)} reasoning steps")
            return steps
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract from text
            logger.warning("Failed to parse response as JSON, attempting extraction")
            
            # Look for JSON array in the response
            import re
            match = re.search(r'\[.*\]', content, re.DOTALL)
            
            if match:
                try:
                    steps = json.loads(match.group(0))
                    logger.info(f"Successfully extracted {len(steps)} reasoning steps from text")
                    return steps
                except:
                    pass
            
            # If JSON extraction fails, split by numbered steps
            logger.warning("JSON extraction failed, splitting by numbered steps")
            lines = content.strip().split('\n')
            steps = []
            
            current_step = ""
            for line in lines:
                if re.match(r'^Step\s+\d+:|^\d+\.', line):
                    if current_step:
                        steps.append(current_step.strip())
                    current_step = line
                else:
                    current_step += " " + line
            
            if current_step:
                steps.append(current_step.strip())
            
            # If no steps were found, use the entire content as a single step
            if not steps:
                logger.warning("No steps found, using entire content as a single step")
                steps = [content.strip()]
            
            logger.info(f"Split response into {len(steps)} reasoning steps")
            return steps
    
    def _regenerate_step(self, problem: str, student_answer: str, correct_answer: str,
                        step_index: int, previous_steps: List[str], subsequent_steps: List[str],
                        verification_feedback: List[str] = None, 
                        previous_confidence: float = None) -> str:
        """
        Regenerate a specific reasoning step based on feedback.
        
        Args:
            problem: The math problem
            student_answer: The student's answer
            correct_answer: The correct answer
            step_index: Index of the step to regenerate
            previous_steps: Steps before the one being regenerated
            subsequent_steps: Steps after the one being regenerated
            verification_feedback: Feedback on why the step needs regeneration
            previous_confidence: Confidence score from previous verification
            
        Returns:
            Regenerated step
        """
        logger.info(f"Regenerating step {step_index} with feedback")
        
        # Create the system prompt for regeneration
        system_prompt = """
        You are an expert math tutor. Your task is to improve a specific step in a 
        step-by-step solution based on the feedback provided.
        
        The step should:
        1. Fix any errors identified in the feedback
        2. Maintain logical flow with previous and subsequent steps
        3. Be clear and educational
        4. Retain the mathematical concepts of the original step
        
        Return only the corrected step as a string, without any additional explanation.
        """
        
        # Format feedback into a string
        feedback_str = "None"
        if verification_feedback and len(verification_feedback) > 0:
            feedback_str = "\n".join(verification_feedback)
        
        # Build the prompt with context
        user_prompt = f"""
        Problem: {problem}
        Student's answer: {student_answer}
        Correct answer: {correct_answer if correct_answer else "Not provided"}
        
        Previous steps:
        {"\n".join(previous_steps) if previous_steps else "None"}
        
        Current step to fix (Step {step_index + 1}):
        {previous_steps[-1] if step_index > 0 and previous_steps else "Start from the beginning"}
        
        Subsequent steps:
        {"\n".join(subsequent_steps) if subsequent_steps else "None"}
        
        Feedback on the current step:
        {feedback_str}
        
        Previous confidence: {previous_confidence if previous_confidence is not None else "Not available"}
        
        Please provide a corrected version of Step {step_index + 1} that addresses the feedback.
        """
        
        # Call the LLM service to regenerate the step
        response = self.llm_service.generate_completion(
            system_prompt,
            user_prompt
        )
        
        # Extract the regenerated step
        regenerated_step = response.get("content", "").strip()
        
        logger.info(f"Regenerated step {step_index}: {regenerated_step[:50]}...")
        return regenerated_step
