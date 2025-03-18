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
import re

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
            state: Current math problem state
            
        Returns:
            Updated state with reasoning steps
        """
        logger.info("Beginning reasoning generation")
        self.record_event(state, "reasoning_start", {
            "question": state.question, 
            "student_answer": state.student_answer
        })
        
        # Check if reasoning mode is specified in context
        reasoning_mode = state.context.get('reasoning_mode', 'dot') if state.context else 'dot'
        
        try:
            # Generate the steps
            full_steps = self._generate_reasoning_steps(
                state.question, 
                state.student_answer,
                state.correct_answer
            )
            
            logger.info(f"Successfully generated {len(full_steps)} reasoning steps")
            self.record_event(state, "reasoning_complete", {
                "step_count": len(full_steps)
            })
            
            # Store full steps for CoT (expander view)
            if not state.context:
                state.context = {}
            state.context['full_reasoning_steps'] = full_steps
            
            # For DoT mode, create concise steps for default display
            if reasoning_mode == 'dot':
                concise_steps = self._generate_concise_reasoning(full_steps)
                state.steps = concise_steps
                state.context['dot_reasoning_steps'] = concise_steps
            else:
                # For CoT mode, use full steps
                state.steps = full_steps
            
            # Verify reasoning with meta agent if available
            if self.meta_agent:
                logger.info("Verifying reasoning steps with meta agent")
                verified_state = self.meta_agent.verify_reasoning_steps(state, 'steps')
                
                # Record verification metrics
                self.record_event(state, "reasoning_verification_complete", {
                    "verified_count": verified_state.context.get('verified_steps_count', 0) if verified_state.context else 0,
                    "regenerated_count": verified_state.context.get('regenerated_steps_count', 0) if verified_state.context else 0
                })
                
                # Use verified steps
                if reasoning_mode == 'dot':
                    # If in DoT mode, regenerate concise steps from verified full steps
                    if 'full_reasoning_steps' in verified_state.context:
                        concise_steps = self._generate_concise_reasoning(verified_state.context['full_reasoning_steps'])
                        verified_state.steps = concise_steps
                        verified_state.context['dot_reasoning_steps'] = concise_steps
                
                return verified_state
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {str(e)}")
            self.log_error(e, state)
            
            # Add minimal fallback explanation
            state.steps = ["Error generating detailed reasoning. Please try again or contact support."]
            return state
    
    def _generate_reasoning_steps(self, question: str, student_answer: str, correct_answer: str = None) -> List[str]:
        """
        Generate step-by-step reasoning for solving the problem.
        
        Args:
            question: The math question
            student_answer: The student's answer
            correct_answer: The known correct answer (if available)
            
        Returns:
            List of reasoning steps
        """
        try:
            # Create prompt
            if correct_answer:
                prompt = f"""
Please provide detailed step-by-step reasoning for solving this math problem:

Problem: {question}

The correct answer is: {correct_answer}

Generate a clear, detailed sequence of steps to solve this problem, explaining each step thoroughly.
Each step should be clearly labeled like "Step 1: ...", "Step 2: ...", etc.

Ensure your solution is accurate, follows sound mathematical principles, and reaches the correct answer.
"""
            else:
                prompt = f"""
Please provide detailed step-by-step reasoning for solving this math problem:

Problem: {question}

Generate a clear, detailed sequence of steps to solve this problem, explaining each step thoroughly.
Each step should be clearly labeled like "Step 1: ...", "Step 2: ...", etc.

Ensure your solution is accurate and follows sound mathematical principles.
"""

            # Get response from LLM
            logger.info("Generating reasoning steps")
            response = self.llm_service.generate_completion(
                system_message="You are a mathematics tutor explaining the step-by-step solution to a math problem.",
                user_prompt=prompt
            )
            
            # Extract steps from text
            steps = self._extract_steps_from_text(response["content"])
            
            if not steps:
                # Fallback if no steps could be extracted
                logger.warning("Failed to extract steps, using whole text as a single step")
                steps = [response["content"]]
            else:
                logger.info(f"Successfully extracted {len(steps)} reasoning steps from text")
                
            return steps
        
        except Exception as e:
            logger.error(f"Error generating reasoning steps: {str(e)}")
            return ["Error generating reasoning steps. The system encountered an issue while processing this problem."]

    def _generate_concise_reasoning(self, steps: List[str], max_steps: int = 2) -> List[str]:
        """
        Generate a more concise version of the reasoning steps for DoT display.
        
        Args:
            steps: The full list of reasoning steps
            max_steps: Maximum number of steps to include (default: 2 for very concise)
            
        Returns:
            List of condensed reasoning steps
        """
        if not steps:
            return steps
            
        if len(steps) <= max_steps:
            # Even if we have few steps, still condense them
            return [self._condense_step(step) for step in steps]
            
        # Always include first and last steps for very concise representation
        selected_indices = [0]
        
        # Add the final step
        if len(steps) > 1:
            selected_indices.append(len(steps) - 1)
            
        # If we want more than 2 steps and there are middle steps available
        if max_steps > 2 and len(steps) > 2:
            # Calculate how many middle steps to include
            middle_steps_count = min(max_steps - 2, len(steps) - 2)
            
            if middle_steps_count > 0:
                # Select evenly spaced middle steps
                step_interval = (len(steps) - 2) / (middle_steps_count + 1)
                
                for i in range(1, middle_steps_count + 1):
                    # Calculate index for this middle step
                    idx = min(int(i * step_interval), len(steps) - 2)
                    selected_indices.append(idx)
        
        # Sort indices to maintain order
        selected_indices.sort()
        
        # Create concise steps
        concise_steps = []
        for i, idx in enumerate(selected_indices):
            step = steps[idx]
            
            # Clean up step formatting with more aggressive condensing
            step = self._condense_step(step, more_aggressive=True)
            
            # Add transition text if steps were skipped
            if i > 0 and selected_indices[i] - selected_indices[i-1] > 1:
                skipped = selected_indices[i] - selected_indices[i-1] - 1
                transition = f"[{skipped} step{'s' if skipped > 1 else ''} omitted] "
                step = transition + step
                
            concise_steps.append(step)
            
        return concise_steps
        
    def _condense_step(self, step: str, more_aggressive: bool = False) -> str:
        """
        Condense a reasoning step to make it more concise.
        
        Args:
            step: The reasoning step to condense
            more_aggressive: Whether to condense more aggressively (for DoT)
            
        Returns:
            The condensed step
        """
        # Remove step prefix if present (will be added back in the display)
        step = re.sub(r'^Step \d+:\s*', '', step)
        step = re.sub(r'^\*\*Step \d+:\*\*\s*', '', step)
        
        # Remove excessive markdown formatting
        step = re.sub(r'\*\*|\*', '', step)
        
        # Remove excessive whitespace
        step = re.sub(r'\s+', ' ', step).strip()
        
        if more_aggressive:
            # Remove explanatory phrases that don't add core math content
            step = re.sub(r'we can see that |it is clear that |we know that |note that |observe that ', '', step, flags=re.IGNORECASE)
            
            # Remove phrases like "first, we need to" or "next, let's"
            step = re.sub(r'(first|next|then|finally|now),\s+(we|let\'s|I)(\'ll|\s+will|\s+need\s+to|\s+can|\s+should)?\s+', '', step, flags=re.IGNORECASE)
            
            # Focus on just the key calculation or transformation
            if "=" in step:
                # Try to extract core equation parts
                parts = step.split("=")
                if len(parts) > 2:
                    # For multi-part equations, keep first and last parts
                    key_parts = [parts[0], parts[-1]]
                    step = " = ".join(key_parts)
            
            # Truncate much more aggressively 
            max_length = 150  # Even shorter for DoT
            if len(step) > max_length:
                # Try to find a good sentence boundary to truncate at
                sentence_end = step.find('. ', 50, 130)
                if sentence_end > 0:
                    step = step[:sentence_end+1]
                else:
                    step = step[:max_length] + "..."
        else:
            # Standard condensing for CoT
            if len(step) > 200:
                # Try to find a good sentence boundary to truncate at
                sentence_end = step.find('. ', 100, 180)
                if sentence_end > 0:
                    step = step[:sentence_end+1]
                else:
                    step = step[:180] + "..."
                
        return step

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

    def _extract_steps_from_text(self, text: str) -> List[str]:
        """
        Extract reasoning steps from the text response.
        
        Args:
            text: Response text from the language model
            
        Returns:
            List of extracted reasoning steps
        """
        # Try to find steps by looking for "Step X:" or numbered points
        steps = []
        
        # Pattern to match step headers like "Step 1:", "1.", "1)", etc.
        step_patterns = [
            r'Step\s+\d+\s*:',  # Step 1:
            r'^\d+\s*\.\s+',    # 1. 
            r'^\d+\s*\)\s+',    # 1) 
            r'^\*\*Step\s+\d+\s*:\*\*',  # **Step 1:**
        ]
        
        # Split the text into lines
        lines = text.split('\n')
        
        current_step = []
        in_step = False
        
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Check if this line starts a new step
            is_step_header = False
            for pattern in step_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_step_header = True
                    break
                    
            if is_step_header:
                # Save the previous step if we were processing one
                if in_step and current_step:
                    steps.append('\n'.join(current_step))
                    
                # Start a new step
                current_step = [line]
                in_step = True
            elif in_step:
                # Continue the current step
                current_step.append(line)
        
        # Add the last step if there is one
        if in_step and current_step:
            steps.append('\n'.join(current_step))
            
        # If no steps were found using the patterns, try to split by sentence
        if not steps:
            # Simple heuristic: split by sentences and group into logical chunks
            logger.warning("No step pattern found, attempting to split by sentences")
            
            # Simple sentence splitting (this is a basic approach)
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Group sentences into logical steps (roughly 2-3 sentences per step)
            if sentences:
                chunk_size = max(1, len(sentences) // 5)  # Aim for about 5 steps
                for i in range(0, len(sentences), chunk_size):
                    chunk = sentences[i:i+chunk_size]
                    step_text = f"Step {len(steps)+1}: " + ' '.join(chunk)
                    steps.append(step_text)
        
        # If still no steps, use the whole text as a single step
        if not steps:
            logger.warning("Could not extract steps, using whole text as one step")
            steps = [f"Step 1: {text.strip()}"]
            
        return steps
