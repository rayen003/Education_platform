"""
Math Feedback Command.

This module contains the command for generating feedback on student
answers to math problems.
"""

import logging
import json
import traceback
import re
from typing import Dict, Any, List, Optional
import sympy
from sympy import symbols, sympify, simplify
from datetime import datetime

from app.math.services.llm.base_service import BaseLLMService

logger = logging.getLogger(__name__)

class MathGenerateFeedbackCommand:
    """Command for generating feedback on student answers to math problems."""
    
    def __init__(self, agent):
        """
        Initialize the feedback command.
        
        Args:
            agent: The agent instance that will use this command
        """
        self.agent = agent
        self.llm_service = agent.llm_service
        self.meta_agent = agent.meta_agent
        logger.info("Initialized MathGenerateFeedbackCommand")
    
    def _get_llm_service(self):
        """
        Get the LLM service from the agent if available.
        
        Returns:
            LLM service or None
        """
        if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'llm_service'):
            return self.agent.llm_service
        return None
    
    def _record_event(self, state: Dict[str, Any], event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record an event for the feedback generation.
        
        Args:
            state: The current state
            event_data: Data about the event
            
        Returns:
            The updated state
        """
        logger.info(f"Feedback command event: {event_data}")
        
        # Add event to state if tracking events
        if "events" not in state:
            state["events"] = []
        
        event = {
            "type": "feedback_generated",
            "timestamp": str(datetime.now()),
            "data": event_data
        }
        
        state["events"].append(event)
        
        return state
        
    def _generate_solution_explanation(self, question: str, correct_answer: str) -> str:
        """
        Generate a detailed explanation of the solution for the final attempt.
        
        Args:
            question: The math question
            correct_answer: The correct answer
            
        Returns:
            HTML formatted solution explanation
        """
        # For NPV calculation
        if "Net Present Value" in question or "NPV" in question:
            explanation = """
            <p><strong>Solution Approach:</strong></p>
            <p>To calculate the Net Present Value (NPV), we need to:</p>
            <ol>
                <li>Identify the initial investment, cash flows, and discount rate</li>
                <li>Calculate the present value of each future cash flow</li>
                <li>Sum up all present values and subtract the initial investment</li>
            </ol>
            
            <p><strong>Given:</strong></p>
            <ul>
                <li>Initial investment: $10,000</li>
                <li>Annual cash flows: $2,500 for 5 years</li>
                <li>Discount rate: 8%</li>
            </ul>
            
            <p><strong>Calculation:</strong></p>
            <p>NPV = -Initial Investment + Σ(Cash Flow_t / (1 + r)^t)</p>
            <p>NPV = -$10,000 + $2,500/(1.08)^1 + $2,500/(1.08)^2 + $2,500/(1.08)^3 + $2,500/(1.08)^4 + $2,500/(1.08)^5</p>
            <p>NPV = -$10,000 + $2,314.81 + $2,143.35 + $1,984.58 + $1,837.57 + $1,701.46</p>
            <p>NPV = -$10,000 + $9,981.77</p>
            <p>NPV = -$18.23</p>
            
            <p>Since the NPV is negative, this investment would not be profitable at the given discount rate.</p>
            """
            return explanation
        else:
            # Generic explanation for other math problems
            return f"""
            <p><strong>Solution:</strong></p>
            <p>The correct answer is: {correct_answer}</p>
            <p>To solve this problem, you need to carefully analyze the question, identify the relevant formulas, 
            and apply the appropriate mathematical techniques.</p>
            """
    
    def _execute_core(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate feedback for the student's answer.
        
        Args:
            state: Current state with problem and student answer
            
        Returns:
            Updated state with feedback
        """
        logger.info("Generating feedback")
        
        # Get the problem and student answer from the state
        problem = state.get("question", "")
        student_answer = state.get("student_answer", "")
        correct_answer = state.get("correct_answer", "")
        
        # Check if we have the necessary information
        if not problem or not student_answer or not correct_answer:
            logger.warning("Missing required information for feedback generation")
            return self._record_event(state, {"status": "error", "message": "Missing required information"})
        
        # Check if the student has reached the maximum number of attempts
        attempts = state.get("attempts", 1)
        max_attempts = state.get("max_attempts", 3)
        
        # Check if the answer is correct with more flexibility
        is_correct = self._check_answer_with_flexibility(student_answer, correct_answer)
        state["is_correct"] = is_correct
        
        # Generate feedback
        feedback = self._generate_feedback(problem, student_answer, correct_answer, is_correct, attempts, max_attempts)
        
        # Add the feedback to the state
        if "feedback" not in state:
            state["feedback"] = []
        state["feedback"].append(feedback)
        
        # Verify the feedback using the meta-agent if available
        if self.meta_agent:
            logger.info("Verifying feedback with meta-agent")
            
            # Define a regeneration function for the meta-agent to use
            def regenerate_feedback(current_state):
                # Get the verification feedback
                verification_feedback = current_state.get("verification_feedback", [])
                confidence = current_state.get("verification_confidence", 0)
                
                logger.info(f"Regenerating feedback with verification: {verification_feedback}, confidence: {confidence}")
                
                # Remove the last feedback that failed verification
                if current_state.get("feedback"):
                    current_state["feedback"].pop()
                
                # Generate new feedback with the verification feedback
                new_feedback = self._generate_feedback(
                    problem, 
                    student_answer, 
                    correct_answer, 
                    is_correct, 
                    attempts, 
                    max_attempts,
                    verification_feedback=verification_feedback,
                    previous_confidence=confidence
                )
                
                # Add the new feedback to the state
                if "feedback" not in current_state:
                    current_state["feedback"] = []
                current_state["feedback"].append(new_feedback)
                
                return current_state
            
            # Verify with the ability to regenerate
            state = self.meta_agent.verify_output(state, "feedback", regenerate_feedback)
        
        # Record success event
        return self._record_event(state, {"status": "success"})
        
    def _generate_feedback(self, problem: str, student_answer: str, correct_answer: str, 
                          is_correct: bool, attempts: int, max_attempts: int,
                          verification_feedback: List[str] = None,
                          previous_confidence: float = None) -> str:
        """
        Generate feedback for the student's answer.
        
        Args:
            problem: The math problem
            student_answer: The student's answer
            correct_answer: The correct answer
            is_correct: Whether the student's answer is correct
            attempts: The number of attempts the student has made
            max_attempts: The maximum number of attempts allowed
            verification_feedback: Feedback from previous verification attempts
            previous_confidence: Confidence score from previous verification
            
        Returns:
            The generated feedback
        """
        try:
            # Use LLM to generate feedback
            if self.llm_service:
                # Create the system prompt
                system_prompt = """
                You are an expert math tutor providing feedback on a student's answer to a math problem.
                Your feedback should be educational, supportive, and tailored to the student's understanding.
                
                For correct answers:
                - Acknowledge the correct answer
                - Reinforce the concepts used
                - Provide additional insights if appropriate
                
                For incorrect answers:
                - Be encouraging and supportive
                - Identify specific misconceptions or errors
                - Provide guidance without giving away the answer
                - Suggest a specific approach or concept to reconsider
                
                Ensure your feedback:
                1. Is mathematically correct and precise
                2. Includes clear reasoning when appropriate
                3. Is tailored to the student's current understanding
                4. Maintains a supportive and encouraging tone
                """
                
                # Add verification feedback if available
                if verification_feedback:
                    system_prompt += "\n\nYour previous feedback had these issues that need to be addressed:\n"
                    for i, feedback in enumerate(verification_feedback, 1):
                        system_prompt += f"{i}. {feedback}\n"
                    
                    if previous_confidence is not None:
                        system_prompt += f"\nThe verification confidence was only {previous_confidence:.2f}. Please improve the quality and correctness of your feedback."
                
                # Construct the user prompt with context
                user_prompt = f"""
                Problem: {problem}
                Student answer: {student_answer}
                Correct answer: {correct_answer}
                Is student answer correct: {is_correct}
                Attempt: {attempts} of {max_attempts}
                
                Generate appropriate feedback for the student:
                """
                
                # Generate the feedback
                response = self.llm_service.generate_completion(system_prompt, user_prompt)
                feedback = response.get("content", "")
                
                return feedback
            else:
                return "I'm sorry, I can't provide feedback at this time."
        except Exception as e:
            logger.error(f"Error generating feedback: {e}")
            logger.error(traceback.format_exc())
            return "I'm sorry, I encountered an error generating feedback."

    def _check_answer_with_flexibility(self, student_answer: str, correct_answer: str) -> bool:
        """
        Check if the student's answer is correct with flexibility for formatting variations.
        
        Args:
            student_answer: The student's answer
            correct_answer: The correct answer
            
        Returns:
            True if the answer is correct, False otherwise
        """
        logger.info(f"Checking answer with flexibility: '{student_answer}' vs '{correct_answer}'")
        
        # Clean and normalize both answers
        student_clean = self._normalize_answer(student_answer)
        correct_clean = self._normalize_answer(correct_answer)
        
        # Direct string comparison after normalization
        if student_clean == correct_clean:
            return True
        
        # Try symbolic comparison using SymPy
        try:
            student_expr = sympify(student_clean)
            correct_expr = sympify(correct_clean)
            
            if simplify(student_expr - correct_expr) == 0:
                return True
        except:
            pass
        
        # Use LLM to check for mathematical equivalence
        try:
            llm_service = self._get_llm_service()
            if llm_service:
                system_prompt = """
                You are a math expert checking if two mathematical expressions are equivalent.
                Respond with ONLY the word "CORRECT" if they are equivalent, or "INCORRECT" if they are not.
                Consider different forms of the same value (e.g., 0.5 = 1/2 = 50%).
                """
                
                user_prompt = f"""
                First expression: {student_answer}
                Second expression: {correct_answer}
                
                Are these mathematically equivalent? Answer with ONLY "CORRECT" or "INCORRECT".
                """
                
                response = llm_service.generate_completion(system_prompt, user_prompt)
                result = response.get("content", "").strip().upper()
                return "CORRECT" in result
            
        except Exception as e:
            logger.error(f"Error using LLM for answer comparison: {e}")
            # Fall back to exact comparison
            return False
        
        return False

    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize an answer for comparison.
        
        Args:
            answer: The answer to normalize
            
        Returns:
            Normalized answer
        """
        if not answer:
            return ""
            
        # Convert to string if not already
        answer = str(answer)
        
        # Remove whitespace, convert to lowercase
        normalized = answer.strip().lower()
        
        # Remove currency symbols, units, and other common formatting
        normalized = re.sub(r'[$€£¥]', '', normalized)
        
        # Remove commas in numbers
        normalized = re.sub(r'(\d),(\d)', r'\1\2', normalized)
        
        # Try to extract just the numerical value if it exists
        numeric_match = re.search(r'-?\d+(\.\d+)?', normalized)
        if numeric_match:
            numeric_value = numeric_match.group(0)
            
            # Try to convert to float for numerical comparison
            try:
                return str(float(numeric_value))
            except:
                pass
        
        return normalized
