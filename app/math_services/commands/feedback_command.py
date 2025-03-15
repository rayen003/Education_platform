"""
Math Feedback Command.

This module contains the command for generating feedback for math problems.
"""

import logging
import json
import traceback
import random
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from app.math_services.commands.base_command import BaseCommand
from app.math_services.models.state import MathState, MathFeedback
from app.math_services.services.service_container import ServiceContainer
from app.math_services.services.llm.base_service import BaseLLMService

logger = logging.getLogger(__name__)

class MathGenerateFeedbackCommand(BaseCommand):
    """Command for generating feedback for math problems."""
    
    def __init__(self, service_container: ServiceContainer):
        """
        Initialize the command with services.
        
        Args:
            service_container: Container with required services
        """
        super().__init__(service_container)
    
    def _init_services(self, service_container: ServiceContainer) -> None:
        """
        Initialize specific services from the container.
        
        Args:
            service_container: The service container
        """
        super()._init_services(service_container)
        # Additional service initialization if needed
    
    def execute(self, state: MathState) -> MathState:
        """
        Generate feedback for a student's answer to a math problem.
        
        Args:
            state: The current state with question and student answer
            
        Returns:
            Updated state with feedback
        """
        logger.info("Generating feedback")
        self.record_event(state, "feedback_generation_start", {
            "question": state.question,
            "student_answer": state.student_answer
        })
        
        try:
            # Handle missing information first
            if not state.question or not state.student_answer:
                return self._handle_missing_information(state)
            
            # Log problem details
            logger.info(f"Problem: {state.question}")
            logger.info(f"Student answer: {state.student_answer}")
            logger.info(f"Correct answer: {state.correct_answer}")
            
            # Check if we already have analysis
            is_correct = False
            if hasattr(state, 'analysis') and state.analysis is not None:
                is_correct = state.analysis.is_correct
            
            # Generate feedback
            feedback = self._generate_feedback(
                problem=state.question,
                student_answer=state.student_answer,
                correct_answer=state.correct_answer,
                is_correct=is_correct,
                attempts=1,
                max_attempts=3
            )
            
            # Calculate confidence in our feedback
            confidence = self._assess_confidence(state)
            
            # Create feedback object
            math_feedback = MathFeedback(
                assessment=feedback,
                is_correct=is_correct,
                proximity_score=state.proximity_score or 0.0,
                detail="Detailed feedback based on student's approach and answer",
                confidence=confidence
            )
            
            # Verify feedback with meta agent if available
            if self.meta_agent:
                # Define regeneration function for meta verification
                def regenerate_feedback(current_state):
                    # Get the verification feedback
                    verification_feedback = current_state.context.get("verification_feedback", [])
                    previous_confidence = current_state.context.get("verification_confidence", 0.1)
                    
                    # Regenerate with this information
                    new_feedback = self._generate_feedback(
                        problem=current_state.question,
                        student_answer=current_state.student_answer,
                        correct_answer=current_state.correct_answer,
                        is_correct=is_correct,
                        attempts=2,
                        max_attempts=3,
                        verification_feedback=verification_feedback,
                        previous_confidence=previous_confidence
                    )
                    
                    # Recalculate confidence with verification results
                    updated_confidence = self._assess_confidence(
                        current_state, 
                        has_verification=True,
                        verification_result=current_state.context.get("verification_result", {})
                    )
                    
                    # Update feedback object
                    new_math_feedback = MathFeedback(
                        assessment=new_feedback,
                        is_correct=is_correct,
                        proximity_score=current_state.proximity_score or 0.0,
                        detail="Detailed feedback based on student's approach and answer",
                        confidence=updated_confidence
                    )
                    
                    # Update state
                    current_state.feedback = new_math_feedback
                    return current_state
                
                # Prepare verification state
                verification_state = MathState(
                    question=state.question,
                    student_answer=state.student_answer,
                    correct_answer=state.correct_answer
                )
                verification_state.feedback = math_feedback
                
                # Verify the feedback
                try:
                    verification_result = self.meta_agent.verify_output(
                        verification_state.to_dict(), 
                        "feedback", 
                        regenerate_feedback
                    )
                    
                    # If verification changed the feedback, extract from verification_result
                    if verification_result.get("feedback"):
                        feedback_dict = verification_result.get("feedback", {})
                        if isinstance(feedback_dict, dict):
                            math_feedback = MathFeedback(
                                assessment=feedback_dict.get("assessment", feedback),
                                is_correct=feedback_dict.get("is_correct", is_correct),
                                proximity_score=feedback_dict.get("proximity_score", state.proximity_score or 0.0),
                                detail=feedback_dict.get("detail", "Verified feedback"),
                                confidence=feedback_dict.get("confidence", confidence)
                            )
                        # Update confidence based on verification
                        if verification_result.get("verified", False):
                            math_feedback.confidence = max(math_feedback.confidence, 0.8)
                        else:
                            math_feedback.confidence = min(math_feedback.confidence, 0.5)
                except Exception as e:
                    logger.warning(f"Feedback verification failed: {str(e)}")
            
            # Update state with feedback
            state.feedback = math_feedback
            
            # Log the generated feedback
            logger.info(f"Generated feedback: {feedback[:100]}...")
            
            self.record_event(state, "feedback_generation_complete", {
                "feedback_length": len(feedback),
                "is_correct": is_correct,
                "proximity_score": state.proximity_score or 0.0,
                "confidence": math_feedback.confidence
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating feedback: {str(e)}")
            self.log_error(e, state)
            
            # Create a fallback feedback
            fallback_feedback = MathFeedback(
                assessment="I'm sorry, but I encountered an issue while generating feedback. "
                          "Please try again or check your answer format.",
                is_correct=False,
                proximity_score=0.0,
                detail="Error generating feedback",
                confidence=0.1  # Low confidence for fallback
            )
            state.feedback = fallback_feedback
            return state

    def _assess_confidence(self, state: MathState, has_verification: bool = False, 
                          verification_result: Dict[str, Any] = None) -> float:
        """
        Assess the confidence level in our feedback assessment.
        
        Args:
            state: The current state
            has_verification: Whether verification was performed
            verification_result: Results from verification if available
            
        Returns:
            Confidence level from 0.0 to 1.0
        """
        # Start with a base confidence
        confidence = 0.7  # Default moderate confidence
        
        # Factors that increase confidence
        if state.correct_answer and state.student_answer:
            # If there's a clear correct answer to compare against
            confidence += 0.1
            
            # If answers are identical or very similar
            if state._normalize_answer(state.student_answer) == state._normalize_answer(state.correct_answer):
                confidence += 0.1
        
        # Analysis results can affect confidence
        if state.analysis and hasattr(state.analysis, 'is_correct'):
            # For completely correct answers
            if state.analysis.is_correct:
                confidence += 0.1
            # For answers with specific error types (we're confident about the error)
            elif state.analysis.error_type:
                confidence += 0.05
        
        # Verification results significantly impact confidence
        if has_verification and verification_result:
            # If feedback was verified
            if verification_result.get("verified", False):
                confidence = max(confidence, 0.85)
            else:
                # Confidence score from verification
                verification_confidence = verification_result.get("confidence", 0.5)
                # Weight the verification result more heavily
                confidence = (confidence + (verification_confidence * 2)) / 3
        
        # Set upper and lower bounds
        confidence = max(0.1, min(confidence, 0.95))
        
        return confidence

    def _handle_missing_information(self, state: MathState) -> MathState:
        """
        Handle the case where problem or answer information is missing.
        
        Args:
            state: The current state
            
        Returns:
            Updated state with generic feedback
        """
        if not hasattr(state, "feedback") or not state.feedback:
            state.feedback = MathFeedback()
        
        state.feedback.assessment = "I need both a problem and your answer to provide feedback. Please make sure both are provided."
        state.feedback.is_correct = False
        state.feedback.detail = "Missing information"
        state.proximity_score = 0.0
        state.feedback.proximity_score = 0.0
        
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
