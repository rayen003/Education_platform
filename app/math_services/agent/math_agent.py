"""
Math Agent for analyzing mathematical calculations in student answers.

This module contains the MathAgent class responsible for analyzing
mathematical calculations in student answers.
"""

import json
import logging
import os
import re
from typing import Dict, Any, List, Optional, TypedDict
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# Try to import SymPy for symbolic math operations
try:
    from sympy import parse_expr, simplify
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("SymPy not available, falling back to text-based comparison")

# Import the OpenAI LLM service
from app.math_services.services.llm.openai_service import OpenAILLMService

# Import the MetaAgent for verification if available
try:
    from app.math_services.agent.meta_agent import MetaAgent
    META_AGENT_AVAILABLE = True
except ImportError:
    META_AGENT_AVAILABLE = False
    print("MetaAgent not available, skipping verification")

from app.math_services.models.state import MathState, InteractionMode, ChatMessage
from app.math_services.services.service_container import ServiceContainer

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Define the state schema
class MathState(TypedDict):
    question: str
    student_answer: str
    correct_answer: Optional[str]
    symbolic_answer: Optional[Any]
    analysis: Dict[str, Any]
    feedback: str
    proximity_score: Optional[float]
    hint_count: int
    hints: List[str]
    needs_hint: bool
    # Fields added for chat-based follow-ups
    context: Optional[Dict[str, Any]]
    chat_history: Optional[List[Dict[str, Any]]]
    chat_response: Optional[str]
    interaction_mode: Optional[str]  # "structured" or "chat"

class MathAgent:
    """
    Agent responsible for analyzing mathematical calculations in student answers.
    """
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the math agent.
        
        Args:
            model: The OpenAI model to use
        """
        # Initialize LLM service and service container
        self.llm_service = OpenAILLMService(model=model)
        
        # Initialize the meta-agent
        self.meta_agent = MetaAgent(
            model=model,
            llm_service=self.llm_service
        )
        
        # Create service container
        self.services = ServiceContainer(
            llm_service=self.llm_service,
            meta_agent=self.meta_agent
        )
        
        # Cache for commands
        self._commands = {}
        
        logger.info(f"Initialized MathAgent with model {model}")
        
    def _check_answer(self, question: str, student_answer: str, correct_answer: str = None) -> bool:
        """
        Use ChatGPT-3.5 to check if the student's answer is correct.
        """
        # Try to use SymPy for exact comparison if we have symbolic answers
        if SYMPY_AVAILABLE and correct_answer:
            try:
                # Try to parse both answers as expressions
                student_expr = parse_expr(student_answer.strip())
                correct_expr = parse_expr(correct_answer.strip())
                
                # Check if they are equivalent
                if simplify(student_expr - correct_expr) == 0:
                    print("SymPy verified the answer is correct")
                    return True
            except Exception as e:
                print(f"SymPy comparison failed: {e}")
        
        # Fall back to ChatGPT-3.5 for assessment
        messages = [
            {
                "role": "system",
                "content": """You are a math evaluation assistant. Analyze the student's answer to determine if it is correct.
                
                Follow these steps:
                1. Solve the math problem yourself first.
                2. Compare your solution with the student's answer.
                3. Consider different formats of correct answers (e.g., '2' vs '2.0' vs 'x=2').
                4. Respond with 'True' if the student's answer is correct, or 'False' if it's incorrect.
                
                Be precise in your evaluation and give the student the benefit of the doubt if their answer is correct but formatted differently.
                """
            }
        ]
        
        if correct_answer:
            messages.append({
                "role": "user",
                "content": f"Question: {question}\nStudent Answer: {student_answer}\nCorrect Answer: {correct_answer}"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Question: {question}\nStudent Answer: {student_answer}"
            })
        
        try:
            # Query ChatGPT-3.5 to assess correctness
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            
            # Extract the response and convert to boolean
            result = response.choices[0].message.content.strip().lower() == "true"
            print(f"Evaluation result: {result}")
            return result
        except Exception as e:
            print(f"Error in OpenAI assessment: {e}")
            # If OpenAI call fails, do a simple string comparison as fallback
            if correct_answer:
                # Remove spaces and make case-insensitive
                student_clean = student_answer.replace(" ", "").lower()
                correct_clean = correct_answer.replace(" ", "").lower()
                return student_clean == correct_clean
            return False
    
    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the student's answer and generate feedback.
        
        Args:
            state: Dictionary containing the question, student_answer, and feedback fields
            
        Returns:
            Updated state with feedback
        """
        print(f"Analyzing math question: {state.get('question', '')}")
        
        # Initialize state fields if not present
        if "hint_count" not in state:
            state["hint_count"] = 0
        if "hints" not in state:
            state["hints"] = []
        if "needs_hint" not in state:
            state["needs_hint"] = False
        if "proximity_score" not in state:
            state["proximity_score"] = None
        if "interaction_mode" not in state:
            state["interaction_mode"] = "structured"  # Default to structured mode
        if "chat_history" not in state:
            state["chat_history"] = []
        
        try:
            # Check if we need to build or update the context
            if "context" not in state:
                state["context"] = {
                    "question": state.get("question", ""),
                    "student_answer": state.get("student_answer", ""),
                    "correct_answer": state.get("correct_answer", ""),
                    "previous_hints": state.get("hints", []),
                    "assessment": "",
                    "is_correct": False,
                    "proximity_score": 0
                }
            
            # For the simplified UI demo, let's just use a direct check rather than the workflow
            # Check if the answer is correct
            is_correct = self._check_answer(
                state["question"], 
                state["student_answer"], 
                state.get("correct_answer")
            )
            
            # Generate feedback
            feedback = self._generate_feedback(
                state["question"], 
                state["student_answer"], 
                is_correct
            )
            
            # Generate hints if needed
            if state.get("needs_hint", False) or not is_correct:
                hints = self._generate_hints(
                    state["question"], 
                    state["student_answer"], 
                    state.get("hint_count", 0) + 1
                )
                state["hints"] = hints
            
            # Calculate proximity score
            proximity_score = self._calculate_proximity(
                state["question"], 
                state["student_answer"], 
                state.get("correct_answer")
            )
            
            # Update the feedback in the state
            if "feedback" not in state:
                state["feedback"] = {}
            if "math" not in state["feedback"]:
                state["feedback"]["math"] = {}
            
            state["feedback"]["math"]["assessment"] = feedback
            state["feedback"]["math"]["is_correct"] = is_correct
            state["feedback"]["math"]["proximity_score"] = proximity_score
            
            # Update the context with the latest information
            state["context"].update({
                "question": state.get("question", ""),
                "student_answer": state.get("student_answer", ""),
                "correct_answer": state.get("correct_answer", ""),
                "previous_hints": state.get("hints", []),
                "assessment": feedback,
                "is_correct": is_correct,
                "proximity_score": proximity_score
            })
            
            return state
        except Exception as e:
            print(f"Error in math agent analyze: {str(e)}")
            if "feedback" not in state:
                state["feedback"] = {}
            if "math" not in state["feedback"]:
                state["feedback"]["math"] = {}
            
            state["feedback"]["math"]["assessment"] = "I encountered an error analyzing your answer."
            state["feedback"]["math"]["is_correct"] = False
            
            # Update context with the error information
            if "context" in state:
                state["context"].update({
                    "assessment": "I encountered an error analyzing your answer.",
                    "is_correct": False
                })
            
            return state
        
    def _generate_feedback(self, question: str, student_answer: str, is_correct: bool) -> str:
        """Generate detailed feedback for the student's answer."""
        system_prompt = """You are a helpful math tutor providing feedback on a student's answer.
        
        Your feedback should:
        1. Be encouraging and supportive even when the answer is incorrect
        2. Highlight what the student did correctly (if anything)
        3. Identify misconceptions or errors in their approach
        4. Suggest a better approach if the answer is incorrect
        5. Keep your response concise (2-3 sentences)
        """
        
        user_prompt = f"""
        Question: {question}
        Student Answer: {student_answer}
        Is Correct: {is_correct}
        
        Provide feedback for this student's answer.
        """
        
        try:
            response = self.llm_service.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7
            )
            return response.get("content", "No feedback available")
        except Exception as e:
            print(f"Error generating feedback: {e}")
            return "I couldn't analyze your answer properly. Please try again." 

    def _generate_hints(self, question: str, student_answer: str, hint_count: int) -> List[str]:
        """Generate a sequence of hints for the problem."""
        system_prompt = f"""You are a math tutor providing hints for a problem. 
        The student needs {hint_count} progressive hints that gradually reveal the solution approach.
        
        Format your response as a list of {hint_count} hints, with each hint building on the previous one.
        Start with general conceptual hints and gradually get more specific.
        
        Return EXACTLY {hint_count} hints, numbered 1 to {hint_count}.
        """
        
        user_prompt = f"""
        Question: {question}
        Student's Current Answer: {student_answer}
        
        Provide {hint_count} progressive hints to help the student solve this problem.
        """
        
        try:
            response = self.llm_service.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7
            )
            
            content = response.get("content", "")
            
            # Parse hints from the response
            hints = []
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith(f"{len(hints)+1}.") or line.startswith(f"Hint {len(hints)+1}:")):
                    hint_text = line.split(":", 1)[-1] if ":" in line else line.split(".", 1)[-1]
                    hints.append(hint_text.strip())
                    if len(hints) >= hint_count:
                        break
            
            # If we couldn't parse the hints properly, create default ones
            if len(hints) < hint_count:
                missing = hint_count - len(hints)
                for i in range(missing):
                    hints.append(f"Hint {len(hints)+1}: Consider reviewing the relevant concepts for this problem.")
            
            return hints
        except Exception as e:
            print(f"Error generating hints: {e}")
            return [f"Hint {i+1}: I couldn't generate a proper hint. Please try again." for i in range(hint_count)]

    def _calculate_proximity(self, question: str, student_answer: str, correct_answer: str = None) -> float:
        """Calculate how close the student's answer is to the correct answer."""
        system_prompt = """You are a math assessment system calculating a proximity score.
        
        Rate how close the student's answer is to the correct solution on a scale of 0-10:
        - 10: Completely correct answer and approach
        - 7-9: Minor errors but correct approach
        - 4-6: Partially correct with significant errors
        - 1-3: Incorrect answer but shows some understanding
        - 0: Completely incorrect or irrelevant
        
        Respond ONLY with a number from 0 to 10.
        """
        
        user_prompt = f"""
        Question: {question}
        Student Answer: {student_answer}
        """
        
        if correct_answer:
            user_prompt += f"\nCorrect Answer: {correct_answer}"
        
        try:
            response = self.llm_service.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3
            )
            
            score_text = response.get("content", "").strip()
            
            # Extract the score
            score = None
            try:
                # Try to extract just the number from the response
                score = float(''.join(c for c in score_text if c.isdigit() or c == '.'))
                # Ensure it's in the range 0-10
                score = max(0, min(10, score))
            except:
                score = 5.0  # Default score if parsing fails
                
            return score
        except Exception as e:
            print(f"Error calculating proximity score: {e}")
            return 5.0  # Default middle score

    def handle_follow_up(self, state: Dict[str, Any], follow_up_question: str) -> Dict[str, Any]:
        """
        Handle a follow-up question in the chat interaction mode.
        
        Args:
            state: Current state dictionary
            follow_up_question: The student's follow-up question
            
        Returns:
            Updated state with chat response
        """
        try:
            # Convert to MathState if needed
            if not isinstance(state, MathState):
                math_state = MathState.from_dict(state)
            else:
                math_state = state
                
            # Get or create the chat command
            if "chat_command" not in self._commands:
                from app.math_services.commands.chat_command import MathChatFollowUpCommand
                self._commands["chat_command"] = MathChatFollowUpCommand(self.services)
            
            chat_command = self._commands["chat_command"]
            
            # Set the interaction mode to "chat"
            if math_state.interaction_mode != InteractionMode.CHAT:
                old_mode = math_state.interaction_mode.value
                math_state.interaction_mode = InteractionMode.CHAT
                # Log the mode change
                print(f"Switching from {old_mode} mode to chat mode")
            
            # Process the follow-up question
            updated_state = chat_command.execute(math_state, follow_up_question)
            
            # Convert back to dict if needed for backward compatibility
            if isinstance(state, dict):
                return updated_state.to_dict()
            return updated_state
                
        except Exception as e:
            print(f"Error handling follow-up question: {str(e)}")
            
            # Provide a fallback response
            fallback_response = (
                "I'm sorry, I encountered an issue while processing your question. "
                "Could you please rephrase or ask something else about this math problem?"
            )
            
            # Try to add to chat history
            try:
                if isinstance(state, MathState):
                    # Add the student's question to chat history
                    student_message = ChatMessage(
                        role="student",
                        message=follow_up_question,
                        timestamp=datetime.now()
                    )
                    state.chat_history.append(student_message)
                    
                    # Add the fallback response to chat history
                    tutor_message = ChatMessage(
                        role="tutor",
                        message=fallback_response,
                        timestamp=datetime.now()
                    )
                    state.chat_history.append(tutor_message)
                    
                    # Add the fallback response to state
                    state.chat_response = fallback_response
                else:
                    # Add to chat history as dict
                    if "chat_history" not in state:
                        state["chat_history"] = []
                        
                    # Record the student's question
                    state["chat_history"].append({
                        "role": "student",
                        "message": follow_up_question,
                        "timestamp": str(datetime.now())
                    })
                    
                    # Record the fallback response
                    state["chat_history"].append({
                        "role": "tutor",
                        "message": fallback_response,
                        "timestamp": str(datetime.now())
                    })
                    
                    # Add the fallback response to state
                    state["chat_response"] = fallback_response
            except Exception as chat_error:
                print(f"Additional error while adding to chat history: {str(chat_error)}")
                # Simplest fallback - just add the response
                if isinstance(state, MathState):
                    state.chat_response = fallback_response
                else:
                    state["chat_response"] = fallback_response
            
            return state

    def toggle_interaction_mode(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Toggle between structured and chat interaction modes.
        
        Args:
            state: Current state dictionary
            
        Returns:
            Updated state with new interaction mode
        """
        current_mode = state.get("interaction_mode", "structured")
        
        # Toggle the mode
        new_mode = "chat" if current_mode == "structured" else "structured"
        state["interaction_mode"] = new_mode
        
        print(f"Toggled interaction mode from {current_mode} to {new_mode}")
        
        return state

# Simple test case
if __name__ == "__main__":
    # Instantiate the MathAgent
    math_agent = MathAgent()

    # Define test cases
    test_cases = [
        {
            "question": "What is 5 + 3?",
            "student_answer": "8"
        },
        {
            "question": "What is 10 - 2?",
            "student_answer": "7"
        },
        {
            "question": "Solve for x in the equation 2x + 5 = 15",
            "student_answer": "x = 4"
        },
        {
            "question": "Solve for x in the equation 3x - 7 = 8",
            "student_answer": "x = 4"
        },
        {
            "question": "Simplify the expression (x^2 - 4)/(x - 2) for x ≠ 2",
            "student_answer": "x + 2"
        },
        {
            "question": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
            "student_answer": "3x^2 + 4x - 5"
        }
    ]

    # Analyze each test case
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        try:
            # Initialize state with required fields
            state = {
                "question": test_case["question"],
                "student_answer": test_case["student_answer"],
                "analysis": {},
                "hint_count": 0,
                "hints": [],
                "needs_hint": False
            }
            
            # Run the analysis
            result = math_agent.analyze(state)
            
            # Display results
            print(f"Math Agent Feedback: {result['feedback']}")
            
            # Display hints if any were generated
            if result.get('hints') and len(result['hints']) > 0:
                print("\nHints provided:")
                for idx, hint in enumerate(result['hints']):
                    print(f"Hint #{idx+1}: {hint}")
                    
            # Display proximity score if available
            if 'proximity_score' in result and result['proximity_score'] is not None:
                print(f"Proximity Score: {result['proximity_score']}/10")
                
        except Exception as e:
            print(f"Error processing test case {i+1}: {e}")