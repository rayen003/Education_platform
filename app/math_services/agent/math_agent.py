"""
Math Agent for analyzing mathematical calculations in student answers.

This module contains the MathAgent class responsible for analyzing
mathematical calculations in student answers.
"""

import json
import logging
import os
import re
from typing import Dict, Any, List, Optional, TypedDict, Tuple, Union
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass, field

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

from app.math_services.models.state import MathState, InteractionMode, ChatMessage, UserAction, UIMode
from app.math_services.services.service_container import ServiceContainer
from app.math_services.orchestration.action_router import ActionRouter

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Define the state schema
class MathStateDict(TypedDict):
    question: str
    student_answer: str
    correct_answer: Optional[str]
    symbolic_answer: Optional[Any]
    analysis: Dict[str, Any]
    feedback: Dict[str, Any]  # Changed from str to Dict to match the actual structure
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
    Agent responsible for orchestrating mathematical analysis and interactions.
    Now optimized to eliminate redundancy with command classes.
    """
    def __init__(self, model="gpt-4o-mini"):
        """Initialize the math agent with optimized command loading."""
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
        
        # Lazy-loaded command cache
        self._commands = {}
        
        # Current state tracking
        self.current_state = None
        
        # Add caching for Chain of Draft outputs
        self._cod_cache = {}
        self._cod_timestamps = {}
        self._cod_ttl = 600  # 10 minutes in seconds
        
        # Initialize the action router
        self.action_router = ActionRouter(self)
        
        logger.info(f"Initialized MathAgent with model {model}")
    
    def _get_command(self, command_name: str):
        """
        Lazy-load command classes to reduce memory footprint.
        
        Args:
            command_name: Name of the command to load
            
        Returns:
            Initialized command instance
        """
        if command_name not in self._commands:
            if command_name == "solve":
                from app.math_services.commands.solve_command import MathSolveSymbolicallyCommand
                self._commands[command_name] = MathSolveSymbolicallyCommand(self.services)
            elif command_name == "analyze":
                from app.math_services.commands.analyze_command import MathAnalyzeCalculationCommand
                self._commands[command_name] = MathAnalyzeCalculationCommand(self.services)
            elif command_name == "hint":
                from app.math_services.commands.hint_command import MathGenerateHintCommand
                self._commands[command_name] = MathGenerateHintCommand(self.services)
            elif command_name == "feedback":
                from app.math_services.commands.feedback_command import MathGenerateFeedbackCommand
                self._commands[command_name] = MathGenerateFeedbackCommand(self.services)
            elif command_name == "chat":
                from app.math_services.commands.chat_command import MathChatFollowUpCommand
                self._commands[command_name] = MathChatFollowUpCommand(self.services)
            elif command_name == "reasoning":
                from app.math_services.commands.reasoning_command import MathGenerateReasoningCommand
                self._commands[command_name] = MathGenerateReasoningCommand(self.services)
            else:
                raise ValueError(f"Unknown command: {command_name}")
        
        return self._commands[command_name]
    
    # Replace the redundant methods with wrappers around command calls
    def solve(self, state_or_problem):
        """Solve a math problem using the dedicated command."""
        # Handle either a problem string or a state object
        if isinstance(state_or_problem, str):
            state = MathState(question=state_or_problem, student_answer="")
        else:
            state = state_or_problem
            
        command = self._get_command("solve")
        updated_state = command.execute(state)
        self.current_state = updated_state
        return updated_state
    
    def analyze(self, state):
        """Analyze a student answer using the dedicated command."""
        command = self._get_command("analyze")
        updated_state = command.execute(state)
        self.current_state = updated_state
        return updated_state
    
    def generate_feedback(self, state):
        """Generate feedback using the dedicated command."""
        command = self._get_command("feedback")
        updated_state = command.execute(state)
        self.current_state = updated_state
        return updated_state
    
    def generate_hint(self, state):
        """Generate a hint using the dedicated command."""
        command = self._get_command("hint")
        updated_state = command.execute(state)
        self.current_state = updated_state
        return updated_state
    
    def handle_follow_up(self, state, follow_up_question):
        """Handle follow-up questions using the dedicated command."""
        # Store the follow_up_question in the state context
        if not hasattr(state, 'context') or state.context is None:
            state.context = {}
        
        state.context['follow_up_question'] = follow_up_question
        
        command = self._get_command("chat")
        updated_state = command.execute(state)
        self.current_state = updated_state
        return updated_state
    
    def generate_reasoning(self, state, use_cot=False):
        """
        Generate reasoning steps with option for Chain of Thought.
        
        Args:
            state: Current math state
            use_cot: Whether to use full Chain of Thought (True) or Chain of Draft (False)
            
        Returns:
            Updated state with reasoning steps
        """
        command = self._get_command("reasoning")
        # Set the reasoning mode in the context before executing
        if not hasattr(state, 'context') or not state.context:
            if hasattr(state, 'context'):
                state.context = {}
            else:
                state['context'] = {}
                
        context_field = state.context if hasattr(state, 'context') else state['context']
        context_field['reasoning_mode'] = 'cot' if use_cot else 'cod'
        
        updated_state = command.execute(state)
        self.current_state = updated_state
        return updated_state
        
    # Remove redundant methods (_generate_feedback, _generate_hints, etc.)
    # as they're now handled by the command classes

    def process_action(self, state: MathState, action: str, **params) -> MathState:
        """
        Process a user action with event-driven orchestration.
        
        Args:
            state: Current math state
            action: Action identifier (e.g., "submit_problem", "request_hint")
            **params: Additional parameters for the action
            
        Returns:
            Updated state after handling the action
        """
        # Prepare action data
        action_data = {
            "action": action,
            **params
        }
        
        # Route the action to the appropriate handler
        updated_state = self.action_router.route_action(state, action_data)
        
        # Update current state reference
        self.current_state = updated_state
        
        return updated_state
    
    def process_natural_input(self, text: str, state: Optional[MathState] = None) -> MathState:
        """
        Process natural language input and determine the appropriate action.
        
        Args:
            text: Natural language input from the user
            state: Optional existing state (creates new if None)
            
        Returns:
            Updated state after processing the input
        """
        # Create new state if not provided
        if state is None:
            state = MathState(
                question="",
                student_answer=""
            )
        
        # Infer the most likely action based on content and context
        action, params = self._infer_action_from_text(text, state)
        
        # Add the text to the parameters
        params["text"] = text
        
        # Process the inferred action
        return self.process_action(state, action.value, **params)
    
    def process_interaction(self, interaction_type: str, content: str, state: Optional[MathState] = None) -> MathState:
        """
        Process user interaction from either button or chat input.
        
        Args:
            interaction_type: Type of interaction ('button' or 'text')
            content: Button action or chat text content
            state: Optional existing state (creates new if None)
            
        Returns:
            Updated state after processing the interaction
        """
        # Create new state if not provided
        if state is None:
            state = MathState(
                question="",
                student_answer=""
            )
            
        # Set the UI interaction mode
        ui_mode = UIMode.BUTTON if interaction_type == 'button' else UIMode.TEXT
        state.ui_mode = ui_mode
        
        # Process based on interaction type
        if ui_mode == UIMode.BUTTON:
            # Handle button actions directly
            if content == 'hint':
                return self.generate_hint(state)
            elif content == 'feedback':
                return self.generate_feedback(state)
            elif content == 'reasoning':
                # Use verified Chain of Thought for detailed reasoning
                return self.generate_reasoning(state, use_cot=True)
            elif content == 'solution':
                # First generate reasoning, then format it as a solution
                state = self.generate_reasoning(state, use_cot=True)
                if hasattr(state, 'chat_history') and state.interaction_mode == InteractionMode.CHAT:
                    # If in chat mode, add a solution message to chat history
                    solution_text = "Here's the complete solution:\n\n"
                    for i, step in enumerate(state.steps):
                        solution_text += f"Step {i+1}: {step}\n\n"
                    
                    tutor_message = ChatMessage(
                        role="tutor",
                        message=solution_text,
                        timestamp=datetime.now()
                    )
                    state.chat_history.append(tutor_message)
                    state.chat_response = solution_text
                return state
            else:
                # For unknown button actions, treat as action string
                return self.process_action(state, content)
        else:  # text mode
            # For chat input, use natural language processing
            return self.process_natural_input(content, state)
    
    def _infer_action_from_text(self, text: str, state: MathState) -> Tuple[UserAction, Dict[str, Any]]:
        """
        Infer the most likely action from natural language input.
        
        Args:
            text: Natural language input
            state: Current state
            
        Returns:
            Tuple of (UserAction, parameters dictionary)
        """
        text_lower = text.lower()
        params = {}
        
        # If we have no question yet, this is a new problem
        if not state.question:
            return UserAction.SUBMIT_PROBLEM, params
        
        # Check for hint requests
        if any(phrase in text_lower for phrase in ["hint", "help me", "i'm stuck", "give me a clue"]):
            return UserAction.REQUEST_HINT, params
        
        # Check for solution requests
        if any(phrase in text_lower for phrase in ["solution", "solve it", "show me the answer", "what's the full solution"]):
            return UserAction.REQUEST_SOLUTION, params
        
        # Check for explanation requests
        explanation_match = re.search(r"explain\s+(\w+)", text_lower)
        if explanation_match or "explain" in text_lower:
            concept = explanation_match.group(1) if explanation_match else ""
            return UserAction.REQUEST_EXPLANATION, {"concept": concept}
        
        # Check for reasoning requests
        if any(phrase in text_lower for phrase in ["reasoning", "steps", "process", "method", "approach"]):
            # Check if full CoT is requested
            use_cot = any(phrase in text_lower for phrase in ["detailed", "full", "complete", "in depth"])
            return UserAction.REQUEST_REASONING, {"use_cot": use_cot}
        
        # Check for mode toggle requests
        if any(phrase in text_lower for phrase in ["switch mode", "toggle", "change to", "chat mode", "structured mode"]):
            return UserAction.TOGGLE_MODE, params
        
        # Check for reset requests
        if any(phrase in text_lower for phrase in ["reset", "start over", "clear", "new problem"]):
            preserve_history = "keep history" in text_lower
            return UserAction.RESET, {"preserve_history": preserve_history}
        
        # If in structured mode with no answer yet, this is likely an answer submission
        if state.interaction_mode == InteractionMode.STRUCTURED and not state.student_answer:
            return UserAction.SUBMIT_ANSWER, {"answer": text}
        
        # Default to follow-up question for existing problems
        return UserAction.ASK_FOLLOWUP, params

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