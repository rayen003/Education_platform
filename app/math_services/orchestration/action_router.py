import logging
from typing import Dict, Any, Callable, Optional
from datetime import datetime
from enum import Enum

from app.math_services.models.state import MathState, InteractionMode, UserAction, ChatMessage

logger = logging.getLogger(__name__)

class ActionRouter:
    """Routes user actions to appropriate handlers, mimicking real interactions."""
    
    def __init__(self, agent):
        """
        Initialize with reference to the agent
        
        Args:
            agent: MathAgent instance
        """
        self.agent = agent
        self.handlers = self._register_handlers()
        
    def _register_handlers(self) -> Dict[UserAction, Callable]:
        """Register action handlers for different user interactions."""
        return {
            UserAction.SUBMIT_PROBLEM: self._handle_submit_problem,
            UserAction.SUBMIT_ANSWER: self._handle_submit_answer,
            UserAction.REQUEST_HINT: self._handle_request_hint,
            UserAction.REQUEST_SOLUTION: self._handle_request_solution,
            UserAction.TOGGLE_MODE: self._handle_toggle_mode,
            UserAction.ASK_FOLLOWUP: self._handle_ask_followup,
            UserAction.REQUEST_EXPLANATION: self._handle_request_explanation,
            UserAction.REQUEST_REASONING: self._handle_request_reasoning,
            UserAction.RESET: self._handle_reset,
        }
    
    def route_action(self, state: MathState, action_data: Dict[str, Any]) -> MathState:
        """
        Route a user action to the appropriate handler
        
        Args:
            state: Current MathState
            action_data: Dictionary containing action and parameters
            
        Returns:
            Updated MathState
        """
        # Extract action from data
        action_str = action_data.get("action", "")
        action = UserAction.from_string(action_str)
        
        # Record the action in state
        self._record_action(state, action, action_data)
        
        # Get the handler
        handler = self.handlers.get(action)
        if handler:
            try:
                logger.info(f"Handling action: {action.value}")
                
                # Store action data in context for potential later use
                if not hasattr(state, 'context') or state.context is None:
                    state.context = {}
                state.context['action_data'] = action_data
                
                return handler(state, action_data)
            except Exception as e:
                logger.error(f"Error handling action {action}: {str(e)}")
                # Add detailed error information
                error_info = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "action": action.value,
                    "action_data": {k: v for k, v in action_data.items() if k != 'state'}
                }
                
                # Record error in state
                self._record_error(state, action, str(e))
                
                # Add a chat response about the error if in chat mode
                if state.interaction_mode == InteractionMode.CHAT:
                    try:
                        error_message = ChatMessage(
                            role="system",
                            message=f"I encountered an error processing your request. Please try again.",
                            timestamp=datetime.now()
                        )
                        state.chat_history.append(error_message)
                        state.chat_response = "I encountered an error processing your request. Please try again."
                    except Exception as chat_error:
                        # If even adding a chat message fails, just log it
                        logger.error(f"Failed to add error message to chat: {str(chat_error)}")
                
                return state
        else:
            logger.warning(f"No handler for action {action}")
            # Add a helpful message when no handler is found
            if state.interaction_mode == InteractionMode.CHAT:
                try:
                    system_message = ChatMessage(
                        role="system",
                        message=f"I don't know how to handle '{action.value}' yet.",
                        timestamp=datetime.now()
                    )
                    state.chat_history.append(system_message)
                    state.chat_response = f"I don't know how to handle '{action.value}' yet."
                except Exception as chat_error:
                    logger.error(f"Failed to add message to chat: {str(chat_error)}")
                    
            return state
    
    def _record_action(self, state: MathState, action: UserAction, metadata: Dict[str, Any]) -> None:
        """Record a user action in the state."""
        # Initialize action_history if needed
        if not hasattr(state, 'action_history'):
            state.action_history = []
        
        # Clean metadata of any sensitive or large data
        safe_metadata = {k: v for k, v in metadata.items() 
                        if k not in ['state', 'full_text'] and not isinstance(v, bytes)}
        
        # Truncate any string values that are too long
        for k, v in safe_metadata.items():
            if isinstance(v, str) and len(v) > 100:
                safe_metadata[k] = v[:100] + "..."
        
        # Create action record
        action_record = {
            "action": action.value,
            "timestamp": datetime.now().isoformat(),
            "metadata": safe_metadata
        }
        
        # Update state
        state.last_action = action
        state.action_history.append(action_record)
        
        # Limit history size to prevent memory issues
        if len(state.action_history) > 20:
            state.action_history = state.action_history[-20:]
    
    def _record_error(self, state: MathState, action: UserAction, error_message: str) -> None:
        """Record an error in the state."""
        if not hasattr(state, 'context') or state.context is None:
            state.context = {}
            
        if "errors" not in state.context:
            state.context["errors"] = []
            
        state.context["errors"].append({
            "action": action.value,
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit error history
        if len(state.context["errors"]) > 10:
            state.context["errors"] = state.context["errors"][-10:]
    
    def _handle_submit_problem(self, state: MathState, action_data: Dict[str, Any]) -> MathState:
        """Handle submission of a new problem."""
        problem_text = action_data.get("text", "")
        
        # Create a new state for the problem
        new_state = MathState(
            question=problem_text,
            student_answer=""
        )
        
        # Preserve chat history from old state if it exists
        if hasattr(state, 'chat_history') and state.chat_history:
            new_state.chat_history = state.chat_history
        
        # Solve the problem
        new_state = self.agent.solve(new_state)
        
        return new_state
    
    def _handle_submit_answer(self, state: MathState, action_data: Dict[str, Any]) -> MathState:
        """Handle submission of an answer to an existing problem."""
        answer = action_data.get("answer", "")
        state.student_answer = answer
        
        # Analyze the answer
        state = self.agent.analyze(state)
        
        # Generate feedback
        state = self.agent.generate_feedback(state)
        
        return state
    
    def _handle_request_hint(self, state: MathState, action_data: Dict[str, Any]) -> MathState:
        """Handle request for a hint."""
        # Generate a single contextual hint
        state = self.agent.generate_hint(state)
        
        # Add hint to chat history if in chat mode
        if state.interaction_mode == InteractionMode.CHAT and state.hints:
            latest_hint = state.hints[-1]
            
            tutor_message = ChatMessage(
                role="tutor",
                message=f"Hint: {latest_hint}",
                timestamp=datetime.now()
            )
            state.chat_history.append(tutor_message)
            state.chat_response = f"Hint: {latest_hint}"
        
        return state
    
    def _handle_request_solution(self, state: MathState, action_data: Dict[str, Any]) -> MathState:
        """Handle request for the full solution with detailed reasoning."""
        # Use full Chain of Thought for solution requests
        state = self.agent.generate_reasoning(state, use_cot=True)
        
        # Add solution to chat history if in chat mode
        if state.interaction_mode == InteractionMode.CHAT and state.steps:
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
    
    def _handle_toggle_mode(self, state: MathState, action_data: Dict[str, Any]) -> MathState:
        """Handle toggling between structured and chat modes."""
        # Toggle the interaction mode
        if state.interaction_mode == InteractionMode.STRUCTURED:
            state.interaction_mode = InteractionMode.CHAT
        else:
            state.interaction_mode = InteractionMode.STRUCTURED
        
        # Add a system message to chat history when changing modes
        message = f"Switched to {state.interaction_mode.value} mode."
        
        tutor_message = ChatMessage(
            role="system",
            message=message,
            timestamp=datetime.now()
        )
        state.chat_history.append(tutor_message)
        
        return state
    
    def _handle_ask_followup(self, state: MathState, action_data: Dict[str, Any]) -> MathState:
        """Handle a follow-up question in chat mode."""
        followup_question = action_data.get("text", "")
        
        # Record the student's question in chat history
        student_message = ChatMessage(
            role="student",
            message=followup_question,
            timestamp=datetime.now()
        )
        state.chat_history.append(student_message)
        
        # Process the follow-up question
        state = self.agent.handle_follow_up(state, followup_question)
        
        return state
    
    def _handle_request_explanation(self, state: MathState, action_data: Dict[str, Any]) -> MathState:
        """Handle request for explanation of a specific concept."""
        concept = action_data.get("concept", "")
        
        # Create a follow-up question about the concept
        if not concept:
            concept = "this concept"  # Fallback
            
        followup_question = f"Can you explain {concept} in more detail?"
        
        # Handle like a follow-up question
        return self._handle_ask_followup(state, {"text": followup_question})
    
    def _handle_request_reasoning(self, state: MathState, action_data: Dict[str, Any]) -> MathState:
        """Handle request for reasoning steps."""
        # Check if full CoT is requested, otherwise use CoD (default)
        use_cot = action_data.get("use_cot", False)
        
        # Generate reasoning
        state = self.agent.generate_reasoning(state, use_cot=use_cot)
        
        # Add reasoning to chat history if in chat mode
        if state.interaction_mode == InteractionMode.CHAT and state.steps:
            reasoning_text = "Here's my reasoning:\n\n"
            for i, step in enumerate(state.steps):
                reasoning_text += f"Step {i+1}: {step}\n\n"
                
            tutor_message = ChatMessage(
                role="tutor",
                message=reasoning_text,
                timestamp=datetime.now()
            )
            state.chat_history.append(tutor_message)
            state.chat_response = reasoning_text
            
        return state
    
    def _handle_reset(self, state: MathState, action_data: Dict[str, Any]) -> MathState:
        """Handle resetting the session."""
        # Preserve chat history if requested
        preserve_history = action_data.get("preserve_history", False)
        chat_history = state.chat_history if preserve_history and hasattr(state, 'chat_history') else []
        
        # Create a fresh state
        new_state = MathState(
            question="",
            student_answer=""
        )
        new_state.chat_history = chat_history
        
        # Add reset message to chat history if preserving it
        if preserve_history and chat_history:
            system_message = ChatMessage(
                role="system",
                message="Session has been reset.",
                timestamp=datetime.now()
            )
            new_state.chat_history.append(system_message)
        
        return new_state 