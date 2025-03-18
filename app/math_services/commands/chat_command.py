"""
Chat Follow-up Command.

This module contains the command for handling follow-up questions
and chat interactions in math problem-solving.
"""

import logging
import traceback
from typing import Dict, Any, Optional
from datetime import datetime

from app.math_services.commands.base_command import BaseCommand
from app.math_services.models.state import MathState, ChatMessage, InteractionMode
from app.math_services.services.service_container import ServiceContainer

logger = logging.getLogger(__name__)

class MathChatFollowUpCommand(BaseCommand):
    """Command for handling chat-based follow-up questions in math problem-solving."""
    
    def execute(self, state: MathState) -> MathState:
        """
        Process a follow-up question and generate a response.
        
        Args:
            state: The current state with follow_up_question in context
            
        Returns:
            Updated state with the response to the follow-up question
        """
        # Get the follow-up question from state
        follow_up_question = ""
        try:
            if hasattr(state, 'context') and state.context and 'follow_up_question' in state.context:
                follow_up_question = state.context['follow_up_question']
            elif hasattr(state, 'context') and state.context and 'action_data' in state.context:
                follow_up_question = state.context['action_data'].get('text', '')
            
            # If still no follow-up question, check if passed as a text attribute
            if not follow_up_question and hasattr(state, 'text'):
                follow_up_question = state.text
                
            if not follow_up_question:
                # Last resort: try to extract from the last action in action_history
                if hasattr(state, 'action_history') and state.action_history:
                    last_action = state.action_history[-1]
                    if isinstance(last_action, dict) and 'metadata' in last_action:
                        follow_up_question = last_action['metadata'].get('text', '')
                
            if not follow_up_question:
                logger.warning("No follow-up question found in state")
                self.record_event(state, "error", {"message": "No follow-up question found"})
                state.chat_response = "I'm sorry, I couldn't understand your question. Can you please try again?"
                return state
        except Exception as e:
            logger.error(f"Error extracting follow-up question: {str(e)}")
            state.chat_response = "I'm sorry, I encountered an error processing your question. Can you please try again?"
            self.record_event(state, "error", {"message": f"Error extracting follow-up question: {str(e)}"})
            return state
            
        logger.info(f"Processing follow-up question: {follow_up_question}")
        
        try:
            # Ensure the interaction mode is set to chat
            if state.interaction_mode != InteractionMode.CHAT:
                state.interaction_mode = InteractionMode.CHAT
                logger.info(f"Switched interaction mode to {state.interaction_mode.value}")
            
            # Extract relevant context from the state
            question = state.question
            student_answer = state.student_answer
            correct_answer = state.correct_answer or ""
            previous_hints = state.hints
            
            # Get feedback assessment if available
            assessment = ""
            if hasattr(state, "feedback") and state.feedback:
                if isinstance(state.feedback, dict) and "math" in state.feedback:
                    # Dictionary format (legacy)
                    assessment = state.feedback.get("math", {}).get("assessment", "")
                elif hasattr(state.feedback, "assessment"):
                    # MathFeedback object format
                    assessment = state.feedback.assessment
            
            # Get is_correct status
            is_correct = False
            if hasattr(state, "analysis") and state.analysis:
                is_correct = state.analysis.is_correct
            elif hasattr(state, "feedback") and state.feedback:
                if isinstance(state.feedback, dict) and "math" in state.feedback:
                    # Dictionary format (legacy)
                    is_correct = state.feedback.get("math", {}).get("is_correct", False)
                elif hasattr(state.feedback, "is_correct"):
                    # MathFeedback object format
                    is_correct = state.feedback.is_correct
            
            # Get proximity score
            proximity_score = state.proximity_score or 0
            if hasattr(state, "feedback") and state.feedback:
                if isinstance(state.feedback, dict) and "math" in state.feedback:
                    # Dictionary format (legacy)
                    proximity_score = state.feedback.get("math", {}).get("proximity_score", 0)
                elif hasattr(state.feedback, "proximity_score"):
                    # MathFeedback object format
                    proximity_score = state.feedback.proximity_score
            
            # Format the chat history for context
            history_formatted = ""
            if state.chat_history:
                for entry in state.chat_history:
                    role = entry.role
                    message = entry.message
                    if role and message:
                        history_formatted += f"{role.capitalize()}: {message}\n"
            
            # Prepare the system prompt with context awareness
            system_prompt = """
            You are a knowledgeable math tutor in an interactive chat session with a student.
            
            You have already provided structured feedback on their math problem, and now they have a follow-up question.
            
            Use the context of the problem, their answer, and previous interactions to provide a helpful, accurate response.
            
            Remember:
            1. Be supportive and encouraging
            2. Give explanations that promote understanding, not just answers
            3. If they're asking for too much help, guide them toward thinking through the problem themselves
            4. Your responses should be clear, concise, and directly address their question
            5. If their question is unclear or unrelated to the math problem, politely ask for clarification
            """
            
            # Prepare the user prompt with complete context
            user_prompt = f"""
            PROBLEM CONTEXT:
            Math Problem: {question}
            Student's Answer: {student_answer}
            Correct Answer: {correct_answer}
            Answer Status: {"Correct" if is_correct else "Incorrect"}
            Proximity Score: {proximity_score}/10
            
            FEEDBACK ALREADY PROVIDED:
            Assessment: {assessment}
            {"Hints Already Given: " + "\n".join([f"- {hint}" for hint in previous_hints]) if previous_hints else "No hints provided yet"}
            
            PREVIOUS CHAT INTERACTIONS:
            {history_formatted if history_formatted else "This is the first follow-up question."}
            
            CURRENT FOLLOW-UP QUESTION:
            {follow_up_question}
            
            Please respond to the student's follow-up question:
            """
            
            # Generate the response using the LLM service
            response = self.llm_service.generate_completion(
                system_prompt,
                user_prompt,
                temperature=0.7,
                max_tokens=500
            )
            
            # Extract the response content
            chat_response = response.get("content", "").strip()
            
            # Log only the content, not the entire response object
            logger.info(f"Generated chat response: {chat_response}")
            
            # Add the student's question to chat history
            student_message = ChatMessage(
                role="student",
                message=follow_up_question,
                timestamp=datetime.now()
            )
            state.chat_history.append(student_message)
            
            # Add the tutor's response to chat history
            tutor_message = ChatMessage(
                role="tutor",
                message=chat_response,
                timestamp=datetime.now()
            )
            state.chat_history.append(tutor_message)
            
            # Add the latest response to state
            state.chat_response = chat_response
            
            # Record the event
            self.record_event(state, "chat_interaction", {
                "follow_up_question": follow_up_question,
                "response": chat_response
            })
            
            return state
            
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            logger.error(f"Error in chat follow-up processing: {error_type}: {error_message}")
            logger.error(traceback.format_exc())
            self.log_error(e, state)
            
            # Determine appropriate fallback message based on error type
            if "ConnectionError" in error_type or "Timeout" in error_type:
                fallback_response = (
                    "I'm having trouble connecting to our reasoning engine right now. "
                    "This might be due to a temporary network issue. Could you please try again in a moment?"
                )
            elif "KeyError" in error_type or "AttributeError" in error_type:
                fallback_response = (
                    "I'm missing some information I need to answer your question properly. "
                    "Could you please rephrase your question and make sure it's related to the math problem we're discussing?"
                )
            else:
                fallback_response = (
                    "I'm sorry, I encountered an issue while processing your question. "
                    "Could you please rephrase or ask something else about this math problem?"
                )
            
            try:
                # Safely add the student's question to chat history
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
                
                # Record the error event
                self.record_event(state, "chat_error", {
                    "error_type": error_type,
                    "follow_up_question": follow_up_question
                })
            except Exception as inner_e:
                # Last resort error handling if adding to chat history also fails
                logger.critical(f"Critical error in fallback handling: {str(inner_e)}")
                logger.critical(traceback.format_exc())
                
                # Create a minimal valid state to return
                if not hasattr(state, "chat_response") or state.chat_response is None:
                    state.chat_response = "I'm sorry, I encountered a technical issue. Please try again later."
            
            return state 