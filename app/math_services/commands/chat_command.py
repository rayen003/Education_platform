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
    
    def execute(self, state: MathState, follow_up_question: str) -> MathState:
        """
        Process a follow-up question and generate a response.
        
        Args:
            state: The current state
            follow_up_question: The student's follow-up question
            
        Returns:
            Updated state with the response to the follow-up question
        """
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
            if state.feedback and "math" in state.feedback:
                assessment = state.feedback.get("math", {}).get("assessment", "")
            
            # Get is_correct status
            is_correct = False
            if state.analysis:
                is_correct = state.analysis.is_correct
            elif state.feedback and "math" in state.feedback:
                is_correct = state.feedback.get("math", {}).get("is_correct", False)
            
            # Get proximity score
            proximity_score = state.proximity_score or 0
            if state.feedback and "math" in state.feedback:
                proximity_score = state.feedback.get("math", {}).get("proximity_score", 0)
            
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
            self.log_error(e, state)
            
            # Add a fallback response in case of errors
            fallback_response = (
                "I'm sorry, I encountered an issue while processing your question. "
                "Could you please rephrase or ask something else about this math problem?"
            )
            
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
            
            return state 