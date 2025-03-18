"""
State models for math services.

This module defines structured data classes for representing states in the math services
instead of using loose Dict[str, Any] structures.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import re


class InteractionMode(Enum):
    """Interaction mode for the math agent."""
    STRUCTURED = "structured"
    CHAT = "chat"


class UIMode(Enum):
    """UI interaction mode for the math agent."""
    BUTTON = "button"          # Button-based interaction 
    TEXT = "text"              # Free-text chat input interaction


class UserAction(Enum):
    """Possible user actions that drive state transitions"""
    SUBMIT_PROBLEM = "submit_problem"       # New problem submission
    SUBMIT_ANSWER = "submit_answer"         # Answer to existing problem
    REQUEST_HINT = "request_hint"           # Request for a hint
    REQUEST_SOLUTION = "request_solution"   # Request for full solution
    TOGGLE_MODE = "toggle_mode"             # Switch between structured/chat
    ASK_FOLLOWUP = "ask_followup"           # Follow-up question in chat mode
    REQUEST_EXPLANATION = "request_explanation"  # Ask for specific explanation
    REQUEST_REASONING = "request_reasoning"  # Request reasoning steps
    RESET = "reset"                         # Reset/start over
    
    @classmethod
    def from_string(cls, action_string: str) -> 'UserAction':
        """Convert string to UserAction enum"""
        try:
            return cls(action_string.lower())
        except ValueError:
            # Default to followup if we can't parse
            return cls.ASK_FOLLOWUP


@dataclass
class ChatMessage:
    """Represents a message in the chat history."""
    role: str  # 'student' or 'tutor'
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "role": self.role,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class MathAnalysis:
    """Analysis results for a math problem."""
    is_correct: bool = False
    error_type: Optional[str] = None
    misconception: Optional[str] = None
    calculation_steps: List[str] = field(default_factory=list)
    verification_result: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_correct": self.is_correct,
            "error_type": self.error_type,
            "misconception": self.misconception,
            "calculation_steps": self.calculation_steps,
            "verification_result": self.verification_result
        }


@dataclass
class MathFeedback:
    """Feedback for a math problem."""
    assessment: str = ""
    is_correct: bool = False
    proximity_score: float = 0.0
    detail: str = ""
    confidence: float = 0.0  # Confidence level (0-1) in our feedback assessment
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "assessment": self.assessment,
            "is_correct": self.is_correct,
            "proximity_score": self.proximity_score,
            "detail": self.detail,
            "confidence": self.confidence
        }


@dataclass
class MathState:
    """Represents the state of a math problem interaction."""
    question: str
    student_answer: str
    correct_answer: Optional[str] = None
    symbolic_answer: Optional[Any] = None
    
    # Analysis and feedback
    analysis: MathAnalysis = field(default_factory=MathAnalysis)
    feedback: Dict[str, Any] = field(default_factory=dict)
    proximity_score: Optional[float] = None
    
    # Hints
    hint_count: int = 0
    hints: List[str] = field(default_factory=list)
    needs_hint: bool = False
    
    # Reasoning
    steps: List[str] = field(default_factory=list)
    
    # Chat interaction
    interaction_mode: InteractionMode = InteractionMode.STRUCTURED
    ui_mode: UIMode = UIMode.TEXT
    chat_history: List[ChatMessage] = field(default_factory=list)
    chat_response: Optional[str] = None
    
    # Additional context
    context: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Action tracking
    last_action: Optional[UserAction] = None
    next_action: Optional[UserAction] = None
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for backward compatibility."""
        return {
            "question": self.question,
            "student_answer": self.student_answer,
            "correct_answer": self.correct_answer,
            "symbolic_answer": self.symbolic_answer,
            "analysis": self.analysis.to_dict(),
            "feedback": self.feedback,
            "proximity_score": self.proximity_score,
            "hint_count": self.hint_count,
            "hints": self.hints,
            "needs_hint": self.needs_hint,
            "steps": self.steps,
            "interaction_mode": self.interaction_mode.value,
            "ui_mode": self.ui_mode.value,
            "chat_history": [msg.to_dict() for msg in self.chat_history],
            "chat_response": self.chat_response,
            "context": self.context,
            "events": self.events,
            "last_action": self.last_action.value if self.last_action else None,
            "next_action": self.next_action.value if self.next_action else None,
            "action_history": self.action_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MathState':
        """Create a MathState from a dictionary (for backward compatibility)."""
        # Convert raw chat history to ChatMessage objects
        chat_history = []
        for msg in data.get("chat_history", []):
            timestamp = msg.get("timestamp")
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.now()
            
            chat_history.append(ChatMessage(
                role=msg.get("role", ""),
                message=msg.get("message", ""),
                timestamp=timestamp
            ))
        
        # Create analysis from dict
        analysis_dict = data.get("analysis", {})
        analysis = MathAnalysis(
            is_correct=analysis_dict.get("is_correct", False),
            error_type=analysis_dict.get("error_type"),
            misconception=analysis_dict.get("misconception"),
            calculation_steps=analysis_dict.get("calculation_steps", []),
            verification_result=analysis_dict.get("verification_result", {})
        )
        
        # Determine interaction mode
        mode_str = data.get("interaction_mode", "structured")
        interaction_mode = InteractionMode.CHAT if mode_str == "chat" else InteractionMode.STRUCTURED
        
        # Determine UI mode
        ui_mode_str = data.get("ui_mode", "text")
        ui_mode = UIMode.TEXT if ui_mode_str == "text" else UIMode.BUTTON
        
        return cls(
            question=data.get("question", ""),
            student_answer=data.get("student_answer", ""),
            correct_answer=data.get("correct_answer"),
            symbolic_answer=data.get("symbolic_answer"),
            analysis=analysis,
            feedback=data.get("feedback", {}),
            proximity_score=data.get("proximity_score"),
            hint_count=data.get("hint_count", 0),
            hints=data.get("hints", []),
            needs_hint=data.get("needs_hint", False),
            steps=data.get("steps", []),
            interaction_mode=interaction_mode,
            ui_mode=ui_mode,
            chat_history=chat_history,
            chat_response=data.get("chat_response"),
            context=data.get("context", {}),
            events=data.get("events", []),
            last_action=UserAction.from_string(data.get("last_action")) if data.get("last_action") else None,
            next_action=UserAction.from_string(data.get("next_action")) if data.get("next_action") else None,
            action_history=data.get("action_history", [])
        ) 