"""
Test script for the iterative verification process of the MetaAgent.

This script demonstrates how the MetaAgent uses confidence scoring and
iterative regeneration to improve the quality of outputs.
"""

import os
import json
import unittest
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv

# Import the components to test
from app.math.agent.meta_agent import MetaAgent
from app.math.commands.hint_command import MathGenerateHintCommand
from app.math.commands.feedback_command import MathGenerateFeedbackCommand
from app.math.tests.mock_agent import MockAgent

# Load environment variables
load_dotenv()

class MockLLMWithConfidence:
    """Mock LLM service that returns responses with varying confidence levels."""
    
    def __init__(self, confidence_progression=None):
        """Initialize with a list of confidence values to use in sequence."""
        self.confidence_progression = confidence_progression or [0.5, 0.7, 0.9]
        self.verification_calls = 0
        self.hint_calls = 0
        self.feedback_calls = 0
        self.call_count = 0
    
    def generate_completion(self, system_prompt, user_prompt):
        """Generate a mock completion with varying confidence levels."""
        # For verification requests
        if "verification agent" in system_prompt.lower():
            self.verification_calls += 1
            confidence = self.confidence_progression[min(self.verification_calls - 1, len(self.confidence_progression) - 1)]
            
            # Simulate different verification responses based on confidence
            if confidence < 0.5:
                return {"content": json.dumps({
                    "verified": False,
                    "confidence": confidence,
                    "reasoning_quality": confidence * 0.8,
                    "issues": [
                        "Mathematical error in step 2", 
                        "Missing explanation for key concept",
                        "Incorrect application of formula"
                    ],
                    "corrected_output": "Improved output with corrections"
                })}
            elif confidence < 0.8:
                return {"content": json.dumps({
                    "verified": False,
                    "confidence": confidence,
                    "reasoning_quality": confidence * 0.9,
                    "issues": [
                        "Minor clarity issue in explanation",
                        "The hint does not provide any specific reasoning steps or guidance on how to solve the equation.",
                        "The hint lacks clarity and educational value as it does not lead the student towards the correct answer."
                    ],
                    "corrected_output": None
                })}
            else:
                return {"content": json.dumps({
                    "verified": True,
                    "confidence": confidence,
                    "reasoning_quality": confidence,
                    "issues": [],
                    "corrected_output": None
                })}
        
        # For hint generation
        if "math tutor providing hints" in system_prompt.lower():
            self.hint_calls += 1
            hint_iteration = min(self.hint_calls, 3)
            
            if hint_iteration == 1:
                return {"content": "Hint iteration 1: Let's think about this problem step by step..."}
            elif hint_iteration == 2:
                return {"content": "Hint iteration 2: To solve the equation 2x + 5 = 15, first isolate the variable term by subtracting 5 from both sides."}
            else:
                return {"content": "Hint iteration 3: After subtracting 5 from both sides, you get 2x = 10. Now divide both sides by 2 to find the value of x."}
        
        # For feedback generation
        if "math tutor providing feedback" in system_prompt.lower():
            self.feedback_calls += 1
            feedback_iteration = min(self.feedback_calls, 3)
            
            if feedback_iteration == 1:
                return {"content": "Feedback iteration 1: Your approach shows understanding, but consider..."}
            elif feedback_iteration == 2:
                return {"content": "Feedback iteration 2: Your work demonstrates good application of the power rule and chain rule for differentiation."}
            else:
                return {"content": "Feedback iteration 3: Excellent work! Your derivative calculation is correct, showing mastery of the differentiation rules."}
        
        # Default response
        self.call_count += 1
        return {"content": f"Default response {self.call_count}"}


def test_iterative_hint_verification():
    """Test the iterative verification process for hints."""
    print("\n--- Testing Iterative Hint Verification ---\n")
    
    # Create a meta-agent with a low confidence threshold to force multiple iterations
    meta_agent = MetaAgent(confidence_threshold=0.8, max_iterations=3)
    
    # Create a mock agent with our special LLM service
    mock_agent = MockAgent()
    mock_agent.meta_agent = meta_agent
    mock_agent.llm_service = MockLLMWithConfidence([0.4, 0.7, 0.9])
    
    # Create the hint command
    hint_command = MathGenerateHintCommand(mock_agent)
    
    # Define a test problem
    problem = "Solve the equation: 2x + 5 = 15"
    student_answer = "x = 4"
    correct_answer = "x = 5"
    
    print(f"Problem: {problem}")
    print(f"Student answer: {student_answer}")
    print(f"Correct answer: {correct_answer}")
    
    # Create a state object
    state = {
        "question": problem,
        "student_answer": student_answer,
        "correct_answer": correct_answer,
        "hints": []
    }
    
    # Mock the _record_event method to prevent None return
    hint_command._record_event = lambda s, e: s
    
    # Execute the hint command
    result_state = hint_command._execute_core(state)
    
    # Check the verification metadata
    verification_key = "hint_verification"
    verification_data = {}
    
    if result_state and verification_key in result_state:
        verification_data = result_state[verification_key]
    
    print("\nVerification Results:")
    print(f"Verified: {verification_data.get('verified', 'N/A')}")
    print(f"Confidence: {verification_data.get('confidence', 'N/A')}")
    print(f"Issues: {verification_data.get('issues', [])}")
    
    # Show the final hint
    print("\nFinal Hint:")
    if result_state and "hints" in result_state and result_state["hints"]:
        print(result_state["hints"][-1])
    else:
        print("No hint generated")
    
    return result_state


def test_iterative_feedback_verification():
    """Test the iterative verification process for feedback."""
    print("\n--- Testing Iterative Feedback Verification ---\n")
    
    # Create a meta-agent with a high confidence threshold to force multiple iterations
    meta_agent = MetaAgent(confidence_threshold=0.8, max_iterations=3)
    
    # Create a mock agent with our special LLM service
    mock_agent = MockAgent()
    mock_agent.meta_agent = meta_agent
    mock_agent.llm_service = MockLLMWithConfidence([0.3, 0.6, 0.85])
    
    # Create the feedback command
    feedback_command = MathGenerateFeedbackCommand(mock_agent)
    
    # Define a test problem
    problem = "Find the derivative of f(x) = x^3 - 4x^2 + 7x - 2"
    student_answer = "f'(x) = 3x^2 - 8x + 7"
    correct_answer = "f'(x) = 3x^2 - 8x + 7"
    
    print(f"Problem: {problem}")
    print(f"Student answer: {student_answer}")
    print(f"Correct answer: {correct_answer}")
    
    # Create a state object
    state = {
        "question": problem,
        "student_answer": student_answer,
        "correct_answer": correct_answer,
        "attempts": 1,
        "max_attempts": 3,
        "feedback": []
    }
    
    # Mock the _record_event method to prevent None return
    feedback_command._record_event = lambda s, e: s
    
    # Execute the feedback command
    result_state = feedback_command._execute_core(state)
    
    # Check the verification metadata
    verification_key = "feedback_verification"
    verification_data = {}
    
    if result_state and verification_key in result_state:
        verification_data = result_state[verification_key]
    
    print("\nVerification Results:")
    print(f"Verified: {verification_data.get('verified', 'N/A')}")
    print(f"Confidence: {verification_data.get('confidence', 'N/A')}")
    print(f"Issues: {verification_data.get('issues', [])}")
    
    # Show the final feedback
    print("\nFinal Feedback:")
    if result_state and "feedback" in result_state and result_state["feedback"]:
        print(result_state["feedback"][-1])
    else:
        print("No feedback generated")
    
    return result_state


if __name__ == "__main__":
    # Test iterative hint verification
    test_iterative_hint_verification()
    
    # Test iterative feedback verification
    test_iterative_feedback_verification()
