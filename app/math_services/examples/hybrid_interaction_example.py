"""
Hybrid Interaction Example.

This script demonstrates the hybrid structured/chat approach for math assessment,
allowing both structured problem-solving flow and free-form follow-up questions.
"""

import os
import sys
import logging
from datetime import datetime

# Add the project root to the Python path
# This is necessary when running the script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from app.math_services.agent.math_agent import MathAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def print_divider():
    """Print a divider for better visual separation."""
    print("\n" + "="*80 + "\n")

def print_chat_history(chat_history):
    """Print the chat history in a nicely formatted way."""
    if not chat_history:
        print("No chat history yet.")
        return
    
    for entry in chat_history:
        role = entry.get("role", "")
        message = entry.get("message", "")
        timestamp = entry.get("timestamp", "")
        
        if role.lower() == "student":
            print(f"\033[1;34m[Student at {timestamp}]:\033[0m {message}")
        elif role.lower() == "tutor":
            print(f"\033[1;32m[Tutor at {timestamp}]:\033[0m {message}")
        else:
            print(f"[{role} at {timestamp}]: {message}")

def main():
    """
    Run the hybrid interaction example.
    """
    print_divider()
    print("HYBRID MATH ASSESSMENT INTERACTION EXAMPLE")
    print("This example demonstrates both structured problem-solving and chat-based follow-up questions.")
    print_divider()
    
    # Initialize the MathAgent
    math_agent = MathAgent()
    logger.info("Math Agent initialized")
    
    # Example problems
    problems = [
        {
            "question": "Solve for x in the equation 2x + 5 = 15",
            "student_answer": "x = 5",
            "correct_answer": "x = 5"
        },
        {
            "question": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
            "student_answer": "3x^2 + 4x - 4",
            "correct_answer": "3x^2 + 4x - 5"
        },
        {
            "question": "If a triangle has sides of length 3 cm, 4 cm, and 5 cm, what is its area?",
            "student_answer": "6 cm²",
            "correct_answer": "6 cm²"
        }
    ]
    
    # Example follow-up questions
    follow_up_questions = [
        "Can you explain why my answer was wrong?",
        "How would I find the second derivative?",
        "What formula did you use to calculate the area?",
        "Can you provide another similar problem for me to practice with?",
        "Why do we use the derivative to find the slope?",
        "What's the relationship between the sides of this triangle?"
    ]
    
    # Select a problem
    problem_index = 0
    problem = problems[problem_index]
    
    # Initialize state
    state = {
        "question": problem["question"],
        "student_answer": problem["student_answer"],
        "correct_answer": problem["correct_answer"],
        "analysis": {},
        "feedback": {},
        "hint_count": 0,
        "hints": [],
        "needs_hint": False,
        "interaction_mode": "structured"  # Start in structured mode
    }
    
    # Step 1: Initial Analysis (Structured Mode)
    print(f"PROBLEM: {state['question']}")
    print(f"STUDENT ANSWER: {state['student_answer']}")
    print(f"CORRECT ANSWER: {state['correct_answer']}")
    print_divider()
    
    print("STEP 1: Initial Analysis (Structured Mode)")
    state = math_agent.analyze(state)
    
    print(f"Feedback: {state['feedback']['math']['assessment']}")
    print(f"Is Correct: {state['feedback']['math']['is_correct']}")
    print(f"Proximity Score: {state['feedback']['math']['proximity_score']}")
    
    if state["hints"]:
        print("\nHints:")
        for i, hint in enumerate(state["hints"]):
            print(f"  Hint #{i+1}: {hint}")
    
    print_divider()
    
    # Step 2: Toggle to Chat Mode for Follow-up Questions
    print("STEP 2: Switching to Chat Mode for Follow-up Questions")
    state = math_agent.toggle_interaction_mode(state)
    print(f"Current Mode: {state['interaction_mode']}")
    print_divider()
    
    # Step 3: Process a few follow-up questions
    print("STEP 3: Handling Follow-up Questions in Chat Mode")
    for i, question in enumerate(follow_up_questions[:3]):  # Use first 3 follow-up questions
        print(f"\nFollow-up Question #{i+1}: {question}")
        state = math_agent.handle_follow_up(state, question)
        print(f"Response: {state.get('chat_response', 'No response generated')}")
    
    print_divider()
    
    # Step 4: Show complete chat history
    print("STEP 4: Complete Chat History")
    print_chat_history(state.get("chat_history", []))
    print_divider()
    
    # Step 5: Toggle back to Structured Mode
    print("STEP 5: Switching back to Structured Mode")
    state = math_agent.toggle_interaction_mode(state)
    print(f"Current Mode: {state['interaction_mode']}")
    print_divider()
    
    # Step 6: Request a hint (Structured mode)
    print("STEP 6: Requesting Hints in Structured Mode")
    state["needs_hint"] = True
    state = math_agent.analyze(state)
    
    if state["hints"]:
        print("\nUpdated Hints:")
        for i, hint in enumerate(state["hints"]):
            print(f"  Hint #{i+1}: {hint}")
    
    print_divider()
    
    # Summary
    print("SUMMARY OF HYBRID INTERACTION APPROACH")
    print("1. The system begins in structured mode, providing formal feedback and hints.")
    print("2. Users can toggle to chat mode for follow-up questions.")
    print("3. In chat mode, the system maintains context of the problem and previous interactions.")
    print("4. Users can toggle back to structured mode for formal hint generation and assessment.")
    print("5. The state maintains all information across mode toggles, ensuring continuity.")
    print_divider()
    
    print("End of Example")

if __name__ == "__main__":
    main() 