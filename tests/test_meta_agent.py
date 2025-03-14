"""
Test script for the MetaAgent and flexible answer checking.

This script demonstrates:
1. The MetaAgent's ability to verify outputs
2. The flexible answer checking system
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.agents.meta_agent import MetaAgent
from app.agents.agents.math_commands.openai_llm_service import OpenAILLMService
from app.agents.agents.math_commands.feedback_command import MathGenerateFeedbackCommand

# Load environment variables
load_dotenv()

def test_meta_agent_verification():
    """Test the meta-agent's verification capabilities."""
    print("\n--- Testing MetaAgent Verification ---\n")
    
    # Initialize the meta-agent
    meta_agent = MetaAgent()
    
    # Test solution verification
    problem = "Calculate the area of a circle with radius 5 cm."
    solution = "The area is 78.5 square cm."
    
    print(f"Problem: {problem}")
    print(f"Solution to verify: {solution}")
    
    result = meta_agent.verify_math_solution(problem, solution)
    
    print("\nVerification Result:")
    print(f"Verified: {result['verified']}")
    
    if not result['verified'] and 'issues' in result:
        print("\nIssues found:")
        for issue in result['issues']:
            print(f"- {issue}")
        
        print(f"\nCorrected solution: {result.get('corrected_solution', 'Not provided')}")

def test_flexible_answer_checking():
    """Test the flexible answer checking system."""
    print("\n--- Testing Flexible Answer Checking ---\n")
    
    # Create a mock state and command
    class MockAgent:
        def __init__(self):
            self.llm_service = OpenAILLMService()
    
    feedback_command = MathGenerateFeedbackCommand(MockAgent())
    
    # Test cases with different formatting
    test_cases = [
        {
            "problem": "Calculate 25% of 80.",
            "correct_answer": "20",
            "student_answers": ["20", "20.0", "$20", "20 units", "twenty"]
        },
        {
            "problem": "Find the area of a circle with radius 5 cm.",
            "correct_answer": "78.54 square cm",
            "student_answers": ["78.54 cm²", "78.5 square cm", "78.5", "π×5²"]
        },
        {
            "problem": "Convert 0.75 to a fraction.",
            "correct_answer": "3/4",
            "student_answers": ["3/4", "0.75", "75%", "three quarters"]
        }
    ]
    
    for case in test_cases:
        print(f"\nProblem: {case['problem']}")
        print(f"Correct answer: {case['correct_answer']}")
        
        for student_answer in case['student_answers']:
            is_correct = feedback_command._check_answer_with_flexibility(
                student_answer, 
                case['correct_answer'],
                case['problem']
            )
            
            print(f"Student answer: '{student_answer}' -> {'Correct' if is_correct else 'Incorrect'}")

if __name__ == "__main__":
    test_meta_agent_verification()
    test_flexible_answer_checking()
