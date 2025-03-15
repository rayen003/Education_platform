"""
Test script for the MetaAgent and flexible answer checking.

This script demonstrates the functionality of the MetaAgent for verifying
outputs and the flexible answer checking capabilities.
"""

import os
import json
import unittest
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv

# Import the components to test
from app.math_services.agent.meta_agent import MetaAgent
from app.math_services.services.llm.openai_service import OpenAILLMService
from app.math_services.commands.feedback_command import MathGenerateFeedbackCommand
from app.math_services.commands.hint_command import MathGenerateHintCommand
from app.math_services.tests.mock_agent import MockAgent

# Load environment variables
load_dotenv()

def test_meta_agent_verification():
    """Test the MetaAgent's verification capabilities."""
    print("\n--- Testing MetaAgent Verification ---\n")
    
    # Create a meta-agent
    meta_agent = MetaAgent()
    
    # Define a problem and solution to verify
    problem = "Calculate the area of a circle with radius 5 cm."
    solution = "The area is 78.5 square cm."
    
    print(f"Problem: {problem}")
    print(f"Solution to verify: {solution}")
    
    # Verify the solution
    result = meta_agent.verify_math_solution(problem, solution)
    
    print("\nVerification Result:")
    print(f"Verified: {result.get('verified', False)}")
    
    if not result.get('verified', False) and result.get('corrected_solution'):
        print(f"Corrected Solution: {result.get('corrected_solution')}")
    
    return result

def test_flexible_answer_checking():
    """Test the flexible answer checking system."""
    print("\n--- Testing Flexible Answer Checking ---\n")
    
    # Create a mock agent with necessary attributes
    mock_agent = MockAgent()
    mock_agent.meta_agent = None  # No need for meta verification in this test
    
    # Mock the LLM service to always return "CORRECT" for equivalence checks
    class MockLLM:
        def generate_completion(self, system_prompt, user_prompt):
            return {"content": "CORRECT"}
    
    mock_agent.llm_service = MockLLM()
    
    # Create the feedback command with our mock agent
    feedback_command = MathGenerateFeedbackCommand(mock_agent)
    
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
    
    # Run the test cases
    for case in test_cases:
        print(f"\nProblem: {case['problem']}")
        print(f"Correct answer: {case['correct_answer']}")
        
        for student_answer in case['student_answers']:
            is_correct = feedback_command._check_answer_with_flexibility(
                student_answer, 
                case['correct_answer']
            )
            print(f"Student answer: '{student_answer}' -> {'Correct' if is_correct else 'Incorrect'}")

def test_logic_problem_hint_generation():
    """Test hint generation for a logic problem with incorrect answer."""
    print("\n--- Testing Logic Problem Hint Generation ---\n")
    
    # Create a mock agent
    mock_agent = MockAgent()
    mock_agent.meta_agent = None
    class MockLLM:
        def generate_completion(self, system_prompt, user_prompt):
            return {"content": "Let's think step by step:\n1. All Bloops are Razzies\n2. All Razzies are Lazzies\n3. Therefore, all Bloops must be Lazzies\n\nHint: Try drawing a Venn diagram to visualize the relationships."}
    mock_agent.llm_service = MockLLM()
    
    # Create the hint command
    hint_command = MathGenerateHintCommand(mock_agent)
    
    # Define a logic problem and incorrect answer
    problem = "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies?"
    incorrect_answer = "No, Bloops are not Lazzies"
    correct_answer = "Yes, all Bloops are Lazzies"
    hint_count = 1
    hints = []
    analysis = {"error_type": "logical_error", "misconception": "transitive_property"}
    
    print(f"Problem: {problem}")
    print(f"Incorrect answer: {incorrect_answer}")
    print(f"Correct answer: {correct_answer}")
    
    # Generate hint
    hint = hint_command._generate_hint(
        problem=problem, 
        student_answer=incorrect_answer, 
        correct_answer=correct_answer, 
        hint_count=hint_count, 
        hints=hints, 
        analysis=analysis
    )
    
    print("\nGenerated Hint:")
    print(hint)
    
    return hint

if __name__ == "__main__":
    # Test meta-agent verification
    test_meta_agent_verification()
    
    # Test flexible answer checking
    test_flexible_answer_checking()
    
    # Test logic problem hint generation
    test_logic_problem_hint_generation()
