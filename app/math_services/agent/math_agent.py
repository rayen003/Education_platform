"""
Math Agent for analyzing mathematical calculations in student answers.

This module contains the MathAgent class responsible for analyzing
mathematical calculations in student answers.
"""

import json
import logging
import os
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Import the Prefect workflow
from app.workflows.math_workflow import math_assessment_with_retry_flow

# Import the OpenAI LLM service
from app.math.services.llm.openai_service import OpenAILLMService

# Import the MetaAgent for verification
from app.math.agent.meta_agent import MetaAgent

# Load environment variables
load_dotenv()

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

class MathAgent:
    """
    Agent responsible for analyzing mathematical calculations in student answers.
    """
    def __init__(self, model="gpt-4o-mini"):
        # Initialize the OpenAI client for direct API calls
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize the LLM service for structured interactions
        self.llm_service = OpenAILLMService(model=model)
        
        # Initialize the meta-agent for verification
        self.meta_agent = MetaAgent(model=model)
        
    def _check_answer(self, question: str, student_answer: str, correct_answer: str = None) -> bool:
        """
        Use ChatGPT-3.5 to check if the student's answer is correct.
        """
        # Try to use SymPy for exact comparison if we have symbolic answers
        if correct_answer:
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
        
        # Query ChatGPT-3.5 to assess correctness
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        # Extract the response and convert to boolean
        result = response.choices[0].message.content.strip().lower() == "true"
        print(f"Evaluation result: {result}")
        return result
    
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
        
        # Run the Prefect workflow
        result_state = math_assessment_with_retry_flow(state, agent=self)
        
        # Update the feedback in the state
        if "math" not in state["feedback"]:
            state["feedback"]["math"] = {}
        
        # Extract feedback from the result state
        if "feedback" in result_state:
            state["feedback"]["math"]["assessment"] = result_state["feedback"]
        
        # Add hints to the feedback
        if "hints" in result_state and result_state["hints"]:
            state["feedback"]["math"]["hints"] = result_state["hints"]
        
        # Add correctness information
        state["feedback"]["math"]["is_correct"] = result_state.get("is_correct", False)
        
        # Add proximity score if available
        if "proximity_score" in result_state and result_state["proximity_score"] is not None:
            state["feedback"]["math"]["proximity_score"] = result_state["proximity_score"]
        
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