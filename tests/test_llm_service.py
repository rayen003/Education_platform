import os
import sys
from dotenv import load_dotenv

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.agents.math_commands.openai_llm_service import OpenAILLMService

# Load environment variables
load_dotenv()

# Initialize the LLM service
llm_service = OpenAILLMService()

# Test hint generation
problem = "Calculate the Net Present Value (NPV) of an investment with an initial cost of $10,000, annual cash flows of $2,500 for 5 years, and a discount rate of 8%."

# First hint
hint_prompt = """
Problem: {problem}
Student answer: 
Correct answer: 
Hint count: 1
""".format(problem=problem)

hint_response = llm_service.generate_completion(
    "You are an expert math tutor providing hints to help a student solve a math problem.",
    hint_prompt
)
print("First Hint:", hint_response['content'])

# Test feedback generation
student_answer = "$1,234"
correct_answer = "-$18.23"

feedback_prompt = """
Problem: {problem}
Student answer: {student_answer}
Correct answer: {correct_answer}
The student's answer is incorrect.
Current attempt: 3 of 3
""".format(
    problem=problem,
    student_answer=student_answer,
    correct_answer=correct_answer
)

feedback_response = llm_service.generate_completion(
    "You are an expert math tutor providing detailed feedback on a student's answer to a math problem.",
    feedback_prompt
)
print("\nFeedback:", feedback_response['content'])
