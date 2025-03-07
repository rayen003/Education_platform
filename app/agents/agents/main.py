import os
import json
import sys
from dotenv import load_dotenv
from pathlib import Path

# Add project root to Python path
root_dir = str(Path(__file__).parent.parent.parent.parent)
sys.path.append(root_dir)
print(f"Added to path: {root_dir}")

# Import agents
from app.agents.agents.logic_agent import LogicAgent
from app.agents.agents.math_agent import MathAgent
from app.agents.agents.language_agent import LanguageAgent
from app.agents.agents.aggregator_agent import AggregatorAgent
from app.agents.feedback_system import FeedbackSystem

# Load environment variables from .env file
load_dotenv()

# Test data
test_cases = {
    "math": {
        "question": "Calculate the Net Present Value (NPV) of a project with an initial investment of $10,000 and expected cash flows of $2,500 per year for 5 years. Use a discount rate of 8%.",
        "student_answer": """
        To calculate NPV, I'll use the formula: NPV = -Initial Investment + Sum of (Cash Flow / (1 + r)^t)
        Initial investment = $10,000
        Annual cash flow = $2,500
        Discount rate = 8% = 0.08
        Time period = 5 years
        
        NPV = -10,000 + 2,500/(1.08)^1 + 2,500/(1.08)^2 + 2,500/(1.08)^3 + 2,500/(1.08)^4 + 2,500/(1.08)^5
        NPV = -10,000 + 2,314.81 + 2,143.35 + 1,984.58 + 1,837.57 + 1,701.46
        NPV = -10,000 + 9,981.77
        NPV = -18.23
        
        Since NPV is negative, the project should be rejected.
        """,
        "feedback": {"math": {}}
    },
    "logic": {
        "question": "Explain the concept of opportunity cost and provide an example.",
        "student_answer": """
        Opportunity cost is what you give up when you make a choice. It's the value of the next best alternative.
        For example, if I decide to go to college, the opportunity cost is the money I could have earned by working full-time instead.
        However, the benefit of education might outweigh this cost in the long run because I could earn more money with a degree.
        """,
        "feedback": {"logic": {}}
    },
    "language": {
        "question": "Describe the main principles of monetary policy and its effects on economic activity.",
        "student_answer": """
        Monetary policy is when central banks control money supply to effect the economy. They use interest rates to make money expensive or cheap. When rates go up, people borrow less and economy slows down. When rates go down, people borrow more and economy speeds up. The central bank does this to control inflation and unemployment. But sometimes it doesn't work right away because of time lags.
        """,
        "feedback": {"language": {}}
    }
}

def test_math_agent():
    """Test the math agent individually"""
    print("\n=== Testing Math Agent ===\n")
    math_agent = MathAgent()
    
    result = math_agent.analyze({
        "question": test_cases["math"]["question"],
        "student_answer": test_cases["math"]["student_answer"],
        "feedback": test_cases["math"]["feedback"]
    })
    
    print(json.dumps(result["feedback"]["math"], indent=2))
    return result

def test_logic_agent():
    """Test the logic agent individually"""
    print("\n=== Testing Logic Agent ===\n")
    logic_agent = LogicAgent()
    
    result = logic_agent.analyze({
        "question": test_cases["logic"]["question"],
        "student_answer": test_cases["logic"]["student_answer"],
        "feedback": test_cases["logic"]["feedback"]
    })
    
    print(json.dumps(result["feedback"]["logic"], indent=2))
    return result

def test_language_agent():
    """Test the language agent individually"""
    print("\n=== Testing Language Agent ===\n")
    language_agent = LanguageAgent()
    
    result = language_agent.analyze({
        "question": test_cases["language"]["question"],
        "student_answer": test_cases["language"]["student_answer"],
        "feedback": test_cases["language"]["feedback"]
    })
    
    print(json.dumps(result["feedback"]["language"], indent=2))
    return result

def test_aggregator(math_feedback, logic_feedback, language_feedback):
    """Test the aggregator agent"""
    print("\n=== Testing Aggregator Agent ===\n")
    aggregator = AggregatorAgent()
    
    # Use the math test case just to have complete data
    state = {
        "question": test_cases["math"]["question"],
        "student_answer": test_cases["math"]["student_answer"],
        "feedback": {
            "math": math_feedback["feedback"]["math"],
            "logic": logic_feedback["feedback"]["logic"],
            "language": language_feedback["feedback"]["language"],
            "aggregate": {}
        }
    }
    
    result = aggregator.aggregate(state)
    print(json.dumps(result["feedback"]["aggregate"], indent=2))
    return result

def test_full_orchestration():
    """Test the complete feedback system"""
    print("\n=== Testing Full Orchestration ===\n")
    feedback_system = FeedbackSystem()
    
    # Define a simple test case
    test_case = {
        "question": "What is 2 + 2?",
        "student_answer": "2 + 2 = 4",
        "feedback": {"math": {}}
    }
    
    feedback = feedback_system.provide_feedback(
        test_case["student_answer"],
        test_case["question"],
        test_case["feedback"]
    )
    
    print(json.dumps(feedback, indent=2))
    return feedback

if __name__ == "__main__":
    # Test individual agents
    math_result = test_math_agent()
    logic_result = test_logic_agent()
    language_result = test_language_agent()
    
    # Test aggregator
    aggregator_result = test_aggregator(math_result, logic_result, language_result)
    
    # Test full orchestration (uncomment when ready)
    orchestration_result = test_full_orchestration()
    
    print("\n=== All tests completed ===\n")