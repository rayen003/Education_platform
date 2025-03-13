"""
Test script for hint generation with step-by-step reasoning verification.

This script demonstrates how the hint generation works alongside
the new step-by-step reasoning verification functionality.
"""

import os
import json
import logging
from dotenv import load_dotenv

from app.math.agent.meta_agent import MetaAgent
from app.math.commands.hint_command import MathGenerateHintCommand
from app.math.commands.reasoning_command import MathGenerateReasoningCommand
from app.math.tests.mock_agent import MockAgent

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MockLLMForHintAndReasoning:
    """Mock LLM service that simulates hint generation and step-by-step verification."""
    
    def __init__(self, problem_type="quadratic"):
        """
        Initialize with a problem type.
        
        Args:
            problem_type: Type of math problem to generate steps for
        """
        self.problem_type = problem_type
        self.hint_calls = 0
        self.reasoning_calls = 0
        self.verification_calls = 0
        self.regeneration_calls = 0
        
    def generate_completion(self, system_prompt, user_prompt):
        """Generate a mock completion based on the request type."""
        
        # For hint generation
        if "expert math tutor providing hints" in system_prompt:
            self.hint_calls += 1
            hint_count = 1
            
            # Extract hint count from user prompt
            for line in user_prompt.split('\n'):
                if "Hint count:" in line:
                    try:
                        hint_count = int(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
            
            # Return different hints based on problem type and hint count
            if self.problem_type == "quadratic":
                if hint_count == 1:
                    return {"content": "Look at the form of the equation. What type of equation is this?"}
                elif hint_count == 2:
                    return {"content": "For a quadratic equation ax² + bx + c = 0, identify the values of a, b, and c."}
                elif hint_count == 3:
                    return {"content": "Consider using the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a"}
                else:
                    return {"content": "Substitute a=1, b=-3, c=2 into the quadratic formula and solve."}
            elif self.problem_type == "calculus":
                if hint_count == 1:
                    return {"content": "When finding the derivative of a composite function, consider using the chain rule."}
                elif hint_count == 2:
                    return {"content": "Identify the outer function (sin) and the inner function (x²)."}
                elif hint_count == 3:
                    return {"content": "The chain rule states that if f(x) = g(h(x)), then f'(x) = g'(h(x)) · h'(x)."}
                else:
                    return {"content": "Calculate the derivative of sin(u) with respect to u, and the derivative of x² with respect to x."}
            else:
                return {"content": f"Hint {hint_count} for {self.problem_type} problem."}
        
        # For step-by-step reasoning generation
        elif "step-by-step reasoning for a math problem" in system_prompt:
            self.reasoning_calls += 1
            
            # Return different sets of reasoning steps based on problem type
            if self.problem_type == "quadratic":
                return {"content": json.dumps([
                    "First, we identify that this is a quadratic equation in the form ax² + bx + c = 0.",
                    "For this equation, a = 1, b = -3, and c = 2.",
                    "Using the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a",
                    "Substituting our values: x = (3 ± √(9 - 8)) / 2 = (3 ± √1) / 2 = (3 ± 1) / 2",
                    "This gives us x = 2 or x = 1"
                ])}
            elif self.problem_type == "calculus":
                return {"content": json.dumps([
                    "To find the derivative of f(x) = sin(x²), we need to apply the chain rule.",
                    "Let u = x² so f(x) = sin(u).",
                    "The derivative of sin(u) with respect to u is cos(u).",
                    "The derivative of u = x² with respect to x is 2x.",
                    "By the chain rule, f'(x) = (d/du)[sin(u)] × (d/dx)[u] = cos(u) × 2x = 2x·cos(x²)."
                ])}
            else:
                return {"content": json.dumps([
                    "Step 1 of the solution process.",
                    "Step 2 of the solution process.",
                    "Step 3 of the solution process.",
                    "Final step with the answer."
                ])}
        
        # For verification (both hint and reasoning steps)
        elif "verification agent" in system_prompt:
            self.verification_calls += 1
            
            # For simplicity, always return a positive verification
            return {"content": json.dumps({
                "verified": True,
                "confidence": 90,
                "reasoning_quality": 85,
                "issues": [],
                "explanation": "This content is correct and clear."
            })}
        
        # For regeneration
        elif "fixing a specific step" in system_prompt or "previous hint had these issues" in system_prompt:
            self.regeneration_calls += 1
            
            # Return an improved version
            return {"content": "Improved version of the content after regeneration."}
        
        # Default response for other prompts
        return {"content": "Default response for unrecognized prompt type"}


def test_hint_with_reasoning(problem_type="quadratic"):
    """Test the hint generation alongside step-by-step reasoning verification."""
    print(f"\n--- Testing Hint Generation with Reasoning: {problem_type.upper()} PROBLEM ---\n")
    
    # Create a meta-agent
    meta_agent = MetaAgent(confidence_threshold=0.75, max_iterations=3)
    
    # Create a mock agent with our special LLM service
    mock_agent = MockAgent()
    mock_agent.meta_agent = meta_agent
    mock_agent.llm_service = MockLLMForHintAndReasoning(problem_type=problem_type)
    
    # Set the LLM service on the meta-agent as well
    meta_agent.llm_service = mock_agent.llm_service
    
    # Create the commands
    hint_command = MathGenerateHintCommand(mock_agent)
    reasoning_command = MathGenerateReasoningCommand(mock_agent)
    
    # Define test problems based on problem type
    if problem_type == "quadratic":
        problem = "Solve the quadratic equation: x² - 3x + 2 = 0"
        student_answer = "I'm not sure how to start."
        correct_answer = "x = 2, x = 1"
    elif problem_type == "calculus":
        problem = "Find the derivative of f(x) = sin(x²)"
        student_answer = "I think I need to use the chain rule, but I'm stuck."
        correct_answer = "f'(x) = 2x·cos(x²)"
    else:
        problem = "Generic math problem"
        student_answer = "Generic partial answer"
        correct_answer = "Generic answer"
    
    print(f"Problem: {problem}")
    print(f"Student answer: {student_answer}")
    print(f"Correct answer: {correct_answer}")
    
    # Create a state object
    state = {
        "question": problem,
        "student_answer": student_answer,
        "correct_answer": correct_answer,
        "analysis": {
            "approach": "incomplete",
            "misconceptions": ["formula application"],
            "completion_level": "minimal"
        }
    }
    
    # First, generate step-by-step reasoning
    print("\n1. Generating Step-by-Step Reasoning:")
    state = reasoning_command._execute_core(state)
    
    # Display the reasoning steps
    print("\nReasoning Steps:")
    for i, step in enumerate(state.get("reasoning_steps", [])):
        print(f"Step {i+1}: {step}")
    
    # Now generate hints
    print("\n2. Generating Progressive Hints:")
    
    # Generate three progressive hints
    for i in range(3):
        state = hint_command._execute_core(state)
        print(f"\nHint {i+1}: {state.get('hints', [])[-1]}")
    
    # Show statistics
    llm_service = mock_agent.llm_service
    print("\nStatistics:")
    print(f"Reasoning generation calls: {llm_service.reasoning_calls}")
    print(f"Hint generation calls: {llm_service.hint_calls}")
    print(f"Verification calls: {llm_service.verification_calls}")
    print(f"Regeneration calls: {llm_service.regeneration_calls}")
    
    return state


def run_all_tests():
    """Run tests for different problem types."""
    test_hint_with_reasoning("quadratic")
    test_hint_with_reasoning("calculus")


if __name__ == "__main__":
    run_all_tests()
