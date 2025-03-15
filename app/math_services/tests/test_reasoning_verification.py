"""
Test script for the step-by-step reasoning verification functionality.

This script demonstrates how the meta-agent verifies each step of 
mathematical reasoning and regenerates steps when issues are found.
"""

import os
import json
import logging
from dotenv import load_dotenv

from app.math_services.agent.meta_agent import MetaAgent
from app.math_services.commands.reasoning_command import MathGenerateReasoningCommand
from app.math_services.tests.mock_agent import MockAgent

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MockLLMWithStepVerification:
    """Mock LLM service that simulates step-by-step verification."""
    
    def __init__(self, step_confidence_progression=None, problem_type="quadratic"):
        """
        Initialize with a sequence of confidence levels for each step.
        
        Args:
            step_confidence_progression: Dictionary mapping step indices to confidence levels
            problem_type: Type of math problem to generate steps for
        """
        self.step_confidence_progression = step_confidence_progression or {
            0: [0.4, 0.7, 0.9],  # First step: initially low confidence, improves with iterations
            1: [0.9],            # Second step: high confidence from the start
            2: [0.3, 0.6, 0.8],  # Third step: requires multiple iterations
            3: [0.85]            # Fourth step: good confidence from the start
        }
        
        self.problem_type = problem_type
        self.reasoning_calls = 0
        self.step_verification_calls = {}  # Maps step index to number of verification calls
        self.step_regeneration_calls = {}  # Maps step index to number of regeneration calls
        
    def generate_completion(self, system_prompt, user_prompt):
        """Generate a mock completion based on the request type."""
        
        # For initial reasoning steps generation
        if "step-by-step reasoning for a math problem" in system_prompt:
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
            elif self.problem_type == "linear_algebra":
                return {"content": json.dumps([
                    "To find the eigenvalues of the matrix A, we need to solve the characteristic equation det(A - λI) = 0.",
                    "For the given 2×2 matrix, we compute det([3-λ, 1], [2, 2-λ]) = 0.",
                    "Expanding the determinant: (3-λ)(2-λ) - 1×2 = 0.",
                    "This simplifies to: 6 - 3λ - 2λ + λ² - 2 = 0.",
                    "Rearranging: λ² - 5λ + 4 = 0.",
                    "Factoring: (λ - 4)(λ - 1) = 0.",
                    "Therefore, the eigenvalues are λ = 4 and λ = 1."
                ])}
            elif self.problem_type == "probability":
                return {"content": json.dumps([
                    "To solve this Bayesian probability problem, we need to apply Bayes' theorem: P(A|B) = P(B|A)·P(A)/P(B).",
                    "Given P(Disease) = 0.01, P(Positive|Disease) = 0.95, and P(Positive|No Disease) = 0.05.",
                    "We need to find P(Disease|Positive), the probability of having the disease given a positive test result.",
                    "First, calculate P(Positive) using the law of total probability: P(Positive) = P(Positive|Disease)·P(Disease) + P(Positive|No Disease)·P(No Disease).",
                    "P(Positive) = 0.95 × 0.01 + 0.05 × 0.99 = 0.0095 + 0.0495 = 0.059.",
                    "Now apply Bayes' theorem: P(Disease|Positive) = (0.95 × 0.01) / 0.059 = 0.0095 / 0.059 ≈ 0.161 or about 16.1%."
                ])}
            else:
                return {"content": json.dumps([
                    "Step 1 of the solution process.",
                    "Step 2 of the solution process.",
                    "Step 3 of the solution process.",
                    "Step 4 of the solution process.",
                    "Final step with the answer."
                ])}
        
        # For step regeneration
        if "fixing a specific step in a mathematical solution" in system_prompt:
            step_index = None
            
            # Extract the step index from the user prompt
            for line in user_prompt.split('\n'):
                if "Step to regenerate:" in line:
                    try:
                        step_index = int(line.split('#')[1].strip()) - 1
                        break
                    except (IndexError, ValueError):
                        pass
            
            if step_index is not None:
                if step_index not in self.step_regeneration_calls:
                    self.step_regeneration_calls[step_index] = 0
                self.step_regeneration_calls[step_index] += 1
                
                # Return a corrected step based on the index and problem type
                if self.problem_type == "quadratic":
                    if step_index == 0:
                        return {"content": "First, we identify that this is a quadratic equation in the form ax² + bx + c = 0, where we need to find the values of x that make the equation true."}
                    elif step_index == 2:
                        return {"content": "Using the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a, which allows us to solve for the roots of the quadratic equation."}
                elif self.problem_type == "calculus":
                    if step_index == 0:
                        return {"content": "To find the derivative of f(x) = sin(x²), we need to apply the chain rule since we have a composition of functions."}
                    elif step_index == 2:
                        return {"content": "The derivative of sin(u) with respect to u is cos(u), according to the derivative rule for sine functions."}
                elif self.problem_type == "linear_algebra":
                    if step_index == 0:
                        return {"content": "To find the eigenvalues of the matrix A, we need to solve the characteristic equation det(A - λI) = 0, where λ represents the eigenvalues and I is the identity matrix."}
                    elif step_index == 2:
                        return {"content": "Expanding the determinant: (3-λ)(2-λ) - 1×2 = 0, which gives us the characteristic polynomial of the matrix."}
                elif self.problem_type == "probability":
                    if step_index == 0:
                        return {"content": "To solve this Bayesian probability problem, we need to apply Bayes' theorem: P(A|B) = P(B|A)·P(A)/P(B), which allows us to update our belief about an event based on new evidence."}
                    elif step_index == 3:
                        return {"content": "First, calculate P(Positive) using the law of total probability: P(Positive) = P(Positive|Disease)·P(Disease) + P(Positive|No Disease)·P(No Disease), which accounts for all possible ways to get a positive test result."}
                
                # Default corrected step
                return {"content": f"Improved and corrected step for index {step_index} in {self.problem_type} problem."}
        
        # For step verification
        if "verification agent for mathematical reasoning" in system_prompt:
            step_index = None
            
            # Extract the step number from the system prompt
            for line in system_prompt.split('\n'):
                if "verify step" in line.lower():
                    try:
                        parts = line.split("step")[1].split("of")[0].strip()
                        step_index = int(parts) - 1  # Convert to 0-indexed
                        break
                    except (IndexError, ValueError):
                        pass
            
            if step_index is not None:
                # Track verification calls for this step
                if step_index not in self.step_verification_calls:
                    self.step_verification_calls[step_index] = 0
                self.step_verification_calls[step_index] += 1
                
                # Get the confidence progression for this step
                progression = self.step_confidence_progression.get(step_index, [0.8])
                call_index = min(self.step_verification_calls[step_index] - 1, len(progression) - 1)
                confidence = progression[call_index]
                
                # Generate verification result based on confidence
                if confidence < 0.5:
                    return {"content": json.dumps({
                        "verified": False,
                        "confidence": confidence,
                        "issues": [
                            "The step lacks clarity and precision",
                            "The mathematical terminology is not used correctly",
                            "The step doesn't clearly connect to the problem"
                        ],
                        "corrected_step": "Suggested correction for step " + str(step_index + 1),
                        "explanation": "This step needs significant improvement in clarity and correctness."
                    })}
                elif confidence < 0.8:
                    return {"content": json.dumps({
                        "verified": False,
                        "confidence": confidence,
                        "issues": [
                            "The step could be more detailed to aid understanding"
                        ],
                        "corrected_step": None,
                        "explanation": "This step is mostly correct but could be improved."
                    })}
                else:
                    return {"content": json.dumps({
                        "verified": True,
                        "confidence": confidence,
                        "issues": [],
                        "corrected_step": None,
                        "explanation": "This step is correct and clear."
                    })}
        
        # Default response for other prompts
        return {"content": "Default response for unrecognized prompt type"}


def test_step_by_step_reasoning_verification(problem_type="quadratic"):
    """Test the step-by-step reasoning verification process."""
    print(f"\n--- Testing Step-by-Step Reasoning Verification: {problem_type.upper()} PROBLEM ---\n")
    
    # Create a meta-agent with a moderate confidence threshold
    meta_agent = MetaAgent(confidence_threshold=0.75, max_iterations=3)
    
    # Create a mock agent with our special LLM service
    mock_agent = MockAgent()
    mock_agent.meta_agent = meta_agent
    mock_agent.llm_service = MockLLMWithStepVerification(problem_type=problem_type)
    
    # Set the LLM service on the meta-agent as well
    meta_agent.llm_service = mock_agent.llm_service
    
    # Create the reasoning command
    reasoning_command = MathGenerateReasoningCommand(mock_agent)
    
    # Define test problems based on problem type
    if problem_type == "quadratic":
        problem = "Solve the quadratic equation: x² - 3x + 2 = 0"
        student_answer = "x = 2, x = 1"
        correct_answer = "x = 2, x = 1"
    elif problem_type == "calculus":
        problem = "Find the derivative of f(x) = sin(x²)"
        student_answer = "f'(x) = 2x·cos(x²)"
        correct_answer = "f'(x) = 2x·cos(x²)"
    elif problem_type == "linear_algebra":
        problem = "Find the eigenvalues of the matrix A = [[3, 1], [2, 2]]"
        student_answer = "λ = 4, λ = 1"
        correct_answer = "λ = 4, λ = 1"
    elif problem_type == "probability":
        problem = "A disease affects 1% of the population. A test for the disease has a 95% true positive rate and a 5% false positive rate. If a person tests positive, what is the probability they have the disease?"
        student_answer = "0.161 or 16.1%"
        correct_answer = "0.161 or 16.1%"
    else:
        problem = "Generic math problem"
        student_answer = "Generic answer"
        correct_answer = "Generic answer"
    
    print(f"Problem: {problem}")
    print(f"Student answer: {student_answer}")
    print(f"Correct answer: {correct_answer}")
    
    # Create a state object
    state = {
        "question": problem,
        "student_answer": student_answer,
        "correct_answer": correct_answer
    }
    
    # Execute the reasoning command
    result_state = reasoning_command._execute_core(state)
    
    # Display the verification results
    verification_key = "reasoning_steps_verification"
    if verification_key in result_state:
        verification_data = result_state[verification_key]
        
        print("\nOverall Verification Results:")
        print(f"Verified: {verification_data.get('overall_verified', False)}")
        print(f"Overall Confidence: {verification_data.get('overall_confidence', 0):.2f}")
        print(f"Regeneration Attempts: {verification_data.get('regeneration_attempts', 0)}")
        
        print("\nStep-by-Step Verification Results:")
        for i, step_result in enumerate(verification_data.get("step_results", [])):
            print(f"\nStep {i+1}:")
            print(f"  Verified: {step_result.get('verified', False)}")
            print(f"  Confidence: {step_result.get('confidence', 0):.2f}")
            if step_result.get("issues"):
                print(f"  Issues: {step_result.get('issues')}")
            print(f"  Explanation: {step_result.get('explanation', '')}")
    
    # Show the final reasoning steps
    print("\nFinal Reasoning Steps:")
    for i, step in enumerate(result_state.get("reasoning_steps", [])):
        print(f"Step {i+1}: {step}")
    
    # Show regeneration statistics
    llm_service = mock_agent.llm_service
    print("\nRegeneration Statistics:")
    for step_index, count in llm_service.step_regeneration_calls.items():
        print(f"Step {step_index + 1} was regenerated {count} times")
    
    return result_state


def run_all_tests():
    """Run tests for different problem types."""
    # Run the standard test
    test_step_by_step_reasoning_verification()
    
    # Add more advanced problem tests here
    # For example, you could add tests for calculus, linear algebra, etc.
    test_step_by_step_reasoning_verification(problem_type="calculus")
    test_step_by_step_reasoning_verification(problem_type="linear_algebra")
    test_step_by_step_reasoning_verification(problem_type="probability")


if __name__ == "__main__":
    run_all_tests()
