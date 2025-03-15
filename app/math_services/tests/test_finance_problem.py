"""
Test script for a complex college-level finance problem.

This script demonstrates how the step-by-step reasoning verification system
handles advanced mathematical concepts in finance.
"""

import os
import json
import logging
from dotenv import load_dotenv

from app.math_services.agent.meta_agent import MetaAgent
from app.math_services.commands.reasoning_command import MathGenerateReasoningCommand
from app.math_services.commands.hint_command import MathGenerateHintCommand
from app.math_services.tests.mock_agent import MockAgent

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MockLLMForFinanceProblems:
    """Mock LLM service that simulates complex finance problem solving."""
    
    def __init__(self):
        """Initialize the mock LLM service."""
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
            
            # Return progressive hints for the finance problem
            if hint_count == 1:
                return {"content": "Consider using the Black-Scholes model for option pricing. What variables do you need to calculate the price of a European call option?"}
            elif hint_count == 2:
                return {"content": "The Black-Scholes formula requires five key inputs: the current stock price (S), the option strike price (K), time to expiration (T), risk-free interest rate (r), and stock price volatility (σ)."}
            elif hint_count == 3:
                return {"content": "The formula for a European call option is C = S·N(d₁) - K·e^(-rT)·N(d₂), where N(x) is the cumulative distribution function of the standard normal distribution."}
            else:
                return {"content": "To calculate d₁ and d₂: d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ·√T) and d₂ = d₁ - σ·√T. Then apply these values to the Black-Scholes formula."}
        
        # For step-by-step reasoning generation
        elif "step-by-step reasoning for a math problem" in system_prompt:
            self.reasoning_calls += 1
            
            # Return detailed steps for the Black-Scholes model
            return {"content": json.dumps([
                "First, we identify that this is an option pricing problem that requires the Black-Scholes model for European call options.",
                "The Black-Scholes formula for a European call option is: C = S·N(d₁) - K·e^(-rT)·N(d₂)",
                "We need to calculate d₁ and d₂ using: d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ·√T) and d₂ = d₁ - σ·√T",
                "Given S = $50, K = $52, T = 0.5 years, r = 0.05 (5%), and σ = 0.3 (30%), we substitute these values.",
                "For d₁: d₁ = [ln(50/52) + (0.05 + 0.3²/2)·0.5] / (0.3·√0.5) = [-0.0392 + (0.05 + 0.045)·0.5] / (0.3·0.7071) = [-0.0392 + 0.0475] / 0.2121 = 0.0083 / 0.2121 = 0.0391",
                "For d₂: d₂ = 0.0391 - 0.3·√0.5 = 0.0391 - 0.2121 = -0.173",
                "Now we need to find N(d₁) and N(d₂), which are the cumulative distribution functions of the standard normal distribution.",
                "Using statistical tables or calculators: N(0.0391) ≈ 0.5156 and N(-0.173) ≈ 0.4313",
                "Substituting into the Black-Scholes formula: C = 50·0.5156 - 52·e^(-0.05·0.5)·0.4313",
                "Simplifying: C = 25.78 - 52·0.9753·0.4313 = 25.78 - 21.82 = $3.96",
                "Therefore, the price of the European call option is $3.96."
            ])}
        
        # For verification
        elif "verification agent" in system_prompt:
            self.verification_calls += 1
            
            # Simulate verification with different confidence levels for different steps
            if "step 5" in user_prompt.lower() or "step 6" in user_prompt.lower():
                # Simulate a calculation error in steps 5 and 6
                return {"content": json.dumps({
                    "verified": False,
                    "confidence": 60,
                    "reasoning_quality": 65,
                    "issues": ["There appears to be a calculation error in the computation of d₁ or d₂", 
                              "The logarithm calculation needs to be double-checked"],
                    "explanation": "The calculation contains errors in the application of the Black-Scholes formula."
                })}
            else:
                # Other steps are correct
                return {"content": json.dumps({
                    "verified": True,
                    "confidence": 90,
                    "reasoning_quality": 85,
                    "issues": [],
                    "explanation": "This step is mathematically correct and clearly explained."
                })}
        
        # For regeneration of specific steps
        elif "fixing a specific step" in system_prompt:
            self.regeneration_calls += 1
            
            # Return corrected steps for steps 5 and 6
            if "step 5" in user_prompt.lower():
                return {"content": "For d₁: d₁ = [ln(50/52) + (0.05 + 0.3²/2)·0.5] / (0.3·√0.5) = [-0.0392 + (0.05 + 0.045)·0.5] / (0.3·0.7071) = [-0.0392 + 0.0475] / 0.2121 = 0.0083 / 0.2121 = 0.0391"}
            elif "step 6" in user_prompt.lower():
                return {"content": "For d₂: d₂ = 0.0391 - 0.3·√0.5 = 0.0391 - 0.2121 = -0.173"}
            else:
                return {"content": "Regenerated step with corrections."}
        
        # Default response for other prompts
        return {"content": "Default response for unrecognized prompt type"}


def test_finance_problem():
    """Test the step-by-step reasoning verification with a complex finance problem."""
    print("\n--- Testing Step-by-Step Reasoning: ADVANCED FINANCE PROBLEM ---\n")
    
    # Create a meta-agent with a lower confidence threshold to trigger regeneration
    meta_agent = MetaAgent(confidence_threshold=0.75, max_iterations=3)
    
    # Create a mock agent with our special LLM service
    mock_agent = MockAgent()
    mock_agent.meta_agent = meta_agent
    mock_agent.llm_service = MockLLMForFinanceProblems()
    
    # Set the LLM service on the meta-agent as well
    meta_agent.llm_service = mock_agent.llm_service
    
    # Create the commands
    reasoning_command = MathGenerateReasoningCommand(mock_agent)
    hint_command = MathGenerateHintCommand(mock_agent)
    
    # Define the finance problem
    problem = "Calculate the price of a European call option using the Black-Scholes model with the following parameters: current stock price (S) = $50, strike price (K) = $52, time to expiration (T) = 0.5 years, risk-free interest rate (r) = 5%, and volatility (σ) = 30%."
    student_answer = "I know I need to use the Black-Scholes formula, but I'm not sure how to apply it with these parameters."
    correct_answer = "$3.96"
    
    print(f"Problem: {problem}")
    print(f"Student answer: {student_answer}")
    print(f"Correct answer: {correct_answer}")
    
    # Create a state object
    state = {
        "question": problem,
        "student_answer": student_answer,
        "correct_answer": correct_answer,
        "analysis": {
            "approach": "partial",
            "misconceptions": ["formula application"],
            "completion_level": "minimal"
        }
    }
    
    # Generate step-by-step reasoning
    print("\n1. Generating Step-by-Step Reasoning:")
    state = reasoning_command._execute_core(state)
    
    # Display the reasoning steps
    print("\nInitial Reasoning Steps:")
    for i, step in enumerate(state.get("reasoning_steps", [])):
        print(f"Step {i+1}: {step}")
    
    # Display verification results
    if "verification_results" in state:
        print("\nVerification Results:")
        for step_num, result in state.get("verification_results", {}).items():
            verified = result.get("verified", False)
            confidence = result.get("confidence", 0)
            issues = result.get("issues", [])
            
            status = "✓" if verified else "✗"
            print(f"Step {step_num}: {status} (Confidence: {confidence:.2f})")
            
            if issues:
                print(f"  Issues: {', '.join(issues)}")
    
    # Generate hints
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


if __name__ == "__main__":
    test_finance_problem()
