"""
Test script for the refactored reasoning command with meta agent integration.

This script demonstrates how to use the refactored MathGenerateReasoningCommand
with the MetaAgent for step verification.
"""

import sys
import os
import logging
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the required modules
from app.math_services.models.state import MathState, InteractionMode
from app.math_services.services.service_container import ServiceContainer
from app.math_services.services.llm.openai_service import OpenAILLMService
from app.math_services.agent.meta_agent import MetaAgent
from app.math_services.commands.reasoning_command import MathGenerateReasoningCommand
from app.math_services.commands.solve_command import MathSolveSymbolicallyCommand

def print_divider():
    """Print a divider line for better readability."""
    print("\n" + "=" * 80 + "\n")

def test_reasoning_with_meta_agent():
    """
    Test the reasoning command with meta agent integration.
    """
    print("Starting test of reasoning command with meta agent integration...")
    
    # 1. Create the LLM service and meta agent
    llm_service = OpenAILLMService(model="gpt-4o-mini")
    meta_agent = MetaAgent(model="gpt-4o-mini", llm_service=llm_service)
    
    # 2. Create the service container
    services = ServiceContainer(
        llm_service=llm_service,
        meta_agent=meta_agent
    )
    
    # 3. Create a test problem
    state = MathState(
        question="Solve the quadratic equation: x^2 - 5x + 6 = 0",
        student_answer="x = 2 or x = 3",
        interaction_mode=InteractionMode.STRUCTURED
    )
    
    print(f"Problem: {state.question}")
    print(f"Student answer: {state.student_answer}")
    
    # 4. First, solve the problem to get the correct answer
    print("\nSolving the problem...")
    solve_command = MathSolveSymbolicallyCommand(services)
    state = solve_command.execute(state)
    
    print(f"Correct answer: {state.correct_answer}")
    print_divider()
    
    # 5. Now generate reasoning steps with verification
    print("Generating reasoning steps with verification...")
    reasoning_command = MathGenerateReasoningCommand(services)
    state = reasoning_command.execute(state)
    
    # 6. Display the results
    print_divider()
    print("REASONING STEPS RESULTS:")
    
    # Get the steps from the state
    steps = state.context.get("reasoning", {}).get("steps", [])
    verification = state.context.get("reasoning", {}).get("verification", {})
    
    # Print the generated steps
    print(f"\nGenerated {len(steps)} steps:")
    for i, step in enumerate(steps, 1):
        print(f"Step {i}: {step}")
    
    # Print verification results
    print_divider()
    print("VERIFICATION RESULTS:")
    
    if verification:
        verified_count = len(verification.get("verified_steps", []))
        problematic_count = len(verification.get("problematic_steps", []))
        regenerated_count = len(verification.get("regenerated_steps", []))
        
        print(f"Overall verified: {verification.get('verified', False)}")
        print(f"Verified steps: {verified_count}")
        print(f"Problematic steps: {problematic_count}")
        print(f"Regenerated steps: {regenerated_count}")
        
        # Show details of any problematic steps
        if problematic_count > 0:
            print("\nProblematic steps details:")
            for step in verification.get("problematic_steps", []):
                print(f"Step {step['index'] + 1}:")
                print(f"  Verified: {step['verified']}")
                print(f"  Confidence: {step['confidence']}")
                print(f"  Issues: {', '.join(step['issues'])}")
        
        # Show details of any regenerated steps
        if regenerated_count > 0:
            print("\nRegenerated steps details:")
            for step in verification.get("regenerated_steps", []):
                print(f"Step {step['index'] + 1}:")
                print(f"  Original: {step['original']}")
                print(f"  Feedback: {step['feedback']}")
                print(f"  Regenerated: {step['regenerated']}")
    else:
        print("No verification results available")
    
    return state

if __name__ == "__main__":
    try:
        result = test_reasoning_with_meta_agent()
        print("\nTest completed successfully!")
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc() 