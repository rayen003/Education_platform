"""
Test script for the refactored MathAgent with data classes and dependency injection.

This script demonstrates how to use the new MathState data class and ServiceContainer
for dependency injection in the math services.
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
from app.math_services.models.state import MathState, InteractionMode, ChatMessage
from app.math_services.services.service_container import ServiceContainer
from app.math_services.services.llm.openai_service import OpenAILLMService
from app.math_services.agent.meta_agent import MetaAgent
from app.math_services.commands.solve_command import MathSolveSymbolicallyCommand
from app.math_services.commands.analyze_command import MathAnalyzeCalculationCommand
from app.math_services.commands.chat_command import MathChatFollowUpCommand

def test_refactored_agent():
    """
    Test the refactored agent with data classes and dependency injection.
    """
    # 1. Create the LLM service
    llm_service = OpenAILLMService(model="gpt-4o-mini")
    
    # 2. Create the meta agent
    meta_agent = MetaAgent(model="gpt-4o-mini", llm_service=llm_service)
    
    # 3. Create the service container
    services = ServiceContainer(
        llm_service=llm_service,
        meta_agent=meta_agent
    )
    
    # 4. Create a MathState instance
    state = MathState(
        question="Solve for x: 2x + 5 = 13",
        student_answer="x = 4",
        interaction_mode=InteractionMode.STRUCTURED
    )
    
    print(f"Initial state: {state}")
    
    # 5. Create and execute the solve command
    solve_command = MathSolveSymbolicallyCommand(services)
    state = solve_command.execute(state)
    
    print(f"After solve: {state.correct_answer}")
    
    # 6. Create and execute the analyze command
    analyze_command = MathAnalyzeCalculationCommand(services)
    state = analyze_command.execute(state)
    
    print(f"Analysis results - Is correct: {state.analysis.is_correct}")
    print(f"Analysis results - Error type: {state.analysis.error_type}")
    print(f"Analysis results - Misconception: {state.analysis.misconception}")
    
    # 7. Test chat mode
    state.interaction_mode = InteractionMode.CHAT
    
    # Add student message
    student_message = ChatMessage(
        role="student",
        message="I don't understand why my answer is wrong. Can you explain?",
        timestamp=datetime.now()
    )
    state.chat_history.append(student_message)
    
    # Create and execute the chat command
    chat_command = MathChatFollowUpCommand(services)
    state = chat_command.execute(state, student_message.message)
    
    print("\nChat response:")
    print(state.chat_response)
    
    # Convert to dictionary for backward compatibility testing
    state_dict = state.to_dict()
    print("\nConverted back to dictionary:")
    print(f"Keys: {list(state_dict.keys())}")
    
    # Test converting back to MathState
    new_state = MathState.from_dict(state_dict)
    print(f"\nConverted back to MathState: {new_state.interaction_mode}")
    
    return state

if __name__ == "__main__":
    try:
        result = test_refactored_agent()
        print("\nTest completed successfully!")
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc() 