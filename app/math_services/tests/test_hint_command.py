"""
Test Hint Command.

This module tests the hint generation functionality with different types of questions
(logical, numerical, mixed) and in different scenarios (chat mode vs. structured mode).
"""

import logging
import sys
import os
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Import the required modules
from app.math_services.commands.hint_command import MathGenerateHintCommand
from app.math_services.agent.meta_agent import MetaAgent
from app.math_services.services.service_container import ServiceContainer
from app.math_services.services.llm.openai_service import OpenAILLMService
from app.math_services.models.state import MathState, InteractionMode, ChatMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def print_divider():
    """Print a divider line."""
    print("\n" + "=" * 80 + "\n")

class TestAgent:
    """Simple agent for testing."""
    
    def __init__(self, model="gpt-4o-mini"):
        """Initialize the test agent."""
        self.service_container = ServiceContainer()
        self.llm_service = OpenAILLMService(model=model)
        self.meta_agent = MetaAgent(model=model, llm_service=self.llm_service)
        
        # Add services to the container
        self.service_container.add_service("llm_service", self.llm_service)
        self.service_container.add_service("meta_agent", self.meta_agent)
    
    def generate_hint(self, state: MathState) -> MathState:
        """Generate a hint based on the current state."""
        # Initialize the hint command
        hint_command = MathGenerateHintCommand(self.service_container)
        
        # Execute the hint command
        return hint_command.execute(state)

def test_numerical_problem():
    """Test hint generation for a numerical problem."""
    print("\nTesting hint generation for numerical problem...")
    
    # Create a test agent
    agent = TestAgent()
    
    # Create a state with a numerical problem
    state = MathState(
        question="Solve the equation: 2x + 5 = 15",
        student_answer="x = 4",
        correct_answer="x = 5",
        interaction_mode=InteractionMode.STRUCTURED,
        chat_history=[]
    )
    
    try:
        # Generate a hint
        result_state = agent.generate_hint(state)
        
        # Display the hint
        if result_state.hints:
            print(f"\nHint: {result_state.hints[-1]}")
            if hasattr(result_state, "context") and "hint_verification" in result_state.context:
                print(f"\nVerification: {result_state.context['hint_verification']}")
            return True
        else:
            print("❌ No hint generated")
            return False
    except Exception as e:
        print(f"❌ Error generating hint: {e}")
        traceback.print_exc()
        return False

def test_logical_problem():
    """Test hint generation for a logical problem."""
    print("\nTesting hint generation for logical problem...")
    
    # Create a test agent
    agent = TestAgent()
    
    # Create a state with a logical problem
    state = MathState(
        question="If all A are B, and all B are C, what can we conclude about the relationship between A and C?",
        student_answer="Nothing can be concluded",
        correct_answer="All A are C",
        interaction_mode=InteractionMode.STRUCTURED,
        chat_history=[]
    )
    
    try:
        # Generate a hint
        result_state = agent.generate_hint(state)
        
        # Display the hint
        if result_state.hints:
            print(f"\nHint: {result_state.hints[-1]}")
            if hasattr(result_state, "context") and "hint_verification" in result_state.context:
                print(f"\nVerification: {result_state.context['hint_verification']}")
            return True
        else:
            print("❌ No hint generated")
            return False
    except Exception as e:
        print(f"❌ Error generating hint: {e}")
        traceback.print_exc()
        return False

def test_mixed_problem():
    """Test hint generation for a mixed (logical and numerical) problem."""
    print("\nTesting hint generation for mixed problem...")
    
    # Create a test agent
    agent = TestAgent()
    
    # Create a state with a mixed problem
    state = MathState(
        question="A train leaves Station A at 3:00 PM traveling at 60 mph. Another train leaves Station B at 4:00 PM traveling at 90 mph in the opposite direction. If the stations are 300 miles apart, at what time will the trains meet?",
        student_answer="6:00 PM",
        correct_answer="6:30 PM",
        interaction_mode=InteractionMode.STRUCTURED,
        chat_history=[]
    )
    
    try:
        # Generate a hint
        result_state = agent.generate_hint(state)
        
        # Display the hint
        if result_state.hints:
            print(f"\nHint: {result_state.hints[-1]}")
            if hasattr(result_state, "context") and "hint_verification" in result_state.context:
                print(f"\nVerification: {result_state.context['hint_verification']}")
            return True
        else:
            print("❌ No hint generated")
            return False
    except Exception as e:
        print(f"❌ Error generating hint: {e}")
        traceback.print_exc()
        return False

def test_chat_mode_hint():
    """Test hint generation in chat mode."""
    print("\nTesting hint generation in chat mode...")
    
    # Create a test agent
    agent = TestAgent()
    
    # Create a state with chat mode
    state = MathState(
        question="Solve the quadratic equation: x^2 - 4x + 4 = 0",
        student_answer="I tried factoring but I'm stuck",
        interaction_mode=InteractionMode.CHAT,
        chat_history=[
            ChatMessage(role="user", message="I need help with this quadratic equation: x^2 - 4x + 4 = 0"),
            ChatMessage(role="tutor", message="What have you tried so far?"),
            ChatMessage(role="user", message="I tried factoring but I'm stuck")
        ]
    )
    
    try:
        # Generate a hint
        result_state = agent.generate_hint(state)
        
        # Display the hint
        if result_state.hints:
            print(f"\nHint: {result_state.hints[-1]}")
            print(f"\nChat History:")
            for msg in result_state.chat_history:
                print(f"{msg.role}: {msg.message}")
            return True
        else:
            print("❌ No hint generated")
            return False
    except Exception as e:
        print(f"❌ Error generating hint: {e}")
        traceback.print_exc()
        return False

def test_progressive_hints():
    """Test generation of progressive hints (multiple hints)."""
    print("\nTesting progressive hint generation...")
    
    # Create a test agent
    agent = TestAgent()
    
    # Create a state
    state = MathState(
        question="Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3",
        student_answer="I'm not sure how to start",
        correct_answer="f'(x) = 3x^2 + 4x - 5",
        interaction_mode=InteractionMode.STRUCTURED,
        chat_history=[]
    )
    
    try:
        # Generate multiple hints
        for i in range(3):
            print(f"\nGenerating hint #{i+1}...")
            state = agent.generate_hint(state)
            
            # Display the hint
            if state.hints:
                print(f"\nHint #{i+1}: {state.hints[-1]}")
            else:
                print(f"❌ No hint #{i+1} generated")
                return False
        return True
    except Exception as e:
        print(f"❌ Error generating progressive hints: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests."""
    print("\n=== Testing Hint Command with Different Problem Types and Scenarios ===\n")
    
    tests = [
        ("Numerical Problem", test_numerical_problem),
        ("Logical Problem", test_logical_problem),
        ("Mixed Problem", test_mixed_problem),
        ("Chat Mode Hint", test_chat_mode_hint),
        ("Progressive Hints", test_progressive_hints)
    ]
    
    results = {}
    
    for name, test_func in tests:
        print_divider()
        print(f"RUNNING TEST: {name}")
        print_divider()
        
        success = test_func()
        results[name] = "✓ PASSED" if success else "❌ FAILED"
    
    # Print summary
    print_divider()
    print("TEST SUMMARY:")
    for name, result in results.items():
        print(f"{name}: {result}")
    print_divider()

if __name__ == "__main__":
    run_all_tests() 