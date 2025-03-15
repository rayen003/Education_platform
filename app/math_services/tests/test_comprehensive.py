"""
Comprehensive Test Suite.

This module provides a comprehensive test suite that tests the end-to-end
flow of the math services, including analyzing problems, generating hints,
providing feedback, and supporting chat interactions.
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
from app.math_services.agent.meta_agent import MetaAgent
from app.math_services.services.service_container import ServiceContainer
from app.math_services.services.llm.openai_service import OpenAILLMService
from app.math_services.models.state import MathState, InteractionMode, ChatMessage
from app.math_services.commands.hint_command import MathGenerateHintCommand
from app.math_services.commands.analyze_command import MathAnalyzeCalculationCommand
from app.math_services.commands.solve_command import MathSolveSymbolicallyCommand
from app.math_services.commands.feedback_command import MathGenerateFeedbackCommand
from app.math_services.commands.reasoning_command import MathGenerateReasoningCommand
from app.math_services.commands.chat_command import MathChatFollowUpCommand

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

class ComprehensiveTestAgent:
    """Agent for comprehensive testing of all commands."""
    
    def __init__(self, model="gpt-4o-mini"):
        """Initialize the test agent with all required services."""
        self.service_container = ServiceContainer()
        self.llm_service = OpenAILLMService(model=model)
        self.meta_agent = MetaAgent(model=model, llm_service=self.llm_service)
        
        # Add services to the container
        self.service_container.add_service("llm_service", self.llm_service)
        self.service_container.add_service("meta_agent", self.meta_agent)
        
        # Initialize commands
        self.analyze_command = MathAnalyzeCalculationCommand(self.service_container)
        self.solve_command = MathSolveSymbolicallyCommand(self.service_container)
        self.hint_command = MathGenerateHintCommand(self.service_container)
        self.feedback_command = MathGenerateFeedbackCommand(self.service_container)
        self.reasoning_command = MathGenerateReasoningCommand(self.service_container)
        self.chat_command = MathChatFollowUpCommand(self.service_container)
    
    def analyze(self, state: MathState) -> MathState:
        """Analyze a math problem."""
        return self.analyze_command.execute(state)
    
    def solve(self, state: MathState) -> MathState:
        """Solve a math problem symbolically."""
        return self.solve_command.execute(state)
    
    def generate_hint(self, state: MathState) -> MathState:
        """Generate a hint."""
        return self.hint_command.execute(state)
    
    def generate_feedback(self, state: MathState) -> MathState:
        """Generate feedback."""
        return self.feedback_command.execute(state)
    
    def generate_reasoning(self, state: MathState) -> MathState:
        """Generate reasoning steps."""
        return self.reasoning_command.execute(state)
    
    def handle_chat_followup(self, state: MathState) -> MathState:
        """Handle a chat follow-up question."""
        # Extract the last user message as the follow-up question
        follow_up_question = ""
        if state.chat_history:
            for msg in reversed(state.chat_history):
                if msg.role == "user":
                    follow_up_question = msg.message
                    break
        
        if not follow_up_question:
            raise ValueError("No user message found in chat history to use as follow-up question")
            
        return self.chat_command.execute(state, follow_up_question)

def test_algebra_problem_structured_mode():
    """Test an algebra problem in structured mode."""
    print("\nTesting algebra problem in structured mode...")
    
    # Create a test agent
    agent = ComprehensiveTestAgent()
    
    # Create a state with an algebra problem
    state = MathState(
        question="Solve for x: 3x - 7 = 5",
        student_answer="x = 4",
        correct_answer="x = 4",
        interaction_mode=InteractionMode.STRUCTURED,
        chat_history=[]
    )
    
    try:
        # Process the problem
        print("Step 1: Analyzing student's answer...")
        state = agent.analyze(state)
        if state.analysis:
            print(f"Analysis: {'Correct' if state.analysis.is_correct else 'Incorrect'}")
            
        print("\nStep 2: Generating correct solution...")
        state = agent.solve(state)
        if state.correct_answer:
            print(f"Correct solution: {state.correct_answer[:100]}...")
            
        print("\nStep 3: Generating feedback...")
        state = agent.generate_feedback(state)
        if state.feedback:
            print(f"Feedback: {state.feedback}")
            
        print("\nStep 4: Generating hint...")
        state = agent.generate_hint(state)
        if state.hints:
            print(f"Hint: {state.hints[-1]}")
            
        print("\nStep 5: Generating reasoning steps...")
        state = agent.generate_reasoning(state)
        if state.context and "reasoning" in state.context and "steps" in state.context["reasoning"]:
            steps = state.context["reasoning"]["steps"]
            print(f"Generated {len(steps)} reasoning steps:")
            for i, step in enumerate(steps):
                print(f"  Step {i+1}: {step[:100]}...")
            
        return state
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return state

def test_geometry_problem_structured_mode():
    """Test a geometry problem in structured mode."""
    print("\nTesting geometry problem in structured mode...")
    
    # Create a test agent
    agent = ComprehensiveTestAgent()
    
    # Create a state with a geometry problem
    state = MathState(
        question="Find the area of a circle with radius 5 cm.",
        student_answer="A = 25π cm^2",
        correct_answer="A = 25π cm^2",
        interaction_mode=InteractionMode.STRUCTURED,
        chat_history=[]
    )
    
    try:
        # Process the problem
        print("Step 1: Analyzing student's answer...")
        state = agent.analyze(state)
        if state.analysis:
            print(f"Analysis: {'Correct' if state.analysis.is_correct else 'Incorrect'}")
            
        print("\nStep 2: Generating correct solution...")
        state = agent.solve(state)
        if state.correct_answer:
            print(f"Correct solution: {state.correct_answer[:100]}...")
            
        print("\nStep 3: Generating feedback...")
        state = agent.generate_feedback(state)
        if state.feedback:
            print(f"Feedback: {state.feedback}")
            
        print("\nStep 4: Generating reasoning steps...")
        state = agent.generate_reasoning(state)
        if state.context and "reasoning" in state.context and "steps" in state.context["reasoning"]:
            steps = state.context["reasoning"]["steps"]
            print(f"Generated {len(steps)} reasoning steps:")
            for i, step in enumerate(steps):
                print(f"  Step {i+1}: {step[:100]}...")
            
        return state
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return state

def test_calculus_problem_chat_mode():
    """Test a calculus problem in chat mode."""
    print("\nTesting calculus problem in chat mode...")
    
    # Create a test agent
    agent = ComprehensiveTestAgent()
    
    # Create a state with a calculus problem in chat mode
    state = MathState(
        question="Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3",
        student_answer="I'm not sure how to approach this problem.",
        interaction_mode=InteractionMode.CHAT,
        chat_history=[
            ChatMessage(role="user", message="Can you help me find the derivative of f(x) = x^3 + 2x^2 - 5x + 3?"),
            ChatMessage(role="tutor", message="Sure, I'd be happy to help! Do you know how to apply the power rule?"),
            ChatMessage(role="user", message="I'm not sure how to approach this problem.")
        ]
    )
    
    try:
        # Process the problem in chat mode
        print("Step 1: Handling the chat follow-up...")
        state = agent.handle_chat_followup(state)
        if state.chat_history:
            print("\nChat history:")
            for msg in state.chat_history:
                print(f"{msg.role}: {msg.message[:100]}...")
                
        print("\nStep 2: Asking a more specific follow-up question...")
        state.chat_history.append(ChatMessage(
            role="user",
            message="Can you explain the power rule in more detail?"
        ))
        state = agent.handle_chat_followup(state)
        if state.chat_history:
            print("\nUpdated chat history:")
            for msg in state.chat_history[-2:]:  # Just show the latest exchange
                print(f"{msg.role}: {msg.message[:100]}...")
                
        print("\nStep 3: Asking for an example...")
        state.chat_history.append(ChatMessage(
            role="user",
            message="Can you work out the first term in this example?"
        ))
        state = agent.handle_chat_followup(state)
        if state.chat_history:
            print("\nFinal chat history:")
            for msg in state.chat_history[-2:]:  # Just show the latest exchange
                print(f"{msg.role}: {msg.message[:100]}...")
            
        return state
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return state

def test_logic_problem_mode_switching():
    """Test a logic problem with mode switching."""
    print("\nTesting logic problem with mode switching...")
    
    # Create a test agent
    agent = ComprehensiveTestAgent()
    
    # Create a state with a logic problem in structured mode
    state = MathState(
        question="If all A are B, and all B are C, what can we conclude about the relationship between A and C?",
        student_answer="Nothing can be concluded",
        correct_answer="All A are C",
        interaction_mode=InteractionMode.STRUCTURED,
        chat_history=[]
    )
    
    try:
        # Process the problem in structured mode first
        print("Step 1: Analyzing student's answer in structured mode...")
        state = agent.analyze(state)
        if state.analysis:
            print(f"Analysis: {'Correct' if state.analysis.is_correct else 'Incorrect'}")
            
        print("\nStep 2: Generating hint in structured mode...")
        state = agent.generate_hint(state)
        if state.hints:
            print(f"Hint: {state.hints[-1]}")
            
        print("\nStep 3: Switching to chat mode...")
        state.interaction_mode = InteractionMode.CHAT
        state.chat_history.append(ChatMessage(
            role="user",
            message="I still don't understand why 'nothing can be concluded' is wrong. Can you explain?"
        ))
        state = agent.handle_chat_followup(state)
        if state.chat_history:
            print("\nChat response:")
            print(f"tutor: {state.chat_history[-1].message[:200]}...")
            
        print("\nStep 4: Continuing the chat conversation...")
        state.chat_history.append(ChatMessage(
            role="user",
            message="Can you give me a real-world example that illustrates this logic?"
        ))
        state = agent.handle_chat_followup(state)
        if state.chat_history:
            print("\nChat response:")
            print(f"tutor: {state.chat_history[-1].message[:200]}...")
            
        print("\nStep 5: Switching back to structured mode...")
        state.interaction_mode = InteractionMode.STRUCTURED
        print("Generating reasoning steps...")
        state = agent.generate_reasoning(state)
        if state.context and "reasoning" in state.context and "steps" in state.context["reasoning"]:
            steps = state.context["reasoning"]["steps"]
            print(f"Generated {len(steps)} reasoning steps:")
            for i, step in enumerate(steps):
                print(f"  Step {i+1}: {step[:100]}...")
            
        return state
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return state

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("\n=== Running Comprehensive Tests ===\n")
    
    tests = [
        ("Algebra Problem (Structured Mode)", test_algebra_problem_structured_mode),
        ("Geometry Problem (Structured Mode)", test_geometry_problem_structured_mode),
        ("Calculus Problem (Chat Mode)", test_calculus_problem_chat_mode),
        ("Logic Problem (Mode Switching)", test_logic_problem_mode_switching)
    ]
    
    results = {}
    
    for name, test_func in tests:
        print_divider()
        print(f"RUNNING TEST: {name}")
        print_divider()
        
        try:
            state = test_func()
            success = True
        except Exception as e:
            print(f"Test failed with exception: {e}")
            traceback.print_exc()
            success = False
            
        results[name] = "✓ PASSED" if success else "❌ FAILED"
    
    # Print summary
    print_divider()
    print("TEST SUMMARY:")
    for name, result in results.items():
        print(f"{name}: {result}")
    print_divider()
    
    return results

if __name__ == "__main__":
    run_comprehensive_tests() 