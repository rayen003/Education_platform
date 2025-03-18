"""
Interactive Math Agent Demo

This script demonstrates the event-driven orchestration and chain of draft capabilities
of the math agent through various simulated user interactions.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import re

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import required modules
from app.math_services.agent.math_agent import MathAgent
from app.math_services.models.state import MathState, InteractionMode, ChatMessage, UserAction

# ANSI color codes for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Demonstration scenarios
SCENARIOS = {
    "basic": {
        "name": "Basic Structured Interaction",
        "description": "Submit a problem, get an answer, receive feedback",
        "steps": [
            ("problem", "Solve for x: 2x + 7 = 15"),
            ("answer", "x = 4"),
            ("hint", None),
        ]
    },
    "step_by_step": {
        "name": "Step-by-Step Guidance",
        "description": "User who needs detailed guidance with multiple hints",
        "steps": [
            ("problem", "Solve the quadratic equation: x^2 - 5x + 6 = 0"),
            ("hint", None), 
            ("answer", "x = 3 or x = 2"),
            ("reasoning", "detailed"),
        ]
    },
    "chat_mode": {
        "name": "Chat-Based Interaction",
        "description": "Natural conversation with follow-up questions",
        "steps": [
            ("problem", "I need to find the derivative of f(x) = x^3 - 4x^2 + 2x"),
            ("toggle", None),
            ("followup", "How do I find the critical points?"),
            ("followup", "What is the second derivative?"),
            ("solution", None),
        ]
    },
    "correction": {
        "name": "Error Correction and Feedback",
        "description": "User submits incorrect answers and receives guidance",
        "steps": [
            ("problem", "If a triangle has sides of length 3, 4, and 5, what is its area?"),
            ("answer", "15"),
            ("hint", None),
            ("answer", "6"),
            ("solution", None),
        ]
    },
    "nlp_inference": {
        "name": "Natural Language Processing",
        "description": "System infers actions from natural language inputs",
        "steps": [
            ("natural", "I need to solve 3x + 2 = 14"),
            ("natural", "I think the answer is x = 4"),
            ("natural", "Can you give me a hint?"),
            ("natural", "I'm still stuck. What's the detailed solution?"),
            ("natural", "Can you explain coefficients?"),
            ("natural", "Switch to chat mode please"),
            ("natural", "Let's start over with a new problem"),
        ]
    },
    "cod_vs_cot": {
        "name": "Chain of Draft vs Chain of Thought",
        "description": "Demonstrating the difference between CoD and CoT reasoning",
        "steps": [
            ("problem", "Find the indefinite integral of 2x^3 - 3x^2 + 4x - 1"),
            ("toggle", None),
            ("followup", "Can you show me your reasoning using Chain of Draft?"),
            ("followup", "Now show me detailed Chain of Thought reasoning"),
        ]
    }
}

def print_divider(title=None):
    """Print a divider with optional title."""
    width = 80
    if title:
        padding = (width - len(title) - 4) // 2
        divider = "=" * padding + f" {title} " + "=" * padding
        # Adjust if odd length
        if len(divider) < width:
            divider += "="
    else:
        divider = "=" * width
    
    print(f"\n{Colors.BOLD}{divider}{Colors.END}\n")

def print_state_summary(state: MathState):
    """Print a summary of the current state."""
    print(f"{Colors.BOLD}Current State Summary:{Colors.END}")
    print(f"  Question: {state.question[:50]}{'...' if len(state.question) > 50 else ''}")
    print(f"  Student Answer: {state.student_answer[:30]}{'...' if len(state.student_answer) > 30 else ''}")
    print(f"  Is Correct: {state.analysis.is_correct}")
    print(f"  Interaction Mode: {state.interaction_mode.value}")
    print(f"  Hint Count: {state.hint_count}")
    print(f"  Steps Count: {len(state.steps)}")
    print(f"  Chat History: {len(state.chat_history)} messages\n")

def print_chat_history(state: MathState):
    """Print the chat history."""
    if not state.chat_history:
        print("  No chat history yet.")
        return
    
    print(f"{Colors.BOLD}Chat History:{Colors.END}")
    for msg in state.chat_history:
        if msg.role == "student":
            prefix = f"{Colors.BLUE}Student:{Colors.END}"
        elif msg.role == "tutor":
            prefix = f"{Colors.GREEN}Tutor:{Colors.END}"
        else:
            prefix = f"{Colors.YELLOW}System:{Colors.END}"
        
        # Format timestamp
        timestamp = msg.timestamp.strftime("%H:%M:%S")
        
        print(f"  [{timestamp}] {prefix} {msg.message}")
    print()

def simulate_interaction(agent: MathAgent, scenario_key: str):
    """Simulate a user interaction based on a scenario."""
    scenario = SCENARIOS.get(scenario_key)
    if not scenario:
        print(f"Unknown scenario: {scenario_key}")
        return
    
    print_divider(scenario["name"])
    print(f"{Colors.BOLD}Description:{Colors.END} {scenario['description']}\n")
    
    # Initialize state
    state = MathState(question="", student_answer="")
    
    # Process each step
    for step_num, (action_type, content) in enumerate(scenario["steps"], 1):
        print(f"{Colors.YELLOW}Step {step_num}: {action_type.title()}{Colors.END}")
        
        if action_type == "problem":
            print(f"{Colors.BLUE}User:{Colors.END} {content}")
            state = agent.process_action(state, "submit_problem", text=content)
            if state.correct_answer:
                print(f"{Colors.GREEN}Agent:{Colors.END} I'll solve this problem for you.")
                print(f"{Colors.GREEN}Agent:{Colors.END} The answer is: {state.correct_answer}")
        
        elif action_type == "answer":
            print(f"{Colors.BLUE}User:{Colors.END} My answer is {content}")
            state = agent.process_action(state, "submit_answer", answer=content)
            if state.feedback:
                feedback_content = ""
                # Check if feedback is a MathFeedback object or a dictionary
                if hasattr(state.feedback, 'assessment'):
                    # It's a MathFeedback object
                    feedback_content = state.feedback.assessment
                elif isinstance(state.feedback, dict) and "math" in state.feedback and "assessment" in state.feedback["math"]:
                    # It's the old dictionary format
                    feedback_content = state.feedback["math"]["assessment"]
                else:
                    feedback_content = str(state.feedback)
                
                print(f"{Colors.GREEN}Agent:{Colors.END} {feedback_content}")
        
        elif action_type == "hint":
            print(f"{Colors.BLUE}User:{Colors.END} Can I get a hint?")
            state = agent.process_action(state, "request_hint")
            if state.hints:
                print(f"{Colors.GREEN}Agent:{Colors.END} Hint: {state.hints[-1]}")
        
        elif action_type == "solution":
            print(f"{Colors.BLUE}User:{Colors.END} Please show me the solution.")
            state = agent.process_action(state, "request_solution")
            if state.steps:
                print(f"{Colors.GREEN}Agent:{Colors.END} Here's the solution:")
                for i, step in enumerate(state.steps):
                    print(f"  Step {i+1}: {step}")
        
        elif action_type == "toggle":
            old_mode = state.interaction_mode.value
            print(f"{Colors.BLUE}User:{Colors.END} Switch to {'chat' if old_mode == 'structured' else 'structured'} mode.")
            state = agent.process_action(state, "toggle_mode")
            print(f"{Colors.GREEN}Agent:{Colors.END} Switched from {old_mode} to {state.interaction_mode.value} mode.")
        
        elif action_type == "followup":
            print(f"{Colors.BLUE}User:{Colors.END} {content}")
            state = agent.process_action(state, "ask_followup", text=content)
            if state.chat_response:
                print(f"{Colors.GREEN}Agent:{Colors.END} {state.chat_response}")
        
        elif action_type == "reasoning":
            print(f"{Colors.BLUE}User:{Colors.END} Show me your reasoning{'in detail' if content == 'detailed' else ''}.")
            state = agent.process_action(state, "request_reasoning", use_cot=content == "detailed")
            if state.steps:
                print(f"{Colors.GREEN}Agent:{Colors.END} Here's my reasoning:")
                for i, step in enumerate(state.steps):
                    print(f"  Step {i+1}: {step}")
        
        elif action_type == "natural":
            print(f"{Colors.BLUE}User:{Colors.END} {content}")
            state = agent.process_natural_input(content, state)
            
            # Display appropriate response based on the inferred action
            if state.last_action == UserAction.SUBMIT_PROBLEM:
                if state.correct_answer:
                    print(f"{Colors.GREEN}Agent:{Colors.END} I'll solve this problem for you.")
                    print(f"{Colors.GREEN}Agent:{Colors.END} The answer is: {state.correct_answer}")
            elif state.last_action == UserAction.SUBMIT_ANSWER:
                if state.feedback:
                    feedback_content = ""
                    # Check if feedback is a MathFeedback object or a dictionary
                    if hasattr(state.feedback, 'assessment'):
                        # It's a MathFeedback object
                        feedback_content = state.feedback.assessment
                    elif isinstance(state.feedback, dict) and "math" in state.feedback and "assessment" in state.feedback["math"]:
                        # It's the old dictionary format
                        feedback_content = state.feedback["math"]["assessment"]
                    else:
                        feedback_content = str(state.feedback)
                    print(f"{Colors.GREEN}Agent:{Colors.END} {feedback_content}")
            elif state.last_action == UserAction.REQUEST_HINT and state.hints:
                print(f"{Colors.GREEN}Agent:{Colors.END} Hint: {state.hints[-1]}")
            elif state.last_action == UserAction.REQUEST_SOLUTION and state.steps:
                print(f"{Colors.GREEN}Agent:{Colors.END} Here's the solution:")
                for i, step in enumerate(state.steps):
                    print(f"  Step {i+1}: {step}")
            elif state.last_action == UserAction.REQUEST_EXPLANATION and state.chat_response:
                print(f"{Colors.GREEN}Agent:{Colors.END} {state.chat_response}")
            elif state.last_action == UserAction.TOGGLE_MODE:
                print(f"{Colors.GREEN}Agent:{Colors.END} Switched to {state.interaction_mode.value} mode.")
            elif state.last_action == UserAction.RESET:
                print(f"{Colors.GREEN}Agent:{Colors.END} Let's start over with a fresh slate.")
            elif state.chat_response:  # Fallback for other actions
                print(f"{Colors.GREEN}Agent:{Colors.END} {state.chat_response}")
        
        else:
            print(f"Unknown action type: {action_type}")
        
        # Add a small delay to make the demo more readable
        time.sleep(0.5)
        
        # Print state summary after each step
        if state.interaction_mode == InteractionMode.CHAT:
            print_chat_history(state)
        else:
            print_state_summary(state)
        
        # Add separator between steps
        print(f"\n{Colors.BOLD}{'-' * 80}{Colors.END}\n")
    
    # Final state summary
    print_divider("Final State")
    print_state_summary(state)
    if state.interaction_mode == InteractionMode.CHAT:
        print_chat_history(state)

def run_demos():
    """Run all demonstration scenarios."""
    # Initialize the agent
    agent = MathAgent()
    
    # Print welcome message
    print_divider("Math Agent Demonstrations")
    print(f"{Colors.BOLD}Welcome to the Math Agent Demo!{Colors.END}")
    print("This demo will showcase various interaction patterns and capabilities.")
    print("Each scenario demonstrates different user behaviors and agent responses.")
    print()
    
    # List available scenarios
    print(f"{Colors.BOLD}Available scenarios:{Colors.END}")
    for key, scenario in SCENARIOS.items():
        print(f"  {Colors.YELLOW}{key}{Colors.END}: {scenario['name']} - {scenario['description']}")
    print()
    
    # Ask user which scenario to run
    while True:
        choice = input("Enter scenario key to run, 'all' for all scenarios, or 'exit' to quit: ").strip().lower()
        
        if choice == 'exit':
            break
            
        if choice == 'all':
            for scenario_key in SCENARIOS.keys():
                simulate_interaction(agent, scenario_key)
                input("Press Enter to continue to the next scenario...")
            break
            
        if choice in SCENARIOS:
            simulate_interaction(agent, choice)
            continue_choice = input("Run another scenario? (y/n): ").strip().lower()
            if continue_choice != 'y':
                break
        else:
            print(f"Unknown scenario: {choice}")

def main():
    """Main entry point for the demo."""
    try:
        run_demos()
    except KeyboardInterrupt:
        print("\nDemo terminated by user.")
    except Exception as e:
        print(f"Error running demo: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nThanks for trying the Math Agent Demo!")

if __name__ == "__main__":
    main() 