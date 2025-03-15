"""
Test Chat Command Functionality.

This script tests the chat command functionality directly,
bypassing the hybrid_interaction_example.py to troubleshoot the
"No module named 'app.math'" error.
"""

import os
import sys
import logging
import traceback
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def print_divider():
    """Print a divider for better visual separation."""
    print("\n" + "="*80 + "\n")

def test_direct_import():
    """Test direct import of the chat command class."""
    print_divider()
    print("TEST 1: Direct Import Test")
    try:
        from app.math_services.commands.chat_command import MathChatFollowUpCommand
        print("✅ Successfully imported MathChatFollowUpCommand")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print(traceback.format_exc())
    print_divider()

def test_agent_integration():
    """Test integration with MathAgent."""
    print_divider()
    print("TEST 2: Integration with MathAgent")
    try:
        from app.math_services.agent.math_agent import MathAgent
        
        # Initialize the MathAgent
        agent = MathAgent()
        print("✅ Successfully initialized MathAgent")
        
        # Check if the handle_follow_up method exists
        if hasattr(agent, 'handle_follow_up'):
            print("✅ MathAgent has handle_follow_up method")
        else:
            print("❌ MathAgent is missing handle_follow_up method")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print(traceback.format_exc())
    print_divider()

def test_minimal_chat():
    """Test minimal chat interaction."""
    print_divider()
    print("TEST 3: Minimal Chat Interaction")
    try:
        from app.math_services.agent.math_agent import MathAgent
        
        # Initialize agent
        agent = MathAgent()
        
        # Create minimal state
        state = {
            "question": "What is 5 + 3?",
            "student_answer": "8",
            "correct_answer": "8",
            "feedback": {
                "math": {
                    "assessment": "That's correct! 5 + 3 = 8.",
                    "is_correct": True,
                    "proximity_score": 10
                }
            }
        }
        
        # Analyze first to build context
        state = agent.analyze(state)
        print("✅ Successfully analyzed initial state")
        
        # Toggle to chat mode
        state = agent.toggle_interaction_mode(state)
        print("✅ Successfully toggled to chat mode")
        
        # Test follow-up question
        follow_up = "Can you show me a similar problem?"
        print(f"Testing follow-up question: '{follow_up}'")
        
        try:
            state = agent.handle_follow_up(state, follow_up)
            print("✅ Successfully handled follow-up question")
            print(f"Response: {state.get('chat_response', 'No response')}")
        except Exception as e:
            print(f"❌ Follow-up question handling failed: {e}")
            print(traceback.format_exc())
            
            # Try debugging by directly importing and using the chat command
            print("\nAttempting direct use of MathChatFollowUpCommand...")
            try:
                from app.math_services.commands.chat_command import MathChatFollowUpCommand
                chat_cmd = MathChatFollowUpCommand(agent)
                state = chat_cmd.execute(state, follow_up)
                print("✅ Direct use of MathChatFollowUpCommand successful")
                print(f"Response: {state.get('chat_response', 'No response')}")
            except Exception as e2:
                print(f"❌ Direct use also failed: {e2}")
                print(traceback.format_exc())
    except Exception as e:
        print(f"❌ Minimal chat test failed: {e}")
        print(traceback.format_exc())
    print_divider()

def test_full_cycle():
    """Test full state cycle with mode switching."""
    print_divider()
    print("TEST 4: Full State Cycle with Mode Switching")
    try:
        from app.math_services.agent.math_agent import MathAgent
        
        # Initialize agent
        agent = MathAgent()
        
        # Create state
        state = {
            "question": "Solve for x in the equation 2x + 5 = 15",
            "student_answer": "x = 5",
            "correct_answer": "x = 5",
            "analysis": {},
            "feedback": {},
            "hint_count": 0,
            "hints": [],
            "needs_hint": False,
            "interaction_mode": "structured"
        }
        
        # Step 1: Initial analysis
        print("Step 1: Initial Analysis (Structured Mode)")
        state = agent.analyze(state)
        print(f"Feedback: {state['feedback']['math']['assessment']}")
        print(f"Is Correct: {state['feedback']['math']['is_correct']}")
        
        # Step 2: Toggle to chat mode
        print("\nStep 2: Toggle to Chat Mode")
        state = agent.toggle_interaction_mode(state)
        print(f"Current Mode: {state['interaction_mode']}")
        
        # Step 3: Ask follow-up
        print("\nStep 3: Follow-up Question")
        follow_up = "Can you explain the steps to solve this?"
        try:
            state = agent.handle_follow_up(state, follow_up)
            print("✅ Successfully handled follow-up question")
            print(f"Response: {state.get('chat_response', 'No response')}")
        except Exception as e:
            print(f"❌ Follow-up handling failed: {e}")
            print(traceback.format_exc())
        
        # Step 4: Toggle back to structured
        print("\nStep 4: Toggle Back to Structured Mode")
        state = agent.toggle_interaction_mode(state)
        print(f"Current Mode: {state['interaction_mode']}")
        
        # Step 5: Request hint
        print("\nStep 5: Request Hint (Structured Mode)")
        state["needs_hint"] = True
        state = agent.analyze(state)
        if state["hints"]:
            print("\nHints:")
            for i, hint in enumerate(state["hints"]):
                print(f"  Hint #{i+1}: {hint}")
        else:
            print("❌ No hints generated")
    except Exception as e:
        print(f"❌ Full cycle test failed: {e}")
        print(traceback.format_exc())
    print_divider()

def inspect_code():
    """Inspect the code for import errors."""
    print_divider()
    print("TEST 5: Code Inspection")
    
    # Check if the chat_command.py file exists
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'commands', 'chat_command.py')
    if os.path.exists(file_path):
        print(f"✅ chat_command.py exists at {file_path}")
        
        # Read the first 20 lines to check imports
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()[:20]
                print("\nFirst 20 lines of chat_command.py:")
                for i, line in enumerate(lines):
                    print(f"{i+1:3d}: {line.rstrip()}")
                    
                # Look for potential import issues
                for i, line in enumerate(lines):
                    if 'import' in line and 'app.math' in line and not 'app.math_services' in line:
                        print(f"\n❌ Problematic import found at line {i+1}: {line.strip()}")
                        print("   Should be importing from 'app.math_services' not 'app.math'")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    else:
        print(f"❌ chat_command.py not found at {file_path}")
    print_divider()
    
    # Also check handle_follow_up in math_agent.py
    try:
        from app.math_services.agent.math_agent import MathAgent
        import inspect
        
        source = inspect.getsource(MathAgent.handle_follow_up)
        print("handle_follow_up method source:")
        print(source)
    except Exception as e:
        print(f"❌ Error inspecting handle_follow_up: {e}")
    print_divider()

def main():
    """Run all tests."""
    print("\n== CHAT COMMAND FUNCTIONALITY TESTS ==\n")
    
    test_direct_import()
    test_agent_integration()
    inspect_code()
    test_minimal_chat()
    test_full_cycle()
    
    print("\n== TESTS COMPLETED ==\n")

if __name__ == "__main__":
    main() 