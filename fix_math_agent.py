#!/usr/bin/env python3
"""
Fix script for MathAgent class import issues.
This script modifies the math_agent.py file to fix import issues.
"""

import os
import re
import sys

def fix_imports(file_path):
    """
    Fix the imports in the math_agent.py file.
    
    Args:
        file_path: Path to the math_agent.py file
    """
    print(f"Fixing imports in {file_path}")
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the position to insert the imports
    import_pattern = r"import os\nimport re\nimport logging\nimport json\nimport traceback"
    match = re.search(import_pattern, content)
    
    if not match:
        print("Could not find the import section. Aborting.")
        return False
    
    # Extract the command class definitions from the file
    command_pattern = r"# Define minimal command implementations for testing\s+class MathParseQuestionCommand.*?class MathGenerateFeedbackCommand.*?return state"
    command_match = re.search(command_pattern, content, re.DOTALL)
    
    if not command_match:
        print("Could not find the command class definitions. Aborting.")
        return False
    
    command_defs = command_match.group(0)
    
    # Remove the old command definitions
    content = re.sub(command_pattern, "", content, flags=re.DOTALL)
    
    # Create the new imports section
    new_imports = """import os
import re
import logging
import json
import traceback
import datetime
from typing import Dict, Any, List, Optional, Union, Tuple, Type

logger = logging.getLogger(__name__)

# Try to import LangGraph for graph-based execution
try:
    import langgraph
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Error importing LangGraph: {e}. Using linear execution fallback.")
    LANGGRAPH_AVAILABLE = False

# Try to import SymPy for symbolic math
try:
    from sympy import sympify
    SYMPY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Error importing SymPy: {e}. Using string-based math fallback.")
    SYMPY_AVAILABLE = False
    
    # Simple fallback for sympify when SymPy is not available.
    def sympify(expr):
        return expr

# Import the base agent - with fallback if needed
try:
    from app.agents.agents.base_agent import BaseAgent
    from app.agents.commands import Command
except ImportError as e:
    logger.warning(f"Error importing BaseAgent: {e}. Using minimal implementation.")
    
    class BaseAgent:
        def __init__(self, model="gpt-4o-mini", llm_service=None, use_graph=False):
            self.model = model
            self.llm_service = llm_service
            self.use_graph = use_graph
            self.commands = []
            
        def _initialize_commands(self):
            return []
    
    class Command:
        def __init__(self, agent=None):
            self.agent = agent
            
        def execute(self, state):
            return state

# Import modular math commands with fallbacks
try:
    from app.agents.agents.math_commands.parse_command import MathParseQuestionCommand
    from app.agents.agents.math_commands.solve_command import MathSolveSymbolicallyCommand
    from app.agents.agents.math_commands.analyze_command import MathAnalyzeCalculationCommand
    from app.agents.agents.math_commands.assess_command import MathAssessProximityCommand
    from app.agents.agents.math_commands.hint_command import MathGenerateHintCommand
    from app.agents.agents.math_commands.feedback_command import MathGenerateFeedbackCommand
    COMMANDS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Error importing math commands: {e}. Using minimal implementations.")
    COMMANDS_AVAILABLE = False
    
    # Define minimal command implementations for testing
    class MathParseQuestionCommand(Command):
        def execute(self, state):
            state["analysis"] = {"parsed_question": {"variables": ["x"], "equations": ["2x + 3 = 7"]}}
            return state
            
    class MathSolveSymbolicallyCommand(Command):
        def execute(self, state):
            state["correct_answer"] = "x = 2"
            return state
            
    class MathAnalyzeCalculationCommand(Command):
        def execute(self, state):
            state["is_correct"] = state.get("student_answer") == state.get("correct_answer")
            state["analysis"]["steps"] = ["Step 1: Subtract 3 from both sides", "Step 2: Divide by 2"]
            return state
            
    class MathAssessProximityCommand(Command):
        def execute(self, state):
            state["proximity_score"] = 1.0 if state.get("is_correct") else 0.5
            state["needs_hint"] = not state.get("is_correct")
            return state
            
    class MathGenerateHintCommand(Command):
        def execute(self, state):
            state["hints"] = [{"text": "Try subtracting 3 from both sides first."}]
            return state
            
    class MathGenerateFeedbackCommand(Command):
        def execute(self, state):
            if state.get("is_correct", False):
                state["feedback"] = {
                    "summary": "Correct!",
                    "strengths": ["Your answer is correct"],
                    "areas_for_improvement": [],
                    "next_steps": ["Try more challenging problems"]
                }
            else:
                state["feedback"] = {
                    "summary": "Incorrect",
                    "strengths": ["You attempted the problem"],
                    "areas_for_improvement": ["Check your calculations"],
                    "next_steps": ["Review the steps in the solution"]
                }
            return state

# Import the mock LLM service for testing
try:
    from app.agents.agents.math_commands.mock_llm_service import MockLLMService
except ImportError as e:
    logger.warning(f"Error importing MockLLMService: {e}. Using minimal implementation.")
    # Define minimal MockLLMService for testing
    class MockLLMService:
        def generate_completion(self, system_prompt, user_prompt):
            return {"content": "This is a mock response"}
"""
    
    # Replace the imports section
    content = re.sub(import_pattern, new_imports, content)
    
    # Remove duplicate command imports if they exist
    duplicate_import_pattern = r"# Import modular math commands with fallbacks.*?class MathGenerateFeedbackCommand\(Command\):.*?return state"
    content = re.sub(duplicate_import_pattern, "", content, flags=re.DOTALL)
    
    # Fix the _initialize_commands method
    initialize_pattern = r"def _initialize_commands\(self\).*?# Import the refactored command classes.*?try:.*?except ImportError as e:.*?# Create command instances"
    initialize_replacement = """def _initialize_commands(self) -> List[Command]:
        \"\"\"
        Initialize math-specific commands.
        
        Returns:
            List of command instances
        \"\"\"
        # Create command instances"""
    
    content = re.sub(initialize_pattern, initialize_replacement, content, flags=re.DOTALL)
    
    # Write the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Successfully fixed imports in math_agent.py")
    return True

def main():
    """
    Main function to fix the math_agent.py file.
    """
    # Get the path to the math_agent.py file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    math_agent_path = os.path.join(script_dir, "app", "agents", "agents", "math_agent.py")
    
    if not os.path.exists(math_agent_path):
        print(f"Error: Could not find math_agent.py at {math_agent_path}")
        return 1
    
    # Fix the imports
    if not fix_imports(math_agent_path):
        print("Failed to fix imports")
        return 1
    
    print("Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
