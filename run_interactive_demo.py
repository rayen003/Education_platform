#!/usr/bin/env python3
"""
Interactive Terminal-Based Math Assessment Demo

This script provides a terminal-based interface for testing the Math Assessment
functionality without requiring the Streamlit UI.
"""

import os
import sys
import time
import asyncio
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
project_root = str(Path(__file__).absolute().parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('math_demo.log', mode='w')
    ]
)

# Create a logger
logger = logging.getLogger("MathDemo")
logger.info("Starting interactive math assessment demo")

# Import required modules - with error handling to bypass problematic imports
try:
    # Try direct import of the math services we need, bypassing the main app imports
    sys.path.insert(0, os.path.join(project_root, 'app', 'math_services'))
    from app.math_services.agent.math_agent import MathAgent
    from app.math_services.agent.meta_agent import MetaAgent
    from app.math_services.models.state import MathState, ChatMessage, InteractionMode
    from app.math_services.services.llm.openai_service import OpenAILLMService
except ImportError as e:
    logger.error(f"Import error: {str(e)}")
    logger.info("Attempting alternative import approach...")
    
    # If there's an issue with imports, try a more direct approach
    # by adding the specific module paths
    math_services_path = os.path.join(project_root, 'app', 'math_services')
    sys.path.insert(0, math_services_path)
    
    # Manual imports from specific files
    try:
        from agent.math_agent import MathAgent
        from agent.meta_agent import MetaAgent
        from models.state import MathState, ChatMessage, InteractionMode
        from services.llm.openai_service import OpenAILLMService
        logger.info("Successfully imported using direct path approach")
    except ImportError as e2:
        logger.error(f"Alternative import also failed: {str(e2)}")
        print(f"Error importing required modules. Please check the project structure.")
        sys.exit(1)

# ANSI color codes for terminal formatting
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(verbose=False):
    """Print the app header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}===== Math Assessment Interactive Demo ====={Colors.END}")
    print(f"{Colors.CYAN}Play the role of a student solving math problems{Colors.END}")
    if verbose:
        print(f"{Colors.YELLOW}(Verbose mode: showing verification and confidence details){Colors.END}")
    print()

def print_divider():
    """Print a divider line."""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_shortcuts():
    """Print available keyboard shortcuts."""
    print(f"\n{Colors.BOLD}Available Commands:{Colors.END}")
    print(f"  {Colors.YELLOW}[H]{Colors.END} - Request a hint")
    print(f"  {Colors.YELLOW}[F]{Colors.END} - Get feedback on your answer")
    print(f"  {Colors.YELLOW}[S]{Colors.END} - Show complete solution")
    print(f"  {Colors.YELLOW}[R]{Colors.END} - Show reasoning steps")
    print(f"  {Colors.YELLOW}[V]{Colors.END} - Toggle verbose mode")
    print(f"  {Colors.YELLOW}[Q]{Colors.END} - Quit the demo")
    print()

def print_chat_message(role: str, message: str, timestamp: Optional[datetime.datetime] = None):
    """Print a formatted chat message."""
    if timestamp is None:
        timestamp = datetime.datetime.now()
    
    time_str = timestamp.strftime("%H:%M")
    
    if role.lower() == 'student':
        print(f"{Colors.YELLOW}You ({time_str}):{Colors.END}")
        print(f"  {message}\n")
    else:
        print(f"{Colors.GREEN}Tutor ({time_str}):{Colors.END}")
        print(f"  {message}\n")

def print_verification_info(state: MathState, operation: str, verbose=False):
    """Print verification information from the meta agent."""
    if not verbose:
        return
    
    print(f"\n{Colors.CYAN}===== Verification Information ====={Colors.END}")
    
    # Get confidence if available
    confidence = None
    if operation == "feedback" and hasattr(state, 'feedback'):
        if isinstance(state.feedback, dict):
            confidence = state.feedback.get('confidence', None)
        elif hasattr(state.feedback, 'confidence'):
            confidence = state.feedback.confidence
    
    # Check for verification info in context
    if hasattr(state, 'context') and state.context:
        if 'verification_result' in state.context:
            print(f"  Verification result: {state.context['verification_result']}")
        
        if 'verified_steps_count' in state.context:
            print(f"  Verified steps: {state.context['verified_steps_count']}")
        
        if 'regenerated_steps_count' in state.context:
            print(f"  Regenerated steps: {state.context['regenerated_steps_count']}")
            
        if 'confidence_scores' in state.context:
            print(f"  Confidence scores: {state.context['confidence_scores']}")
    
    # Display confidence from feedback
    if confidence is not None:
        confidence_percent = int(confidence * 100)
        confidence_bar = "=" * (confidence_percent // 5)
        confidence_level = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.5 else "Low"
        
        print(f"  {operation.title()} confidence: {confidence_percent}% ({confidence_level})")
        print(f"  [{confidence_bar}{' ' * (20 - len(confidence_bar))}] {confidence_percent}%")
    
    print(f"{Colors.CYAN}==============================={Colors.END}\n")

def print_reasoning_steps(steps: List[str], title="Chain of Thought Reasoning"):
    """Print reasoning steps in a formatted way."""
    print(f"\n{Colors.PURPLE}===== {title} ====={Colors.END}")
    
    for i, step in enumerate(steps):
        print(f"{Colors.BOLD}Step {i+1}:{Colors.END} {step}")
        
        # Add a small delay between steps for effect
        time.sleep(0.3)
    
    print()

def simulate_typing(message: str, delay: float = 0.01):
    """Simulate typing effect for tutor messages."""
    for char in message:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

async def main():
    """Run the interactive math assessment demo."""
    logger = logging.getLogger("MathDemo")
    logger.info("Starting interactive math assessment demo")
    
    clear_screen()
    print_header()
    print("Initializing Math Agent...")
    
    # Initialize the Math Agent with Meta Agent for verification
    try:
        # Create the Meta Agent for verification
        meta_agent = MetaAgent()
        logger.info("Meta Agent initialized")
        
        # Create the Math Agent with the Meta Agent
        math_agent = MathAgent(model="gpt-4o-mini")
        math_agent.meta_agent = meta_agent
        logger.info("Math Agent initialized with Meta Agent for verification")
        
        print(f"{Colors.GREEN}✓ Math Agent initialized successfully!{Colors.END}")
        print(f"{Colors.GREEN}✓ Meta Agent connected for verification!{Colors.END}")
    except Exception as e:
        logger.error(f"Error initializing agents: {str(e)}")
        print(f"{Colors.RED}Error initializing Math Agent: {str(e)}{Colors.END}")
        return
    
    # Demo state variables
    problem_submitted = False
    answer_submitted = False
    solution_revealed = False
    reasoning_generated = False
    verbose_mode = True
    hint_count = 0
    chat_history = []
    reasoning_steps = []
    current_problem = ""
    
    print_divider()
    print("Welcome to the Math Assessment Interactive Demo!")
    print("You can start by entering a math problem, then interact using the commands below.")
    print_shortcuts()
    
    while True:
        if not problem_submitted:
            # Ask for the initial problem
            print(f"{Colors.BOLD}Enter a math problem:{Colors.END} (or 'q' to quit)")
            user_input = input("> ")
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            if not user_input.strip():
                print(f"{Colors.RED}Please enter a valid problem.{Colors.END}")
                continue
            
            print_chat_message('student', user_input)
            current_problem = user_input
            problem_submitted = True
            
            # Process the problem with the Math Agent
            print(f"{Colors.CYAN}Processing problem...{Colors.END}")
            try:
                logger.info(f"Processing problem: {current_problem}")
                math_state = MathState(question=current_problem, student_answer="")
                math_state = math_agent.solve(math_state)
                
                # Store reasoning steps if available
                if hasattr(math_state, 'steps') and math_state.steps:
                    reasoning_steps = math_state.steps
                    reasoning_generated = True
                    logger.info(f"Generated {len(reasoning_steps)} reasoning steps")
                
                # Add response to chat history
                response = "I've analyzed this problem. Please provide your answer, and I'll give you feedback. You can also ask for a hint if you need help."
                print_chat_message('tutor', response)
                chat_history.append({"role": "tutor", "message": response})
                
                # Update current state in math agent
                math_agent.current_state = math_state
                
            except Exception as e:
                logger.error(f"Error processing problem: {str(e)}")
                print(f"{Colors.RED}Error processing problem: {str(e)}{Colors.END}")
                problem_submitted = False
                continue
        
        elif not answer_submitted:
            # Ask for the student's answer
            print(f"{Colors.BOLD}Enter your answer:{Colors.END} (or type a command: h/f/s/r/v/q)")
            user_input = input("> ")
            
            # Handle commands
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            elif user_input.lower() == 'h':
                # Request hint
                if solution_revealed:
                    print(f"{Colors.RED}Solution already revealed. Hints are no longer available.{Colors.END}")
                    continue
                
                print(f"{Colors.CYAN}Generating hint...{Colors.END}")
                try:
                    logger.info("Requesting hint")
                    hint_count += 1
                    updated_state = math_agent.process_interaction('button', 'hint', math_agent.current_state)
                    
                    # Extract hint - only get the latest hint
                    hint = "I don't have any hints for this problem yet."
                    if hasattr(updated_state, 'hints') and updated_state.hints:
                        hint = updated_state.hints[-1]  # Just get the most recent hint
                    
                    hint_message = f"💡 Hint #{hint_count}: {hint}"
                    print_chat_message('tutor', hint_message)
                    chat_history.append({"role": "tutor", "message": hint_message})
                    
                    # Show verification info if available
                    print_verification_info(updated_state, "hint", verbose_mode)
                    
                    # Update current state
                    math_agent.current_state = updated_state
                    logger.info(f"Generated hint #{hint_count}")
                except Exception as e:
                    logger.error(f"Error generating hint: {str(e)}")
                    print(f"{Colors.RED}Error generating hint: {str(e)}{Colors.END}")
                
                continue
            elif user_input.lower() == 'f':
                # Provide feedback on current answer
                if not math_agent.current_state.student_answer:
                    print(f"{Colors.RED}Please provide an answer first.{Colors.END}")
                    continue
                
                print(f"{Colors.CYAN}Generating feedback...{Colors.END}")
                try:
                    logger.info("Generating feedback")
                    updated_state = math_agent.process_interaction('button', 'feedback', math_agent.current_state)
                    
                    # Extract feedback
                    feedback_text = "I don't have specific feedback for your answer yet."
                    if hasattr(updated_state, 'feedback'):
                        if isinstance(updated_state.feedback, dict) and 'assessment' in updated_state.feedback:
                            feedback_text = updated_state.feedback['assessment']
                        elif hasattr(updated_state.feedback, 'assessment'):
                            feedback_text = updated_state.feedback.assessment
                        else:
                            feedback_text = str(updated_state.feedback)
                    
                    print_chat_message('tutor', feedback_text)
                    chat_history.append({"role": "tutor", "message": feedback_text})
                    
                    # Show verification info if available
                    print_verification_info(updated_state, "feedback", verbose_mode)
                    
                    # Update answer submitted status if correct
                    if hasattr(updated_state.analysis, 'is_correct'):
                        answer_submitted = True
                    
                    # Update current state
                    math_agent.current_state = updated_state
                    logger.info("Generated feedback")
                except Exception as e:
                    logger.error(f"Error generating feedback: {str(e)}")
                    print(f"{Colors.RED}Error generating feedback: {str(e)}{Colors.END}")
                
                continue
            elif user_input.lower() == 's':
                # Show solution
                print(f"{Colors.CYAN}Generating complete solution...{Colors.END}")
                try:
                    logger.info("Generating solution")
                    updated_state = math_agent.process_interaction('button', 'solution', math_agent.current_state)
                    
                    solution_text = "Here's the solution:\n\n"
                    if hasattr(updated_state, 'correct_answer') and updated_state.correct_answer:
                        solution_text += f"Answer: {updated_state.correct_answer}\n\n"
                    
                    # Include reasoning steps if available
                    if hasattr(updated_state, 'steps') and updated_state.steps:
                        reasoning_steps = updated_state.steps
                        reasoning_generated = True
                        solution_text += "Explanation:\n\n"
                        for i, step in enumerate(updated_state.steps):
                            solution_text += f"Step {i+1}: {step}\n"
                    
                    print_chat_message('tutor', solution_text)
                    chat_history.append({"role": "tutor", "message": solution_text})
                    
                    # Show verification info if available
                    print_verification_info(updated_state, "solution", verbose_mode)
                    
                    # Update current state
                    math_agent.current_state = updated_state
                    solution_revealed = True
                    logger.info("Generated solution")
                except Exception as e:
                    logger.error(f"Error generating solution: {str(e)}")
                    print(f"{Colors.RED}Error generating solution: {str(e)}{Colors.END}")
                
                continue
            elif user_input.lower() == 'r':
                # Show reasoning
                if not reasoning_generated:
                    print(f"{Colors.CYAN}Generating reasoning steps...{Colors.END}")
                    try:
                        logger.info("Generating reasoning steps")
                        # Set up context for CoT
                        if not math_agent.current_state.context:
                            math_agent.current_state.context = {}
                        math_agent.current_state.context['reasoning_mode'] = 'cot'
                        
                        updated_state = math_agent.process_interaction('button', 'reasoning', math_agent.current_state)
                        
                        # Store reasoning steps if available
                        if hasattr(updated_state, 'steps') and updated_state.steps:
                            reasoning_steps = updated_state.steps
                            reasoning_generated = True
                            logger.info(f"Generated {len(reasoning_steps)} reasoning steps")
                            
                        # Update current state
                        math_agent.current_state = updated_state
                        
                        # Show verification info if available
                        print_verification_info(updated_state, "reasoning", verbose_mode)
                        
                    except Exception as e:
                        logger.error(f"Error generating reasoning: {str(e)}")
                        print(f"{Colors.RED}Error generating reasoning: {str(e)}{Colors.END}")
                
                if reasoning_steps:
                    # Display the DoT steps if available
                    dot_steps = []
                    if hasattr(math_agent.current_state, 'context') and math_agent.current_state.context:
                        dot_steps = math_agent.current_state.context.get('dot_reasoning_steps', [])
                    
                    # Show the DoT steps if available
                    if dot_steps:
                        print_reasoning_steps(dot_steps, "Chain of Draft (Concise Reasoning)")
                        
                        # Get full reasoning for the expander view
                        full_steps = math_agent.current_state.context.get('full_reasoning_steps', reasoning_steps)
                        
                        # Ask if user wants to see full reasoning
                        if len(full_steps) > len(dot_steps):
                            print(f"{Colors.BOLD}Show full detailed reasoning? (y/n){Colors.END}")
                            show_full = input("> ").lower() == 'y'
                            
                            if show_full:
                                print_reasoning_steps(full_steps, "Full Chain of Thought Reasoning")
                    else:
                        # Just show the regular reasoning steps
                        print_reasoning_steps(reasoning_steps)
                
                continue
            elif user_input.lower() == 'v':
                # Toggle verbose mode
                verbose_mode = not verbose_mode
                print(f"{Colors.CYAN}Verbose mode {'enabled' if verbose_mode else 'disabled'}{Colors.END}")
                print_header(verbose_mode)
                continue
            
            # If not a command, treat as an answer
            print_chat_message('student', user_input)
            
            # Update the math state with the student's answer
            try:
                logger.info(f"Processing student answer: {user_input}")
                # Update the student answer in the state
                math_agent.current_state.student_answer = user_input
                
                # Generate feedback automatically
                updated_state = math_agent.process_interaction('button', 'feedback', math_agent.current_state)
                
                # Extract feedback
                feedback_text = "I don't have specific feedback for your answer yet."
                if hasattr(updated_state, 'feedback'):
                    if isinstance(updated_state.feedback, dict) and 'assessment' in updated_state.feedback:
                        feedback_text = updated_state.feedback['assessment']
                    elif hasattr(updated_state.feedback, 'assessment'):
                        feedback_text = updated_state.feedback.assessment
                    else:
                        feedback_text = str(updated_state.feedback)
                
                print_chat_message('tutor', feedback_text)
                chat_history.append({"role": "tutor", "message": feedback_text})
                
                # Show verification info if available
                print_verification_info(updated_state, "feedback", verbose_mode)
                
                # Check if answer is correct
                is_correct = False
                if hasattr(updated_state.analysis, 'is_correct'):
                    is_correct = updated_state.analysis.is_correct
                
                # Update answer submitted status if correct
                if is_correct:
                    answer_submitted = True
                    print(f"{Colors.GREEN}✓ Your answer is correct!{Colors.END}")
                    
                    # Suggest viewing reasoning
                    if not reasoning_generated:
                        print(f"\n{Colors.YELLOW}Tip:{Colors.END} Type 'r' to see the step-by-step reasoning.")
                else:
                    print(f"{Colors.YELLOW}Your answer needs some improvement. Try again or use the commands.{Colors.END}")
                
                # Update current state
                math_agent.current_state = updated_state
                
            except Exception as e:
                logger.error(f"Error processing answer: {str(e)}")
                print(f"{Colors.RED}Error processing answer: {str(e)}{Colors.END}")
        
        else:
            # After correctly answering, provide more options
            print(f"{Colors.BOLD}What would you like to do next?{Colors.END} (r=reasoning, s=solution, n=new problem, q=quit)")
            user_input = input("> ")
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            elif user_input.lower() in ['n', 'new']:
                # Reset for a new problem
                problem_submitted = False
                answer_submitted = False
                solution_revealed = False
                reasoning_generated = False
                hint_count = 0
                chat_history = []
                reasoning_steps = []
                current_problem = ""
                print(f"{Colors.CYAN}Starting new problem...{Colors.END}")
                print_divider()
            elif user_input.lower() == 'r':
                # Show reasoning (reuse the 'r' command handler)
                user_input = 'r'
                continue
            elif user_input.lower() == 's':
                # Show solution (reuse the 's' command handler)
                user_input = 's'
                continue
            else:
                print(f"{Colors.RED}Unknown command. Please choose a valid option.{Colors.END}")
    
    print(f"\n{Colors.BLUE}Thank you for using the Math Assessment Interactive Demo!{Colors.END}")
    logger.info("Demo session ended")

if __name__ == "__main__":
    asyncio.run(main()) 