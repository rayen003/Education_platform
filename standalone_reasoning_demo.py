#!/usr/bin/env python3
"""
Standalone Reasoning Demo

This script demonstrates a math tutoring experience with:
1. Draft of Thought (DoT) and Chain of Thought (CoT) reasoning approaches
2. Hint generation and feedback
3. Student answer assessment
4. Meta agent verification
"""

import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
from enum import Enum
import re

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
        logging.FileHandler('math_tutor_demo.log', mode='w')
    ]
)

# Create a logger
logger = logging.getLogger("MathTutorDemo")

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

def print_header():
    """Print the app header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}===== Interactive Math Tutor Demo ====={Colors.END}")
    print(f"{Colors.CYAN}With Draft of Thought (DoT), Chain of Thought (CoT), hints and assessment{Colors.END}\n")

def print_divider(title=None):
    """Print a divider with optional title."""
    if title:
        print(f"\n{Colors.BOLD}{Colors.BLUE}===== {title} ====={Colors.END}\n")
    else:
        print(f"\n{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_verification_info(verification_data):
    """Print meta agent verification information."""
    print(f"\n{Colors.CYAN}===== Verification Information ====={Colors.END}")
    
    if isinstance(verification_data, dict):
        if 'verified_steps' in verification_data:
            verified = verification_data.get('verified_steps', [])
            print(f"  Verified steps: {len(verified)}")
            if verified:
                print(f"  Steps verified: {', '.join(str(s+1) for s in verified)}")
        
        if 'regenerated_steps' in verification_data:
            regenerated = verification_data.get('regenerated_steps', [])
            print(f"  Regenerated steps: {len(regenerated)}")
            if regenerated:
                print(f"  Steps regenerated: {', '.join(str(s+1) for s in regenerated)}")
            
        if 'problematic_steps' in verification_data:
            problematic = verification_data.get('problematic_steps', [])
            print(f"  Problematic steps: {len(problematic)}")
            if problematic:
                print(f"  Steps with issues: {', '.join(str(s+1) for s in problematic)}")
            
        if 'confidence_scores' in verification_data:
            scores = verification_data['confidence_scores']
            print(f"  Overall confidence: {scores.get('overall', 0)*100:.1f}%")
            
            # Print individual step confidence scores
            step_scores = {k: v for k, v in scores.items() if k.startswith('step_')}
            if step_scores:
                print(f"  Step confidence scores:")
                for step, score in step_scores.items():
                    step_num = step.replace('step_', '')
                    print(f"    Step {step_num}: {score*100:.1f}%")
    else:
        print(f"  {verification_data}")
    
    print(f"{Colors.CYAN}==============================={Colors.END}\n")

def print_steps(steps, title="Chain of Thought Reasoning", show_typing=False):
    """Print reasoning steps in a formatted way."""
    print(f"\n{Colors.PURPLE}===== {title} ====={Colors.END}")
    
    for i, step in enumerate(steps):
        step_prefix = f"{Colors.BOLD}Step {i+1}:{Colors.END} "
        
        if show_typing:
            print(step_prefix, end='')
            simulate_typing(step)
        else:
            print(f"{step_prefix}{step}")
        
        # Add a small delay between steps for effect
        time.sleep(0.2)
    
    print()

def print_confidence_bar(confidence, label="Confidence"):
    """Print a confidence bar visualization."""
    if confidence is None:
        return
    
    confidence_percent = int(confidence * 100)
    confidence_bar = "=" * (confidence_percent // 5)
    confidence_level = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.5 else "Low"
    bar_color = Colors.GREEN if confidence >= 0.8 else Colors.YELLOW if confidence >= 0.5 else Colors.RED
    
    print(f"{Colors.BOLD}{label}:{Colors.END} {confidence_percent}% ({confidence_level})")
    print(f"[{bar_color}{confidence_bar}{Colors.END}{' ' * (20 - len(confidence_bar))}] {confidence_percent}%")
    print()

def print_options(options, prompt="What would you like to do?"):
    """Print numbered options for the user to select from."""
    print(f"\n{Colors.BOLD}{prompt}{Colors.END}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    print()

def simulate_typing(text, delay=0.01):
    """Simulate typing effect for text."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def init_services():
    """Initialize LLM service, Meta Agent and Math Agent safely."""
    try:
        print(f"{Colors.YELLOW}Initializing services...{Colors.END}")
        
        # Dynamic import of required components
        from app.math_services.services.llm.openai_service import OpenAILLMService
        from app.math_services.agent.meta_agent import MetaAgent
        from app.math_services.agent.math_agent import MathAgent
        from app.math_services.services.service_container import ServiceContainer
        
        # Initialize LLM service with lower temperature for more deterministic results
        llm_service = OpenAILLMService(model="gpt-4o-mini")
        logger.info("Successfully initialized OpenAI LLM service")
        
        # Initialize Meta Agent with the LLM service
        meta_agent = MetaAgent(model="gpt-4o-mini", llm_service=llm_service)
        logger.info("Successfully initialized Meta Agent")
        
        # Initialize service container
        service_container = ServiceContainer(llm_service, meta_agent)
        logger.info("Successfully initialized Service Container")
        
        # Initialize Math Agent with the service container
        math_agent = MathAgent(model="gpt-4o-mini")
        math_agent.service_container = service_container
        logger.info("Successfully initialized Math Agent")
        
        return llm_service, meta_agent, math_agent
    
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}", exc_info=True)
        print(f"\n{Colors.RED}Error initializing services: {str(e)}{Colors.END}")
        return None, None, None

def extract_steps(text):
    """Extract reasoning steps from text response."""
    logger.info("Extracting steps from text response")
    
    # Try to find steps marked with "Step X:" format
    step_pattern = r"(?:Step|STEP)\s*(\d+)[:\.]\s*(.*?)(?=(?:Step|STEP)\s*\d+[:\.]\s*|$)"
    matches = re.findall(step_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        logger.info(f"Found {len(matches)} steps using step pattern")
        return [step.strip() for _, step in matches]
    
    # Fallback: try to find numbered lines (e.g., "1. " or "1) ")
    lines = text.split('\n')
    steps = []
    for line in lines:
        if re.match(r'^\s*\d+[\.\)]\s+', line):
            steps.append(re.sub(r'^\s*\d+[\.\)]\s+', '', line))
    
    if steps:
        logger.info(f"Found {len(steps)} steps using numbered lines")
        return steps
    
    # Second fallback: split by sentences and take non-empty ones
    logger.info("No step format found, splitting by sentences")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    steps = [s.strip() for s in sentences if s.strip()]
    
    # If still no steps or too many small fragments, just return entire text as one step
    if not steps or (len(steps) > 10 and all(len(s) < 50 for s in steps)):
        logger.info("Returning entire text as a single step")
        return [text]
    
    return steps

def generate_hint(math_agent, problem, hint_level=1, previous_hints=None):
    """Generate a progressive hint at the specified level."""
    if previous_hints is None:
        previous_hints = []
    
    logger.info(f"Generating hint level {hint_level} for problem: {problem}")
    
    try:
        print(f"{Colors.CYAN}Generating hint...{Colors.END}")
        
        # Create a state for the problem
        from app.math_services.models.state import MathState
        state = MathState(question=problem, student_answer="")
        
        # Set the hint count to control the hint level
        state.hint_count = hint_level - 1
        state.hints = previous_hints
        
        # First solve the problem to get the correct answer
        if not state.correct_answer:
            state = math_agent.solve(state)
            logger.info(f"Solved problem, answer: {state.correct_answer}")
        
        # Generate the hint
        state = math_agent.generate_hint(state)
        hint = state.hints[-1] if state.hints else "Unable to generate a hint."
        logger.info(f"Generated hint: {hint}")
        
        return hint, state
    
    except Exception as e:
        logger.error(f"Error generating hint: {str(e)}", exc_info=True)
        print(f"{Colors.RED}Error generating hint: {str(e)}{Colors.END}")
        return "Sorry, I couldn't generate a hint for this problem.", None

def assess_answer(math_agent, problem, student_answer, correct_answer=None):
    """Assess a student's answer and provide feedback."""
    logger.info(f"Assessing answer for problem: {problem}")
    logger.info(f"Student answer: {student_answer}")
    
    try:
        print(f"{Colors.CYAN}Assessing your answer...{Colors.END}")
        
        # Create a state for the problem and answer
        from app.math_services.models.state import MathState
        state = MathState(question=problem, student_answer=student_answer)
        
        # Set the correct answer if provided
        if correct_answer:
            state.correct_answer = correct_answer
        else:
            # Solve the problem to get the correct answer
            solved_state = math_agent.solve(MathState(question=problem, student_answer=""))
            state.correct_answer = solved_state.correct_answer
            logger.info(f"Solved problem, answer: {state.correct_answer}")
        
        # Analyze the answer
        state = math_agent.analyze(state)
        logger.info(f"Analysis complete: is_correct={state.analysis.is_correct}")
        
        # Generate feedback
        state = math_agent.generate_feedback(state)
        
        # Extract feedback details
        if isinstance(state.feedback, dict):
            assessment = state.feedback.get("assessment", "")
            details = state.feedback.get("detail", "")
            is_correct = state.feedback.get("is_correct", False)
            confidence = state.feedback.get("confidence", 0.7)
        else:
            assessment = "Unable to generate detailed feedback."
            details = ""
            is_correct = False
            confidence = 0.5
        
        feedback = {
            "assessment": assessment,
            "details": details,
            "is_correct": is_correct,
            "confidence": confidence
        }
        
        logger.info(f"Generated feedback: {assessment}")
        return feedback, state
    
    except Exception as e:
        logger.error(f"Error assessing answer: {str(e)}", exc_info=True)
        print(f"{Colors.RED}Error assessing answer: {str(e)}{Colors.END}")
        return {
            "assessment": "Sorry, I couldn't assess your answer.",
            "details": f"Error: {str(e)}",
            "is_correct": False,
            "confidence": 0.0
        }, None

def generate_dot_reasoning(math_agent, problem):
    """Generate Draft of Thought (DoT) reasoning - shorter and more concise."""
    logger.info(f"Generating DoT reasoning for problem: {problem}")
    
    try:
        print(f"{Colors.CYAN}Generating Draft of Thought (DoT) reasoning...{Colors.END}")
        
        # Create a state for the problem
        from app.math_services.models.state import MathState
        state = MathState(question=problem, student_answer="")
        
        # Set the context to indicate we want DoT (not CoT)
        if not state.context:
            state.context = {}
        state.context['reasoning_mode'] = 'dot'
        
        # Generate the reasoning
        dot_state = math_agent.generate_reasoning(state)
        
        # Extract steps
        dot_steps = dot_state.steps
        logger.info(f"Generated {len(dot_steps)} DoT steps")
        
        # Fix verification results parsing
        verification_data = {}
        confidence = 0.7  # Default
        
        if 'verification_result' in dot_state.context:
            verification_data = dot_state.context['verification_result']
            logger.info(f"Found verification data: {verification_data}")
            
        if 'confidence_scores' in verification_data:
            scores = verification_data['confidence_scores']
            if 'overall' in scores:
                confidence = scores['overall']
                logger.info(f"Found overall confidence: {confidence}")
        
        return dot_steps, verification_data, confidence, dot_state
    
    except Exception as e:
        logger.error(f"Error generating DoT reasoning: {str(e)}", exc_info=True)
        print(f"{Colors.RED}Error generating DoT reasoning: {str(e)}{Colors.END}")
        return ["Error generating draft reasoning"], {}, 0.5, None

def generate_cot_reasoning(math_agent, problem):
    """Generate Chain of Thought (CoT) reasoning - detailed and comprehensive."""
    logger.info(f"Generating CoT reasoning for problem: {problem}")
    
    try:
        print(f"{Colors.CYAN}Generating Chain of Thought (CoT) reasoning...{Colors.END}")
        
        # Create a state for the problem
        from app.math_services.models.state import MathState
        state = MathState(question=problem, student_answer="")
        
        # Set the context to indicate we want CoT (not DoT)
        if not state.context:
            state.context = {}
        state.context['reasoning_mode'] = 'cot'
        
        # Generate the reasoning with full CoT
        cot_state = math_agent.generate_reasoning(state, use_cot=True)
        
        # Extract steps
        cot_steps = cot_state.steps
        logger.info(f"Generated {len(cot_steps)} CoT steps")
        
        # Fix verification results parsing
        verification_data = {}
        confidence = 0.7  # Default
        
        if 'verification_result' in cot_state.context:
            verification_data = cot_state.context['verification_result']
            logger.info(f"Found verification data: {verification_data}")
            
        if 'confidence_scores' in verification_data:
            scores = verification_data['confidence_scores']
            if 'overall' in scores:
                confidence = scores['overall']
                logger.info(f"Found overall confidence: {confidence}")
        
        return cot_steps, verification_data, confidence, cot_state
    
    except Exception as e:
        logger.error(f"Error generating CoT reasoning: {str(e)}", exc_info=True)
        print(f"{Colors.RED}Error generating CoT reasoning: {str(e)}{Colors.END}")
        return ["Error generating detailed reasoning"], {}, 0.5, None

def get_question_comprehension(llm_service, problem):
    """Check if the problem is clear and generate a comprehension note."""
    logger.info(f"Generating question comprehension for: {problem}")
    
    system_prompt = """You are a helpful math tutor. Your task is to analyze a math problem and:
1. Check if the problem is clear and well-defined
2. Identify what the problem is asking for
3. Note any necessary clarifications or assumptions
Be concise and direct in your assessment."""

    user_prompt = f"""Please analyze this math problem:

PROBLEM:
{problem}

Provide a brief assessment of:
1. Is the problem statement clear and complete?
2. What specifically is the problem asking for?
3. Any clarification needed?
"""

    try:
        response = llm_service.generate_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3
        )
        
        content = response.get("content", "")
        logger.info(f"Got comprehension response, length: {len(content)}")
        
        return content
    
    except Exception as e:
        logger.error(f"Error generating comprehension: {str(e)}", exc_info=True)
        return "I'm not sure if I understand the problem correctly. Could you provide more details?"

def print_feedback(feedback):
    """Print feedback on student's answer."""
    assessment = feedback.get("assessment", "")
    details = feedback.get("details", "")
    is_correct = feedback.get("is_correct", False)
    confidence = feedback.get("confidence", 0.7)
    
    if is_correct:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Correct!{Colors.END} {assessment}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Not quite right.{Colors.END} {assessment}")
    
    if details:
        print(f"\n{details}")
    
    print_confidence_bar(confidence, "Feedback Confidence")

# Define a problem type enumeration
class ProblemType(Enum):
    UNKNOWN = "unknown"
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    GEOMETRY = "geometry"
    ARITHMETIC = "arithmetic"
    STATISTICS = "statistics"
    PROBABILITY = "probability"
    PHYSICS = "physics"
    WORD_PROBLEM = "word_problem"

# Define a session state structure
class TutorSession:
    def __init__(self, problem=""):
        self.problem = problem            # The math problem text
        self.student_answer = None        # The student's answer (None if not provided yet)
        self.correct_answer = None        # The correct answer
        self.problem_type = ProblemType.UNKNOWN  # Type of problem
        self.variables = []               # Key variables in the problem
        self.hints = []                   # Hints provided so far
        self.hint_count = 0               # Number of hints provided
        self.steps_shown = False          # Whether solution steps have been shown
        self.dot_shown = False            # Whether DoT has been shown
        self.cot_shown = False            # Whether CoT has been shown
        self.comprehension_shown = False  # Whether problem comprehension has been shown
        self.attempts = 0                 # Number of solution attempts
        
    def has_answer(self):
        """Check if student has provided an answer."""
        return self.student_answer is not None and self.student_answer.strip() != ""
    
    def update_problem_info(self, llm_service):
        """Extract problem type and variables."""
        try:
            # Only try to classify if not already done
            if self.problem_type == ProblemType.UNKNOWN:
                self.problem_type = classify_problem_type(llm_service, self.problem)
            
            # Extract variables if not already done
            if not self.variables:
                self.variables = extract_variables(llm_service, self.problem)
                
            return True
        except Exception as e:
            logger.error(f"Error updating problem info: {str(e)}")
            return False

def classify_problem_type(llm_service, problem):
    """Classify the type of math problem."""
    logger.info(f"Classifying problem type for: {problem}")
    
    system_prompt = """You are a math problem classifier. Analyze the given math problem and determine its primary category.
Return ONLY ONE of these categories without explanation:
- ALGEBRA (solving equations, inequalities, systems of equations)
- CALCULUS (derivatives, integrals, limits)
- GEOMETRY (shapes, angles, areas, volumes)
- ARITHMETIC (basic operations, fractions, decimals)
- STATISTICS (mean, median, mode, standard deviation)
- PROBABILITY (chance, combinations, permutations)
- PHYSICS (motion, forces, energy problems)
- WORD_PROBLEM (story problems that need mathematical modeling)"""

    user_prompt = f"Classify this math problem into one category: {problem}"

    try:
        response = llm_service.generate_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1
        )
        
        content = response.get("content", "").strip().upper()
        logger.info(f"Problem classified as: {content}")
        
        # Try to match with enum
        for problem_type in ProblemType:
            if problem_type.name in content:
                return problem_type
        
        # Default to unknown if no match
        return ProblemType.UNKNOWN
    
    except Exception as e:
        logger.error(f"Error classifying problem: {str(e)}", exc_info=True)
        return ProblemType.UNKNOWN

def extract_variables(llm_service, problem):
    """Extract key variables from the problem."""
    logger.info(f"Extracting variables from problem: {problem}")
    
    system_prompt = """You are a math problem analyzer. Extract the key variables from the problem.
Return a comma-separated list of variables with their descriptions, e.g.: "x: unknown value, r: radius, v: velocity".
Only include actual variables that need to be found or used in calculations."""

    user_prompt = f"Extract the key variables from this math problem: {problem}"

    try:
        response = llm_service.generate_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1
        )
        
        content = response.get("content", "").strip()
        logger.info(f"Extracted variables: {content}")
        
        # Parse the response into a list of variables
        variables = []
        for var in content.split(','):
            var = var.strip()
            if ':' in var:
                var_name, var_desc = var.split(':', 1)
                variables.append((var_name.strip(), var_desc.strip()))
            elif var:
                variables.append((var, ""))
        
        return variables
    
    except Exception as e:
        logger.error(f"Error extracting variables: {str(e)}", exc_info=True)
        return []

def interactive_tutoring_session():
    """Run an interactive math tutoring session."""
    print_header()
    
    # Initialize services
    llm_service, meta_agent, math_agent = init_services()
    
    if not llm_service or not meta_agent or not math_agent:
        print(f"{Colors.RED}Could not initialize required services. Please check your API key and try again.{Colors.END}")
        return False
    
    print(f"{Colors.BOLD}Welcome to the Interactive Math Tutor!{Colors.END}")
    print(f"{Colors.CYAN}I can help you solve math problems step-by-step.{Colors.END}")
    print(f"{Colors.CYAN}You can enter a problem and choose how you want to proceed.{Colors.END}")
    print()
    
    # Main tutoring loop
    while True:
        # Create a new session state for each problem
        session = TutorSession()
        
        # Get the problem
        print(f"{Colors.BOLD}Enter a math problem (or 'q' to quit):{Colors.END}")
        problem = input("> ").strip()
        
        if problem.lower() in ('q', 'quit', 'exit'):
            break
        
        if not problem:
            print(f"{Colors.YELLOW}Please enter a valid math problem.{Colors.END}")
            continue
        
        # Update session with problem
        session.problem = problem
        print_divider(f"Problem: {problem}")
        
        # Initial problem comprehension
        comprehension = get_question_comprehension(llm_service, problem)
        print(f"{Colors.CYAN}Problem Analysis:{Colors.END}\n{comprehension}\n")
        session.comprehension_shown = True
        
        # Extract problem type and variables
        session.update_problem_info(llm_service)
        if session.variables:
            var_display = ", ".join([f"{name}: {desc}" for name, desc in session.variables])
            print(f"{Colors.CYAN}Key Variables:{Colors.END} {var_display}\n")
        
        print(f"{Colors.CYAN}Problem Type:{Colors.END} {session.problem_type.value.capitalize()}\n")
        
        # Solve to get the correct answer (but don't display)
        from app.math_services.models.state import MathState
        solved_state = math_agent.solve(MathState(question=problem, student_answer=""))
        session.correct_answer = solved_state.correct_answer
        logger.info(f"Correct answer: {session.correct_answer}")
        
        # Learning path selection - always ask first what they want to do
        while True:
            # Present options based on current state
            if not session.has_answer() and not session.dot_shown and not session.cot_shown and session.hint_count == 0:
                # Initial options when first entering the problem
                initial_options = [
                    "I want to try solving this myself",
                    "I need a hint",
                    "Show me how to solve this (Draft of Thought)",
                    "Show me detailed solution steps (Chain of Thought)"
                ]
                
                print_options(initial_options, "How would you like to proceed?")
                choice = input("Enter your choice (1-4): ").strip()
                
                # *** FIX THE INPUT HANDLING ***
                # Simplify choice handling - just check if it's a number and in range
                if choice in ["1", "2", "3", "4"]:
                    choice_num = int(choice)
                else:
                    # Non-numeric input, provide helpful message and repeat
                    print(f"{Colors.YELLOW}Please enter a number from 1-4 corresponding to your choice.{Colors.END}")
                    continue
                
                # Process choice based on numeric input
                if choice_num == 1:
                    # Student wants to try solving
                    print(f"\n{Colors.CYAN}Great! Please enter your answer when ready.{Colors.END}")
                    student_answer = input("Your answer: ").strip()
                    
                    if student_answer:
                        session.student_answer = student_answer
                        session.attempts += 1
                        
                        # Assess the student's answer
                        feedback, assessed_state = assess_answer(math_agent, problem, student_answer, session.correct_answer)
                        
                        # Display feedback
                        print_feedback(feedback)
                        
                        # If incorrect, offer options
                        if not feedback.get("is_correct", False):
                            post_feedback_options = [
                                "I want a hint",
                                "Show me the Draft of Thought solution",
                                "Show me the detailed Chain of Thought solution",
                                "Let me try again"
                            ]
                            
                            print_options(post_feedback_options, "What would you like to do next?")
                            pf_choice = input("Enter your choice (1-4): ").strip()
                            
                            # *** SIMPLIFIED INPUT HANDLING ***
                            if pf_choice == "1":
                                # Student wants a hint
                                hint, hint_state = generate_hint(math_agent, problem, session.hint_count + 1, session.hints)
                                session.hints.append(hint)
                                session.hint_count += 1
                                
                                print(f"\n{Colors.YELLOW}Hint {session.hint_count}:{Colors.END} {hint}")
                                
                            elif pf_choice == "2":
                                # Show DoT solution
                                dot_steps, dot_verification, dot_confidence, dot_state = generate_dot_reasoning(math_agent, problem)
                                print_steps(dot_steps, "Draft of Thought (DoT) Solution", show_typing=True)
                                print_verification_info(dot_verification)
                                print_confidence_bar(dot_confidence, "DoT Confidence")
                                session.dot_shown = True
                                
                            elif pf_choice == "3":
                                # Show CoT solution
                                cot_steps, cot_verification, cot_confidence, cot_state = generate_cot_reasoning(math_agent, problem)
                                print_steps(cot_steps, "Chain of Thought (CoT) Solution", show_typing=True)
                                print_verification_info(cot_verification)
                                print_confidence_bar(cot_confidence, "CoT Confidence")
                                session.cot_shown = True
                                
                            elif pf_choice == "4":
                                print(f"\n{Colors.CYAN}Please enter your new answer:{Colors.END}")
                                new_answer = input("Your answer: ").strip()
                                
                                if new_answer:
                                    session.student_answer = new_answer
                                    session.attempts += 1
                                    
                                    feedback, assessed_state = assess_answer(math_agent, problem, new_answer, session.correct_answer)
                                    print_feedback(feedback)
                            else:
                                print(f"{Colors.YELLOW}Invalid choice. Returning to main menu.{Colors.END}")
                        else:
                            # They got it correct, celebrate and ask if they want explanation
                            if not session.dot_shown and not session.cot_shown:
                                post_correct_options = [
                                    "Show me the solution steps to confirm my approach",
                                    "I'm ready for another problem"
                                ]
                                
                                print_options(post_correct_options, "What would you like to do next?")
                                pc_choice = input("Enter your choice (1-2): ").strip()
                                
                                if pc_choice == "1":
                                    dot_steps, dot_verification, dot_confidence, dot_state = generate_dot_reasoning(math_agent, problem)
                                    print_steps(dot_steps, "Draft of Thought (DoT) Solution", show_typing=True)
                                    print_verification_info(dot_verification)
                                    print_confidence_bar(dot_confidence, "DoT Confidence")
                                    session.dot_shown = True
                                else:
                                    # They want to move on to a new problem
                                    break
                    else:
                        print(f"{Colors.YELLOW}No answer provided. Let me help you with this problem.{Colors.END}")
                
                elif choice_num == 2:
                    # Student wants a hint
                    hint, hint_state = generate_hint(math_agent, problem, session.hint_count + 1, session.hints)
                    session.hints.append(hint)
                    session.hint_count += 1
                    
                    print(f"\n{Colors.YELLOW}Hint {session.hint_count}:{Colors.END} {hint}")
                    
                    # After providing hint, ask if they want to try or get more help
                    post_hint_options = [
                        "Let me try now",
                        "I need another hint",
                        "Show me the Draft of Thought solution",
                        "Show me the detailed Chain of Thought solution"
                    ]
                    
                    print_options(post_hint_options, "What would you like to do next?")
                    ph_choice = input("Enter your choice (1-4): ").strip()
                    
                    # *** SIMPLIFIED INPUT HANDLING ***
                    if ph_choice == "1":
                        print(f"\n{Colors.CYAN}Great! Please enter your answer:{Colors.END}")
                        student_answer = input("Your answer: ").strip()
                        
                        if student_answer:
                            session.student_answer = student_answer
                            session.attempts += 1
                            
                            feedback, assessed_state = assess_answer(math_agent, problem, student_answer, session.correct_answer)
                            print_feedback(feedback)
                    
                    elif ph_choice == "2":
                        hint, hint_state = generate_hint(math_agent, problem, session.hint_count + 1, session.hints)
                        session.hints.append(hint)
                        session.hint_count += 1
                        
                        print(f"\n{Colors.YELLOW}Hint {session.hint_count}:{Colors.END} {hint}")
                        
                    elif ph_choice == "3":
                        # Show DoT solution
                        dot_steps, dot_verification, dot_confidence, dot_state = generate_dot_reasoning(math_agent, problem)
                        print_steps(dot_steps, "Draft of Thought (DoT) Solution", show_typing=True)
                        print_verification_info(dot_verification)
                        print_confidence_bar(dot_confidence, "DoT Confidence")
                        session.dot_shown = True
                        
                    elif ph_choice == "4":
                        # Show CoT solution
                        cot_steps, cot_verification, cot_confidence, cot_state = generate_cot_reasoning(math_agent, problem)
                        print_steps(cot_steps, "Chain of Thought (CoT) Solution", show_typing=True)
                        print_verification_info(cot_verification)
                        print_confidence_bar(cot_confidence, "CoT Confidence")
                        session.cot_shown = True
                    else:
                        print(f"{Colors.YELLOW}Invalid choice. Returning to main menu.{Colors.END}")
                
                elif choice_num == 3:
                    # Student wants DoT solution
                    dot_steps, dot_verification, dot_confidence, dot_state = generate_dot_reasoning(math_agent, problem)
                    print_steps(dot_steps, "Draft of Thought (DoT) Solution", show_typing=True)
                    print_verification_info(dot_verification)
                    print_confidence_bar(dot_confidence, "DoT Confidence")
                    session.dot_shown = True
                    
                    # After showing DoT, ask if they want detailed steps
                    if not session.cot_shown:
                        post_dot_options = [
                            "I'd like to see the detailed Chain of Thought solution",
                            "This is sufficient, let's continue"
                        ]
                        
                        print_options(post_dot_options, "Would you like to see more detailed steps?")
                        pd_choice = input("Enter your choice (1-2): ").strip()
                        
                        if pd_choice == "1":
                            # Show CoT solution
                            cot_steps, cot_verification, cot_confidence, cot_state = generate_cot_reasoning(math_agent, problem)
                            print_steps(cot_steps, "Chain of Thought (CoT) Solution", show_typing=True)
                            print_verification_info(cot_verification)
                            print_confidence_bar(cot_confidence, "CoT Confidence")
                            session.cot_shown = True
                
                elif choice_num == 4:
                    # Student wants CoT solution
                    cot_steps, cot_verification, cot_confidence, cot_state = generate_cot_reasoning(math_agent, problem)
                    print_steps(cot_steps, "Chain of Thought (CoT) Solution", show_typing=True)
                    print_verification_info(cot_verification)
                    print_confidence_bar(cot_confidence, "CoT Confidence")
                    session.cot_shown = True
                
            else:
                # They've already started interacting with the problem
                # Give them actions based on current state
                current_options = []
                
                if not session.has_answer():
                    current_options.append("Try solving this problem")
                    
                if session.hint_count < 3:
                    hint_text = "Get a hint" if session.hint_count == 0 else "Get another hint"
                    current_options.append(hint_text)
                
                if not session.dot_shown:
                    current_options.append("Show Draft of Thought solution")
                    
                if not session.cot_shown:
                    current_options.append("Show detailed Chain of Thought solution")
                
                current_options.append("Try a new problem")
                
                # Only ask for next action if they've done something already
                # and there are still meaningful options
                if len(current_options) > 1 and (session.has_answer() or session.hint_count > 0 or 
                                               session.dot_shown or session.cot_shown):
                    print_options(current_options, "What would you like to do next?")
                    next_choice = input(f"Enter your choice (1-{len(current_options)}): ").strip()
                    
                    # *** SIMPLIFIED INPUT HANDLING ***
                    # Try to parse as number and check range
                    try:
                        next_choice_num = int(next_choice)
                        if 1 <= next_choice_num <= len(current_options):
                            next_action = current_options[next_choice_num - 1]
                        else:
                            print(f"{Colors.YELLOW}Please enter a number between 1 and {len(current_options)}.{Colors.END}")
                            continue
                    except ValueError:
                        print(f"{Colors.YELLOW}Please enter a valid number.{Colors.END}")
                        continue
                    
                    # Process next action
                    if "try solving" in next_action.lower():
                        print(f"\n{Colors.CYAN}Please enter your answer:{Colors.END}")
                        answer = input("Your answer: ").strip()
                        
                        if answer:
                            session.student_answer = answer
                            session.attempts += 1
                            
                            feedback, assessed_state = assess_answer(math_agent, problem, answer, session.correct_answer)
                            print_feedback(feedback)
                    
                    elif "hint" in next_action.lower():
                        hint, hint_state = generate_hint(math_agent, problem, session.hint_count + 1, session.hints)
                        session.hints.append(hint)
                        session.hint_count += 1
                        
                        print(f"\n{Colors.YELLOW}Hint {session.hint_count}:{Colors.END} {hint}")
                    
                    elif "draft" in next_action.lower() or "dot" in next_action.lower():
                        dot_steps, dot_verification, dot_confidence, dot_state = generate_dot_reasoning(math_agent, problem)
                        print_steps(dot_steps, "Draft of Thought (DoT) Solution", show_typing=True)
                        print_verification_info(dot_verification)
                        print_confidence_bar(dot_confidence, "DoT Confidence")
                        session.dot_shown = True
                    
                    elif "chain" in next_action.lower() or "cot" in next_action.lower() or "detailed" in next_action.lower():
                        cot_steps, cot_verification, cot_confidence, cot_state = generate_cot_reasoning(math_agent, problem)
                        print_steps(cot_steps, "Chain of Thought (CoT) Solution", show_typing=True)
                        print_verification_info(cot_verification)
                        print_confidence_bar(cot_confidence, "CoT Confidence")
                        session.cot_shown = True
                    
                    elif "new" in next_action.lower() or "another" in next_action.lower() or "different" in next_action.lower():
                        break  # Break the inner loop to get a new problem
                else:
                    # If they've seen all possible tutoring options, ask if they want to continue
                    print("\nWould you like to try another problem? (y/n)")
                    continue_response = input().lower()
                    if continue_response not in ['y', 'yes', 'sure', 'yeah']:
                        return True  # End the tutoring session
                    break  # Break the inner loop to get a new problem
        
        # Ask if they want to try another problem (only reaches here when inner loop breaks)
        if session.has_answer() or session.dot_shown or session.cot_shown or session.hint_count > 0:
            print("\nPress Enter to continue with another problem or enter 'q' to quit...")
            exit_check = input().lower()
            if exit_check in ('q', 'quit', 'exit'):
                break
    
    print(f"\n{Colors.BLUE}Thank you for using the Interactive Math Tutor!{Colors.END}")
    return True

def main():
    """Main entry point for the math tutor demo."""
    try:
        interactive_tutoring_session()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Demo interrupted by user.{Colors.END}")
    except Exception as e:
        logger.error(f"Unexpected error in demo: {str(e)}", exc_info=True)
        print(f"\n{Colors.RED}An unexpected error occurred: {str(e)}{Colors.END}")
        print(f"{Colors.RED}See math_tutor_demo.log for details.{Colors.END}")
    
    print(f"\n{Colors.BLUE}Exiting Math Tutor Demo.{Colors.END}")

if __name__ == "__main__":
    main() 