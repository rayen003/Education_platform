import os
import re
import sympy
import numpy as np
from openai import OpenAI
from typing import Dict, Any, TypedDict, Annotated, Literal, Optional, List, Union
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from sympy import symbols, sympify, solve, Eq, simplify
from sympy.parsing.sympy_parser import parse_expr

# Load environment variables
load_dotenv()

# Define the state schema
class MathState(TypedDict):
    question: str
    student_answer: str
    correct_answer: Optional[str]
    symbolic_answer: Optional[Any]
    analysis: Dict[str, Any]
    feedback: str
    proximity_score: Optional[float]
    hint_count: int
    hints: List[str]
    needs_hint: bool

class MathAgent:
    """
    Agent responsible for analyzing mathematical calculations in student answers.
    """
    def __init__(self):
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Create the state graph
        self.graph = StateGraph(MathState)
        
        # Add nodes
        self.graph.add_node("parse_question", self.parse_question)
        self.graph.add_node("solve_symbolically", self.solve_symbolically)
        self.graph.add_node("analyze", self.analyze_calculation)
        self.graph.add_node("assess_proximity", self.assess_proximity)
        self.graph.add_node("generate_hint", self.generate_hint)
        self.graph.add_node("generate_feedback", self.generate_feedback)
        
        # Add conditional edges
        self.graph.add_edge("parse_question", "solve_symbolically")
        self.graph.add_edge("solve_symbolically", "analyze")
        self.graph.add_edge("analyze", "assess_proximity")
        
        # Add conditional branching based on needs_hint and hint_count
        self.graph.add_conditional_edges(
            "assess_proximity",
            self.decide_next_step,
            {
                "hint": "generate_hint",
                "feedback": "generate_feedback"
            }
        )
        
        # Add edge from generate_hint back to assess_proximity for iteration
        self.graph.add_edge("generate_hint", "assess_proximity")
        
        # Set the entry point
        self.graph.set_entry_point("parse_question")
        
        # Compile the graph
        self.workflow = self.graph.compile()
        
    def parse_question(self, state: MathState) -> MathState:
        """
        Parse the math question to identify variables and equations.
        """
        question = state["question"]
        print(f"Parsing question: {question}")
        
        # Initialize state fields if not present
        if "analysis" not in state:
            state["analysis"] = {}
        if "hint_count" not in state:
            state["hint_count"] = 0
        if "hints" not in state:
            state["hints"] = []
        if "needs_hint" not in state:
            state["needs_hint"] = False
        if "proximity_score" not in state:
            state["proximity_score"] = None
            
        # Use OpenAI to extract the mathematical problem in a format suitable for SymPy
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a math parsing assistant. Extract the mathematical problem from the question 
                    and convert it to a format suitable for symbolic computation with SymPy.
                    
                    For equations, express them in the form: equation = "x + 2 = 5" or "2*x - 3 = 7"
                    For expressions, express them in the form: expression = "x + 2" or "2*x - 3"
                    For variables, list them as: variables = ["x", "y"]
                    
                    Return your response in a format that can be directly used in Python code.
                    """
                },
                {
                    "role": "user",
                    "content": f"Question: {question}"
                }
            ]
        )
        
        parsed_info = response.choices[0].message.content.strip()
        print(f"Parsed info: {parsed_info}")
        
        # Store the parsed information in the state
        state["analysis"]["parsed_info"] = parsed_info
        
        return state
    
    def solve_symbolically(self, state: MathState) -> MathState:
        """
        Solve the math problem symbolically using SymPy.
        """
        parsed_info = state["analysis"]["parsed_info"]
        question = state["question"]
        
        try:
            # Extract equation or expression from parsed_info
            if "equation =" in parsed_info:
                equation_match = re.search(r'equation\s*=\s*"([^"]+)"', parsed_info)
                if equation_match:
                    equation_str = equation_match.group(1)
                    print(f"Extracted equation: {equation_str}")
                    
                    # Check if it's a simple arithmetic expression without equals sign
                    if "=" not in equation_str:
                        # Treat it as a simple expression
                        expr = parse_expr(equation_str.strip())
                        result = float(expr.evalf())
                        # Store the result
                        state["correct_answer"] = str(result)
                        state["symbolic_answer"] = result
                        print(f"Evaluated expression: {result}")
                    else:
                        # Parse the equation using SymPy
                        left_side, right_side = equation_str.split("=")
                        left_expr = parse_expr(left_side.strip())
                        right_expr = parse_expr(right_side.strip())
                        equation = Eq(left_expr, right_expr)
                        
                        # Extract variables
                        variables = list(equation.free_symbols)
                        if variables:
                            # Solve the equation
                            solution = solve(equation, variables[0])
                            if solution:
                                symbolic_answer = solution[0]
                                correct_answer = str(symbolic_answer)
                                print(f"Symbolic solution: {symbolic_answer}")
                                print(f"Correct answer: {correct_answer}")
                                
                                # Store the correct answer and symbolic answer in the state
                                state["correct_answer"] = correct_answer
                                state["symbolic_answer"] = symbolic_answer
            elif "expression =" in parsed_info:
                expression_match = re.search(r'expression\s*=\s*"([^"]+)"', parsed_info)
                if expression_match:
                    expression_str = expression_match.group(1)
                    print(f"Extracted expression: {expression_str}")
                    
                    # Parse the expression using SymPy
                    expr = parse_expr(expression_str.strip())
                    simplified_expr = simplify(expr)
                    
                    # Store the correct answer and symbolic answer in the state
                    state["correct_answer"] = str(simplified_expr)
                    state["symbolic_answer"] = simplified_expr
        except Exception as e:
            print(f"Error solving symbolically: {e}")
            # If symbolic solving fails, use OpenAI to get the correct answer
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a math solving assistant. Solve the given math problem and provide the correct answer."
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}"
                    }
                ]
            )
            
            correct_answer = response.choices[0].message.content.strip()
            state["correct_answer"] = correct_answer
            state["symbolic_answer"] = None
        
        return state
    
    def analyze_calculation(self, state: MathState) -> MathState:
        """
        Analyze the mathematical calculations in a student's answer.
        """
        question = state["question"]
        student_answer = state["student_answer"]
        correct_answer = state.get("correct_answer")
        
        print(f"Analyzing Question: {question}")
        print(f"Student Answer: {student_answer}")
        print(f"Correct Answer: {correct_answer}")
        
        # Use ChatGPT-3.5 to assess correctness
        is_correct = self._check_answer(question, student_answer, correct_answer)
        
        # Update state with analysis results
        state["analysis"]["is_correct"] = is_correct
        state["analysis"]["details"] = "The student's answer is correct." if is_correct else "The student's answer is incorrect."
        
        return state
    
    def generate_feedback(self, state: MathState) -> MathState:
        """
        Generate detailed feedback based on the analysis results.
        """
        question = state["question"]
        student_answer = state["student_answer"]
        correct_answer = state.get("correct_answer", "")
        is_correct = state["analysis"]["is_correct"]
        hints = state.get("hints", [])
        proximity_score = state.get("proximity_score", None)
        symbolic_answer = state.get("symbolic_answer", None)
        
        # Generate detailed feedback using OpenAI
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a supportive math tutor providing feedback on student answers.
                    
                    Follow these guidelines:
                    1. Start with positive reinforcement, acknowledging what the student did correctly.
                    2. If the answer is incorrect, explain the error and provide the correct approach.
                    3. Include a brief explanation of the relevant mathematical concept.
                    4. Offer a concise, step-by-step solution.
                    5. End with encouragement for future problems.
                    
                    Format your feedback using LaTeX for mathematical expressions where appropriate, using $...$ for inline math and $$...$$
                    for display math. For example, instead of writing "x^2 + 3x + 2", write "$x^2 + 3x + 2$".
                    
                    If hints were provided to the student, reference them in your feedback and build upon them.
                    If a proximity score was given, tailor your feedback to acknowledge how close they were to the correct answer.
                    
                    Please ensure that your response is properly formatted with LaTeX syntax.
                    Keep your feedback concise, clear, and educational.
                    """
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nStudent Answer: {student_answer}\nCorrect Answer: {correct_answer}\nIs Correct: {is_correct}\nHints Provided: {hints if hints else 'None'}\nProximity Score: {proximity_score if proximity_score is not None else 'Not available'}/10"
                }
            ]
        )
        
        # Extract the detailed feedback
        feedback = response.choices[0].message.content.strip()
        
        # Update state with feedback
        state["feedback"] = feedback
        print(f"Generated Feedback: {feedback}")
        
        return state
    
    def assess_proximity(self, state: MathState) -> MathState:
        """
        Assess how close the student's answer is to the correct answer.
        """
        question = state["question"]
        student_answer = state["student_answer"]
        is_correct = state["analysis"]["is_correct"]
        
        # If the answer is correct, no need for proximity assessment
        if is_correct:
            state["proximity_score"] = 10
            state["needs_hint"] = False
            return state
        
        # Ensure we have a correct answer
        if "correct_answer" not in state or not state.get("correct_answer"):
            # Get the correct answer using OpenAI
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a math solving assistant. Solve the given math problem and provide the correct answer."
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}"
                    }
                ]
            )
            
            correct_answer = response.choices[0].message.content.strip()
            state["correct_answer"] = correct_answer
            print(f"Generated correct answer for proximity assessment: {correct_answer}")
        else:
            correct_answer = state["correct_answer"]
        
        # Use OpenAI to assess proximity on a scale of 0-10
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a math assessment assistant. Evaluate how close the student's answer is to the correct answer 
                    on a scale of 0-10, where 0 is completely wrong and 10 is completely correct.
                    
                    Consider the following factors:
                    1. Numerical proximity (if applicable)
                    2. Conceptual understanding
                    3. Approach used
                    
                    Return only a number between 0 and 10.
                    """
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nStudent Answer: {student_answer}\nCorrect Answer: {correct_answer}"
                }
            ]
        )
        
        # Extract the proximity score
        proximity_score_text = response.choices[0].message.content.strip()
        try:
            proximity_score = float(proximity_score_text)
            # Ensure the score is between 0 and 10
            proximity_score = max(0, min(10, proximity_score))
        except ValueError:
            # Default to a mid-range score if parsing fails
            proximity_score = 5
        
        print(f"Proximity Score: {proximity_score}/10")
        
        # Update state with proximity score
        state["proximity_score"] = proximity_score
        
        # Determine if a hint is needed (score < 5)
        state["needs_hint"] = proximity_score < 5 and state["hint_count"] < 3
        
        return state
    
    def decide_next_step(self, state: MathState) -> Literal["hint", "feedback"]:
        """
        Decide whether to provide a hint or generate feedback.
        """
        needs_hint = state["needs_hint"]
        hint_count = state["hint_count"]
        
        if needs_hint and hint_count < 3:
            return "hint"
        else:
            return "feedback"
    
    def generate_hint(self, state: MathState) -> MathState:
        """
        Generate a hint for the student based on their answer and the correct answer.
        """
        question = state["question"]
        student_answer = state["student_answer"]
        hint_count = state["hint_count"]
        proximity_score = state["proximity_score"]
        
        # Ensure we have a correct answer
        if "correct_answer" not in state or not state["correct_answer"]:
            # Get the correct answer using OpenAI
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a math solving assistant. Solve the given math problem and provide the correct answer."
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}"
                    }
                ]
            )
            
            correct_answer = response.choices[0].message.content.strip()
            state["correct_answer"] = correct_answer
            print(f"Generated correct answer: {correct_answer}")
        else:
            correct_answer = state["correct_answer"]
        
        # Generate a hint using OpenAI
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a supportive math tutor providing hints to students. 
                    Based on the student's answer and the correct answer, provide a helpful hint that guides them 
                    toward the correct solution without giving it away completely.
                    
                    For the first hint (hint_count=0), provide a general conceptual hint.
                    For the second hint (hint_count=1), provide a more specific procedural hint.
                    For the third hint (hint_count=2), provide a very specific hint that almost gives away the answer.
                    
                    Format your hint using LaTeX for mathematical expressions where appropriate, using $...$ for inline math and $$...$$
                    for display math. Keep the hint concise and clear.
                    
                    Please ensure that your response is properly formatted with LaTeX syntax.
                    """
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nStudent Answer: {student_answer}\nCorrect Answer: {correct_answer}\nHint Count: {hint_count}\nProximity Score: {proximity_score}/10"
                }
            ]
        )
        
        # Extract the hint
        hint = response.choices[0].message.content.strip()
        
        # Update state with the hint
        state["hints"].append(hint)
        state["hint_count"] += 1
        
        # Check if we've reached the maximum number of hints
        if state["hint_count"] >= 3:
            state["needs_hint"] = False
        
        print(f"Generated Hint #{state['hint_count']}: {hint}")
        
        return state
    
    def _check_answer(self, question: str, student_answer: str, correct_answer: str = None) -> bool:
        """
        Use ChatGPT-3.5 to check if the student's answer is correct.
        """
        # Try to use SymPy for exact comparison if we have symbolic answers
        if correct_answer:
            try:
                # Try to parse both answers as expressions
                student_expr = parse_expr(student_answer.strip())
                correct_expr = parse_expr(correct_answer.strip())
                
                # Check if they are equivalent
                if simplify(student_expr - correct_expr) == 0:
                    print("SymPy verified the answer is correct")
                    return True
            except Exception as e:
                print(f"SymPy comparison failed: {e}")
        
        # Fall back to ChatGPT-3.5 for assessment
        messages = [
            {
                "role": "system",
                "content": """You are a math evaluation assistant. Analyze the student's answer to determine if it is correct.
                
                Follow these steps:
                1. Solve the math problem yourself first.
                2. Compare your solution with the student's answer.
                3. Consider different formats of correct answers (e.g., '2' vs '2.0' vs 'x=2').
                4. Respond with 'True' if the student's answer is correct, or 'False' if it's incorrect.
                
                Be precise in your evaluation and give the student the benefit of the doubt if their answer is correct but formatted differently.
                """
            }
        ]
        
        if correct_answer:
            messages.append({
                "role": "user",
                "content": f"Question: {question}\nStudent Answer: {student_answer}\nCorrect Answer: {correct_answer}"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Question: {question}\nStudent Answer: {student_answer}"
            })
        
        # Query ChatGPT-3.5 to assess correctness
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        # Extract the response and convert to boolean
        result = response.choices[0].message.content.strip().lower() == "true"
        print(f"Evaluation result: {result}")
        return result
    
    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the student's answer and generate feedback.
        """
        # Initialize analysis and feedback if not present
        if "analysis" not in state:
            state["analysis"] = {}
        if "feedback" not in state:
            state["feedback"] = ""
            
        # Run the workflow
        result = self.workflow.invoke(state)
        
        # Return the final state
        return result

# Simple test case
if __name__ == "__main__":
    # Instantiate the MathAgent
    math_agent = MathAgent()

    # Define test cases
    test_cases = [
        {
            "question": "What is 5 + 3?",
            "student_answer": "8"
        },
        {
            "question": "What is 10 - 2?",
            "student_answer": "7"
        },
        {
            "question": "Solve for x in the equation 2x + 5 = 15",
            "student_answer": "x = 4"
        },
        {
            "question": "Solve for x in the equation 3x - 7 = 8",
            "student_answer": "x = 4"
        },
        {
            "question": "Simplify the expression (x^2 - 4)/(x - 2) for x ≠ 2",
            "student_answer": "x + 2"
        },
        {
            "question": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
            "student_answer": "3x^2 + 4x - 5"
        }
    ]

    # Analyze each test case
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        try:
            # Initialize state with required fields
            state = {
                "question": test_case["question"],
                "student_answer": test_case["student_answer"],
                "analysis": {},
                "hint_count": 0,
                "hints": [],
                "needs_hint": False
            }
            
            # Run the analysis
            result = math_agent.analyze(state)
            
            # Display results
            print(f"Math Agent Feedback: {result['feedback']}")
            
            # Display hints if any were generated
            if result.get('hints') and len(result['hints']) > 0:
                print("\nHints provided:")
                for idx, hint in enumerate(result['hints']):
                    print(f"Hint #{idx+1}: {hint}")
                    
            # Display proximity score if available
            if 'proximity_score' in result and result['proximity_score'] is not None:
                print(f"Proximity Score: {result['proximity_score']}/10")
                
        except Exception as e:
            print(f"Error processing test case {i+1}: {e}")