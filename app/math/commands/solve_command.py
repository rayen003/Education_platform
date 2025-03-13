"""
Math Solve Command.

This module contains the command for solving math problems symbolically.
"""

import logging
import json
import traceback
import re
from typing import Dict, Any, List, Optional
import sympy
from sympy import symbols, sympify, solve, Eq, simplify
from sympy.parsing.sympy_parser import parse_expr

from app.math.services.llm.base_service import BaseLLMService

logger = logging.getLogger(__name__)

class MathSolveSymbolicallyCommand:
    """Command for solving math problems symbolically."""
    
    def __init__(self, agent):
        """
        Initialize the solve command.
        
        Args:
            agent: The agent instance that will use this command
        """
        self.agent = agent
        self.llm_service = agent.llm_service
        self.meta_agent = agent.meta_agent
        logger.info("Initialized MathSolveSymbolicallyCommand")
    
    def _execute_core(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the problem symbolically.
        
        Args:
            state: Current state with parsed question
            
        Returns:
            Updated state with solution
        """
        self.logger.info("Solving symbolic problem")
        
        # Get the parsed question from the state
        parsed_question = state.get("analysis", {}).get("parsed_question", {})
        
        # Get the equations and variables
        equations = parsed_question.get("equations", [])
        variables = parsed_question.get("variables", [])
        target_variable = parsed_question.get("target_variable")
        
        if not equations:
            self.logger.warning("No equations found in parsed question")
            state["correct_answer"] = "Unknown"
            return state
        
        if not variables:
            self.logger.warning("No variables found in parsed question")
            state["correct_answer"] = "Unknown"
            return state
        
        # Try to solve with SymPy first
        try:
            solution = self._solve_with_sympy(equations, variables, target_variable)
            state["correct_answer"] = solution
            return state
        except Exception as e:
            self.logger.error(f"Error in SymPy solving: {e}")
        
        # If SymPy fails, try with LLM
        self.logger.warning("Failed to solve with SymPy, trying with LLM")
        
        try:
            if hasattr(self.agent, 'llm_service') and self.agent.llm_service:
                solution = self._solve_with_llm(equations, variables, target_variable)
                state["correct_answer"] = solution
                return state
            else:
                self.logger.warning("No LLM service available for solving")
                state["correct_answer"] = "Unknown"
                return state
        except Exception as e:
            self.logger.error(f"Error in LLM solving: {e}")
            state["correct_answer"] = "Unknown"
            return state
    
    def _solve_with_sympy(self, equations: List[str], variables: List[str], target_variable: Optional[str] = None) -> str:
        """
        Solve the equations using SymPy.
        
        Args:
            equations: List of equation strings
            variables: List of variable names
            target_variable: Variable to solve for
            
        Returns:
            Solution as a string
        """
        # If target_variable is not specified, use the first variable
        if not target_variable and variables:
            target_variable = variables[0]
        
        # Create SymPy symbols for variables
        symbols = {}
        for var in variables:
            symbols[var] = sympy.Symbol(var)
        
        # Parse and solve each equation
        solutions = []
        
        for eq_str in equations:
            # Clean up the equation string
            eq_str = eq_str.replace("Solve for", "").replace("solve for", "").strip()
            
            # Remove any variable specifications from the equation
            eq_str = re.sub(r'^\s*[a-zA-Z]+\s*:', '', eq_str).strip()
            
            # Split the equation at the equals sign
            if "=" in eq_str:
                left_str, right_str = eq_str.split("=", 1)
                
                # Parse the left and right sides
                left_expr = sympy.sympify(left_str.strip())
                right_expr = sympy.sympify(right_str.strip())
                
                # Create the equation
                equation = sympy.Eq(left_expr, right_expr)
                
                # Solve for the target variable
                solution = sympy.solve(equation, symbols[target_variable])
                
                # Format the solution
                solution_str = f"{target_variable} = {solution[0]}" if solution else "No solution found"
                solutions.append(solution_str)
        
        # Return the solutions
        return solutions[0] if solutions else "No solution found"
    
    def _solve_with_llm(self, equations: List[str], variables: List[str], target_variable: Optional[str] = None) -> str:
        """
        Solve the equations using LLM.
        
        Args:
            equations: List of equation strings
            variables: List of variable names
            target_variable: Variable to solve for
            
        Returns:
            Solution as a string
        """
        # If target_variable is not specified, use the first variable
        if not target_variable and variables:
            target_variable = variables[0]
        
        # Create the system prompt
        system_prompt = """
        You are a mathematical solving assistant. Your task is to solve the given equations for the specified variable.
        
        Provide your solution in the following format:
        {
            "solution": "x = 2",
            "steps": ["Step 1: ...", "Step 2: ...", ...],
            "explanation": "Brief explanation of the solution"
        }
        """
        
        # Create the user prompt
        user_prompt = f"Equations: {', '.join(equations)}\nVariables: {', '.join(variables)}\nSolve for: {target_variable}"
        
        # Get the response from the LLM
        response = self.agent.llm_service.generate_completion(system_prompt, user_prompt)
        
        # Extract the solution from the response
        try:
            # If response is a string, try to parse it as JSON
            if isinstance(response, str):
                solution_data = json.loads(response)
            # If response is already a dict, use it directly
            elif isinstance(response, dict) and "content" in response:
                content = response["content"]
                # Extract JSON from the content if needed
                if isinstance(content, str):
                    # Try to find JSON in the string
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        solution_data = json.loads(json_str)
                    else:
                        # Try to extract the solution directly
                        solution_match = re.search(r'([a-zA-Z]+\s*=\s*[^,\n]+)', content)
                        if solution_match:
                            return solution_match.group(1).strip()
                        else:
                            raise ValueError("No solution found in LLM response")
                elif isinstance(content, dict):
                    solution_data = content
                else:
                    raise ValueError(f"Unexpected LLM response format: {type(content)}")
            else:
                raise ValueError(f"Unexpected LLM response format: {type(response)}")
            
            # Extract the solution
            if "solution" in solution_data:
                return solution_data["solution"]
            else:
                # Try to find the solution in the response
                for key, value in solution_data.items():
                    if isinstance(value, str) and "=" in value:
                        return value
                
                raise ValueError("No solution found in LLM response")
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM solution: {e}")
            self.logger.error(f"LLM response: {response}")
            
            # Try to extract the solution directly from the response
            if isinstance(response, dict) and "content" in response:
                content = response["content"]
                if isinstance(content, str):
                    solution_match = re.search(r'([a-zA-Z]+\s*=\s*[^,\n]+)', content)
                    if solution_match:
                        return solution_match.group(1).strip()
            
            # If all else fails, return a default message
            return f"Unable to solve for {target_variable}"
