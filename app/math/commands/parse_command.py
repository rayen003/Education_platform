"""
Math Parse Question Command.

This module contains the command for parsing mathematical questions.
"""

import json
import logging
import traceback
import re
from typing import Any, Dict, List, Optional

# Define a base command class since utilities folder is deleted
class BaseMathCommand:
    """Base class for all math commands."""
    
    def __init__(self, agent=None):
        """Initialize the command."""
        self.agent = agent
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the command.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        self.logger.info(f"Executing {self.__class__.__name__}")
        
        try:
            return self._execute_core(state)
        except Exception as e:
            self.logger.error(f"Error executing {self.__class__.__name__}: {e}")
            self.logger.error(traceback.format_exc())
            
            # Add error to state
            if "errors" not in state:
                state["errors"] = []
            
            state["errors"].append({
                "command": self.__class__.__name__,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
            return state
    
    def _execute_core(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core execution logic to be implemented by subclasses.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        raise NotImplementedError("Subclasses must implement _execute_core")

# Define utility function that was in the deleted utilities folder
def extract_mathematical_entities(text: str) -> Dict[str, Any]:
    """
    Extract mathematical entities from text.
    
    Args:
        text: Text to extract entities from
        
    Returns:
        Dictionary of extracted entities
    """
    # Simple implementation to extract variables and equations
    variables = []
    equations = []
    
    # Extract variables (look for single letters)
    var_matches = re.findall(r'\b([a-zA-Z])\b', text)
    if var_matches:
        variables = list(set(var_matches))
    
    # Extract equations (look for equals sign)
    if "=" in text:
        eq_matches = re.findall(r'([^.;]+=[^.;]+)', text)
        if eq_matches:
            equations = [eq.strip() for eq in eq_matches]
    
    return {
        "variables": variables,
        "equations": equations,
        "type": "symbolic" if equations else "word_problem",
        "target_variable": variables[0] if variables else None
    }

logger = logging.getLogger(__name__)

class MathParseQuestionCommand(BaseMathCommand):
    """Command to parse a math question and identify variables and equations."""
    
    def _execute_core(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the question to identify variables and equations.
        
        Args:
            state: Current state with question
            
        Returns:
            Updated state with parsed question
        """
        # Get the question from the state
        question = state.get("question", "")
        self.logger.info(f"Parsing question: {question}")
        
        # Parse the question
        parsed_data = self._parse_question(question)
        
        # Update the state with the parsed data
        if "analysis" not in state:
            state["analysis"] = {}
        
        state["analysis"]["parsed_question"] = parsed_data
        
        return state
    
    def _parse_question(self, question: str) -> Dict[str, Any]:
        """
        Parse the question using available methods.
        
        Args:
            question: Question to parse
            
        Returns:
            Parsed data
        """
        # Try to use LLM for parsing if available
        if hasattr(self.agent, 'llm_service') and self.agent.llm_service:
            try:
                return self._parse_with_llm(question)
            except Exception as e:
                self.logger.error(f"Error parsing with LLM: {e}")
                self.logger.info("Using basic parsing as fallback")
        else:
            self.logger.info("Using basic parsing as fallback")
        
        # Fallback to basic parsing
        parsed_data = extract_mathematical_entities(question)
        parsed_data["explanation"] = "Basic parsing fallback"
        
        self.logger.info(f"Fallback parsed data: {parsed_data}")
        
        return parsed_data
    
    def _parse_with_llm(self, question: str) -> Dict[str, Any]:
        """
        Parse the question using LLM.
        
        Args:
            question: Question to parse
            
        Returns:
            Parsed data
        """
        # Define the system prompt
        system_prompt = """
        You are a mathematical parsing assistant. Your task is to analyze a mathematical question and extract key information.
        
        For the given question, identify:
        1. The type of problem (symbolic, word problem, etc.)
        2. Variables involved
        3. Equations or constraints
        4. The target variable to solve for
        
        Format your response as a JSON object with the following structure:
        {
            "type": "symbolic|word_problem",
            "variables": ["x", "y", ...],
            "equations": ["2x + 3 = 7", ...],
            "target_variable": "x",
            "explanation": "Brief explanation of the problem"
        }
        """
        
        # Define the user prompt
        user_prompt = f"Question: {question}\n\nParse this mathematical question and provide the structured output."
        
        # Get the response from the LLM
        response = self.agent.llm_service.generate_completion(system_prompt, user_prompt)
        
        # Extract the JSON from the response
        try:
            # If response is a string, try to parse it as JSON
            if isinstance(response, str):
                parsed_data = json.loads(response)
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
                        parsed_data = json.loads(json_str)
                    else:
                        raise ValueError("No JSON found in LLM response")
                elif isinstance(content, dict):
                    parsed_data = content
                else:
                    raise ValueError(f"Unexpected LLM response format: {type(content)}")
            else:
                raise ValueError(f"Unexpected LLM response format: {type(response)}")
            
            # Validate the parsed data
            required_keys = ["type", "variables", "equations"]
            for key in required_keys:
                if key not in parsed_data:
                    parsed_data[key] = []
            
            return parsed_data
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            self.logger.error(f"LLM response: {response}")
            raise ValueError(f"Failed to parse LLM response: {e}")
