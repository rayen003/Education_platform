from transformers import pipeline
from typing import List, Dict
from pydantic import BaseModel, Field, validator
import re

# Define Pydantic models
class Concept(BaseModel):
    name: str
    description: str
    difficulty: str = Field(..., regex="^(beginner|intermediate|advanced)$")

class Relationship(BaseModel):
    source: str
    target: str
    type: str = Field(..., regex="^(REQUIRES|HAS_RESOURCE)$")
    strength: int = Field(..., ge=1, le=5)

class SyllabusParser:
    def __init__(self):
        self.extractor = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device="cpu"
        )
        
    def parse_syllabus(self, text: str) -> Dict:
        try:
            concepts = self._extract_concepts(text)
            relationships = self._identify_relationships(concepts, text)
            
            return {
                "concepts": [c.dict() for c in concepts],
                "relationships": [r.dict() for r in relationships],
                "metadata": {
                    "source": text,
                    "version": "1.0"
                }
            }
        except Exception as e:
            print(f"Error parsing syllabus: {e}")
            return {"error": str(e)}

    def _extract_concepts(self, text: str) -> List[Concept]:
        prompt = f"""
        Extract key concepts from this syllabus:
        {text}
        
        Return as a Python list of dictionaries with fields:
        - name: concept name
        - description: brief explanation
        - difficulty: beginner/intermediate/advanced
        
        Example output:
        [{{"name": "Time Value of Money", "description": "Understanding present and future value", "difficulty": "beginner"}}]
        """
        
        response = self.extractor(prompt, max_length=500)
        return self._parse_structured_output(response[0]['generated_text'], Concept)

    def _identify_relationships(self, concepts: List[Concept], text: str) -> List[Relationship]:
        concept_names = [c.name for c in concepts]
        prompt = f"""
        Analyze these concepts: {concept_names}
        From this syllabus: {text}
        
        Identify relationships between concepts.
        Return as a Python list of dictionaries with fields:
        - source: concept name
        - target: concept name
        - type: REQUIRES/HAS_RESOURCE
        - strength: 1-5 (5=strongest)
        
        Example output:
        [{{"source": "Present Value", "target": "Future Value", "type": "REQUIRES", "strength": 3}}]
        """
        
        response = self.extractor(prompt, max_length=500)
        return self._parse_structured_output(response[0]['generated_text'], Relationship)

    def _parse_structured_output(self, text: str, model: BaseModel) -> List[BaseModel]:
        try:
            # Extract the list portion from the response
            match = re.search(r'\[.*\]', text)
            if not match:
                raise ValueError("No valid list found in response")
            
            # Parse and validate each item
            items = eval(match.group(0))
            return [model.parse_obj(item) for item in items]
        except Exception as e:
            print(f"Error parsing structured output: {e}")
            return []

# Example Usage
syllabus_text = """
1. Time Value of Money
   - Present Value
   - Future Value
2. Net Present Value
   - Discount Rate
   - Cash Flow Analysis
3. Risk Analysis
   - Probability Distributions
   - Monte Carlo Simulation
"""

parser = SyllabusParser()
graph_data = parser.parse_syllabus(syllabus_text)
print(json.dumps(graph_data, indent=2))