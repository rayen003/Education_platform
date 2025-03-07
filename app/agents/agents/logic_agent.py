import os
import json
from typing import Dict, Any, TypedDict, List, Optional, Union, Literal
from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# Define typed dictionary for state
class LogicState(TypedDict, total=False):
    question: str
    student_answer: str
    analysis: Dict[str, Any]
    feedback: str
    complexity_score: float
    needs_detailed_analysis: bool
    hint_count: int
    hints: List[str]
    needs_hint: bool
    proximity_score: Optional[float]

class LogicAgent:
    """
    Agent responsible for analyzing logical reasoning in student answers using LangGraph.
    """
    def __init__(self):
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Build the feedback graph
        self.feedback_graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph for analyzing logical reasoning.
        """
        # Create a new graph
        builder = StateGraph(LogicState)
        
        # Add nodes to the graph
        builder.add_node("assess_complexity", self.assess_complexity)
        builder.add_node("analyze_logic", self.analyze_logic)
        builder.add_node("assess_proximity", self.assess_proximity)
        builder.add_node("generate_hint", self.generate_hint)
        builder.add_node("generate_feedback", self.generate_feedback)
        
        # Define the edges in the graph
        builder.add_edge("assess_complexity", "analyze_logic")
        
        # Conditional edge: If complexity is low, skip detailed analysis
        builder.add_conditional_edges(
            "analyze_logic",
            lambda state: "assess_proximity" if state["analysis"]["is_correct"] is False else "generate_feedback"
        )
        
        # Conditional edge: If proximity score is below threshold, generate hint
        builder.add_conditional_edges(
            "assess_proximity",
            lambda state: "generate_hint" if state["needs_hint"] else "generate_feedback"
        )
        
        # Connect hint generation to feedback
        builder.add_edge("generate_hint", "generate_feedback")
        
        # Connect feedback to the end
        builder.add_edge("generate_feedback", END)
        
        # Set the entry point
        builder.set_entry_point("assess_complexity")
        
        return builder
    
    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the logical reasoning in a student's answer.
        
        Args:
            state: The current state containing question and student_answer
            
        Returns:
            Updated state with feedback
        """
        # Compile the graph into an executable workflow
        workflow = self.feedback_graph.compile()
        
        # Initialize state with default values if not present
        if "analysis" not in state:
            state["analysis"] = {}
        if "hint_count" not in state:
            state["hint_count"] = 0
        if "hints" not in state:
            state["hints"] = []
        if "needs_hint" not in state:
            state["needs_hint"] = False
        
        # Run the compiled workflow
        return workflow.invoke(state)
    
    def assess_complexity(self, state: LogicState) -> LogicState:
        """
        Assess the complexity of the question to determine if detailed logic analysis is needed.
        """
        question = state["question"]
        student_answer = state["student_answer"]
        
        # Use OpenAI to assess complexity on a scale of 0-10
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a logic assessment assistant. Evaluate the complexity of the given question 
                    on a scale of 0-10, where 0 is extremely simple and 10 is highly complex.
                    
                    Consider the following factors:
                    1. Logical reasoning required
                    2. Number of concepts involved
                    3. Depth of analysis needed
                    4. Potential for logical fallacies
                    
                    Return only a number between 0 and 10.
                    """
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nStudent Answer: {student_answer}"
                }
            ]
        )
        
        try:
            complexity_score = float(response.choices[0].message.content.strip())
        except ValueError:
            # Default to medium complexity if parsing fails
            complexity_score = 5.0
            
        # Determine if detailed analysis is needed (threshold: 3.0)
        needs_detailed_analysis = complexity_score >= 3.0
        
        # Update state
        state["complexity_score"] = complexity_score
        state["needs_detailed_analysis"] = needs_detailed_analysis
        print(f"Complexity Score: {complexity_score}/10")
        print(f"Needs Detailed Analysis: {needs_detailed_analysis}")
        
        return state
    
    def analyze_logic(self, state: LogicState) -> LogicState:
        """
        Analyze the logical reasoning in the student's answer.
        """
        question = state["question"]
        student_answer = state["student_answer"]
        
        # Debug print statements
        print(f"Analyzing Question: {question}")
        print(f"Student Answer: {student_answer}")
        
        # Use OpenAI to analyze logical reasoning
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a logic evaluation agent specializing in analyzing the logical structure, coherence, 
                    and argumentation of student answers.

                    INSTRUCTIONS:
                    - Analyze the student's answer for logical flow, clear argumentation, and conceptual understanding.
                    - Identify any logical fallacies, unclear reasoning, or non-sequiturs.
                    - Evaluate if conclusions follow from premises and if reasoning chains are complete.
                    
                    Return a JSON object with the following structure:
                    {
                        "is_correct": boolean,
                        "logical_structure": string (description of the logical structure),
                        "fallacies": [list of identified fallacies],
                        "strengths": [list of logical strengths],
                        "weaknesses": [list of logical weaknesses]
                    }
                    """
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nStudent Answer: {student_answer}"
                }
            ]
        )
        
        try:
            # Parse the JSON response
            analysis = json.loads(response.choices[0].message.content.strip())
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            analysis = {
                "is_correct": False,
                "logical_structure": "Unable to parse logical structure",
                "fallacies": ["Unable to identify fallacies"],
                "strengths": [],
                "weaknesses": ["Unable to identify weaknesses"]
            }
        
        # Update state
        state["analysis"] = analysis
        print(f"Analysis: {analysis}")
        
        return state
    
    def assess_proximity(self, state: LogicState) -> LogicState:
        """
        Assess how close the student's answer is to a logically sound answer.
        """
        question = state["question"]
        student_answer = state["student_answer"]
        is_correct = state["analysis"]["is_correct"]
        
        # If the answer is correct, no need for proximity assessment
        if is_correct:
            state["proximity_score"] = 10
            state["needs_hint"] = False
            return state
        
        # Use OpenAI to assess proximity on a scale of 0-10
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a logic assessment assistant. Evaluate how close the student's answer is to a logically sound answer 
                    on a scale of 0-10, where 0 is completely illogical and 10 is perfectly logical.
                    
                    Consider the following factors:
                    1. Logical structure and flow
                    2. Presence of fallacies
                    3. Clarity of reasoning
                    4. Connection between premises and conclusions
                    
                    Return only a number between 0 and 10.
                    """
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nStudent Answer: {student_answer}"
                }
            ]
        )
        
        try:
            proximity_score = float(response.choices[0].message.content.strip())
        except ValueError:
            # Default to medium proximity if parsing fails
            proximity_score = 5.0
            
        # Determine if a hint is needed (threshold: 7.0)
        needs_hint = proximity_score < 7.0
        
        # Update state
        state["proximity_score"] = proximity_score
        state["needs_hint"] = needs_hint
        print(f"Proximity Score: {proximity_score}/10")
        print(f"Needs Hint: {needs_hint}")
        
        return state
    
    def generate_hint(self, state: LogicState) -> LogicState:
        """
        Generate a hint for the student based on their answer.
        """
        question = state["question"]
        student_answer = state["student_answer"]
        hint_count = state["hint_count"]
        proximity_score = state["proximity_score"]
        
        # Generate a hint using OpenAI
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a supportive logic tutor providing hints to students. 
                    Based on the student's answer, provide a helpful hint that guides them 
                    toward a more logically sound answer without giving it away completely.
                    
                    For the first hint (hint_count=0), provide a general conceptual hint about logical structure.
                    For the second hint (hint_count=1), provide a more specific hint about potential fallacies or gaps in reasoning.
                    For the third hint (hint_count=2), provide a very specific hint that almost gives away the logical structure needed.
                    
                    Keep the hint concise and clear.
                    """
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nStudent Answer: {student_answer}\nHint Count: {hint_count}\nProximity Score: {proximity_score}/10"
                }
            ]
        )
        
        # Extract the hint
        hint = response.choices[0].message.content.strip()
        
        # Update state
        state["hints"].append(hint)
        state["hint_count"] += 1
        print(f"Generated Hint #{hint_count + 1}: {hint}")
        
        return state
    
    def generate_feedback(self, state: LogicState) -> LogicState:
        """
        Generate feedback for the student based on the analysis.
        """
        question = state["question"]
        student_answer = state["student_answer"]
        analysis = state["analysis"]
        is_correct = analysis.get("is_correct", False)
        hints = state.get("hints", [])
        proximity_score = state.get("proximity_score")
        
        # Generate detailed feedback using OpenAI
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a supportive logic tutor providing feedback on student answers.
                    
                    Follow these guidelines:
                    1. Start with positive reinforcement, acknowledging what the student did correctly.
                    2. If the answer has logical flaws, explain them clearly and constructively.
                    3. Highlight strengths in the student's logical reasoning.
                    4. Offer specific suggestions for improving logical structure and avoiding fallacies.
                    5. End with encouragement for future reasoning tasks.
                    
                    If hints were provided to the student, reference them in your feedback and build upon them.
                    If a proximity score was given, tailor your feedback to acknowledge how close they were to a logically sound answer.
                    
                    Keep your feedback concise, clear, and educational.
                    """
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nStudent Answer: {student_answer}\nAnalysis: {json.dumps(analysis)}\nIs Correct: {is_correct}\nHints Provided: {hints if hints else 'None'}\nProximity Score: {proximity_score if proximity_score is not None else 'Not available'}/10"
                }
            ]
        )
        
        # Extract the feedback
        feedback = response.choices[0].message.content.strip()
        
        # Update state
        state["feedback"] = feedback
        print(f"Generated Feedback: {feedback}")
        
        return state


# Test cases for the LogicAgent
if __name__ == "__main__":
    # Instantiate the LogicAgent
    logic_agent = LogicAgent()

    # Define test cases
    test_cases = [
        {
            "question": "Explain why supply and demand curves intersect at the equilibrium price.",
            "student_answer": "The equilibrium price is where the quantity supplied equals the quantity demanded. This is because at this price, the market is balanced, and there is no surplus or shortage."
        },
        {
            "question": "What is the logical fallacy in the following statement: 'If we allow students to use calculators, they will stop learning basic math.'",
            "student_answer": "The statement commits a slippery slope fallacy by assuming that one action will inevitably lead to a negative outcome without providing evidence."
        },
        {
            "question": "Is the following argument valid? All cats are mammals. Some mammals can swim. Therefore, some cats can swim.",
            "student_answer": "Yes, the argument is valid because cats are mammals and some mammals can swim, so some cats can swim."
        },
        {
            "question": "What is the primary difference between correlation and causation?",
            "student_answer": "Correlation means two things happen together, while causation means one thing causes the other."
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
            result = logic_agent.analyze(state)
            
            # Display results
            print(f"Logic Agent Feedback: {result['feedback']}")
            
            # Display hints if any were generated
            if result.get('hints') and len(result['hints']) > 0:
                print("\nHints provided:")
                for idx, hint in enumerate(result['hints']):
                    print(f"Hint #{idx+1}: {hint}")
                    
            # Display proximity score if available
            if 'proximity_score' in result and result['proximity_score'] is not None:
                print(f"Proximity Score: {result['proximity_score']}/10")
                
            # Display complexity score
            if 'complexity_score' in result:
                print(f"Complexity Score: {result['complexity_score']}/10")
                
        except Exception as e:
            print(f"Error processing test case {i+1}: {e}")





