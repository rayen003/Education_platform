"""
Meta-agent for economic animation generation and verification.

This module provides an agent that uses LLM to analyze questions,
generate appropriate visualizations, and verify their accuracy.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple

try:
    import openai
except ImportError:
    print("OpenAI package not installed. Please install with: pip install openai")

# Import our Pydantic models for configuration
from config_models import AnimationConfig, SupplyDemandConfig, SupplyShiftConfig, DemandShiftConfig, TimeValueConfig, PerpetuityConfig

class MetaAgent:
    """
    Meta-agent for economic animation generation using OpenAI's GPT models.
    
    This agent handles:
    1. Analyzing economic questions
    2. Generating appropriate visualization configurations
    3. Verifying animation accuracy
    4. Creating synchronized narration
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the meta-agent.
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY environment variable)
            model: OpenAI model to use (default: gpt-4)
        """
        # Set up OpenAI client
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            
        if not openai.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment")
            
        self.model = model
        self.client = openai.OpenAI()
        
    def analyze_question(self, question: str, explanation: str) -> AnimationConfig:
        """
        Analyze an economic question and explanation to determine the appropriate visualization.
        
        Args:
            question: The economic question
            explanation: Detailed text explanation/answer
            
        Returns:
            AnimationConfig: Configuration for the appropriate visualization
        """
        # Create a prompt for the LLM
        prompt = f"""
        You are an expert economics and finance educator. Analyze this question and explanation:
        
        QUESTION:
        {question}
        
        EXPLANATION:
        {explanation}
        
        Your task is to determine the most appropriate economic visualization to help explain this concept.
        
        1. First, identify the core economic concept (supply/demand, time value of money, etc.)
        2. Determine the specific type of visualization needed
        3. Extract all necessary parameters for that visualization
        4. Format your response as a structured JSON object
        
        Valid visualization types:
        - "supply_demand" (basic equilibrium)
        - "supply_shift" (shifting supply curve)
        - "demand_shift" (shifting demand curve)
        - "time_value" (time value of money)
        - "perpetuity" (perpetuity calculations)
        
        For supply/demand, include:
        - supply_slope (MUST be positive)
        - supply_intercept
        - demand_slope (MUST be negative)
        - demand_intercept
        
        For supply_shift, also include:
        - new_supply_intercept
        - new_supply_slope (optional)
        
        For demand_shift, also include:
        - new_demand_intercept
        - new_demand_slope (optional)
        
        For time_value, include:
        - cash_flows: list of cash flows
        - time_periods: list of time periods
        - interest_rate (as decimal, e.g., 0.05 for 5%)
        
        For perpetuity, include:
        - payment: regular payment amount
        - interest_rate (as decimal)
        - periods_to_show: number of periods to display
        
        Ensure the visualization parameters accurately represent the economic situation described.
        
        IMPORTANT: Carefully check that:
        1. Supply slopes are POSITIVE
        2. Demand slopes are NEGATIVE
        3. Interest rates are in DECIMAL form (e.g., 0.05 for 5%)
        4. All necessary parameters are provided
        
        Return a complete JSON object conforming to these specifications.
        """
        
        # Call the OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            # Extract and parse the JSON response
            response_text = response.choices[0].message.content
            config_dict = json.loads(response_text)
            
            # Validate the configuration using Pydantic
            try:
                config = AnimationConfig.parse_obj(config_dict)
                print(f"Generated visualization configuration for type: {config.visualization_type}")
                return config
            except Exception as e:
                raise ValueError(f"Invalid configuration generated: {str(e)}\n\nRaw config: {config_dict}")
                
        except Exception as e:
            raise RuntimeError(f"Error during question analysis: {str(e)}")
    
    def verify_animation(self, config: AnimationConfig, rendering_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify the accuracy and educational quality of the generated animation.
        
        Args:
            config: Animation configuration
            rendering_info: Information about the rendered animation
            
        Returns:
            Dict containing verification results
        """
        # Create a prompt for verification
        prompt = f"""
        You are an expert economics educator reviewing an economic visualization. 
        
        ANIMATION CONFIGURATION:
        ```
        {config.json(indent=2)}
        ```
        
        RENDERING INFORMATION:
        ```
        {json.dumps(rendering_info, indent=2)}
        ```
        
        Verify this animation for:
        1. Mathematical accuracy - Are the calculations correct?
        2. Educational clarity - Will it help students understand the concept?
        3. Visual effectiveness - Is the visualization clear and well-designed?
        4. Parameter appropriateness - Are the parameters suitable for teaching this concept?
        
        For each category, provide:
        - A score from 0.0 to 1.0
        - Specific feedback
        - Improvement suggestions (if score < 0.9)
        
        Also perform specific verification based on the visualization type:
        
        For supply/demand visualizations:
        - Verify equilibrium calculation: P = {config.supply_demand_config.supply_slope}Q + {config.supply_demand_config.supply_intercept} = {config.supply_demand_config.demand_slope}Q + {config.supply_demand_config.demand_intercept}
        - Check that the equilibrium point is within the visible range
        - Confirm supply slope is positive and demand slope is negative
        
        For time value/perpetuity:
        - Verify present value calculation for perpetuity: PV = payment / interest_rate
        - Confirm interest rate is appropriate (not too small or large)
        
        Return your response as a JSON object with these keys:
        - overall_score (0.0 to 1.0)
        - category_scores (object with scores for each category)
        - specific_verifications (list of verification checks and results)
        - issues (list of any identified issues)
        - improvement_suggestions (list of suggested improvements)
        - is_approved (boolean, true if overall_score >= 0.8)
        """
        
        # Call the OpenAI API for verification
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            # Extract and parse the verification result
            verification_text = response.choices[0].message.content
            verification_result = json.loads(verification_text)
            
            print(f"Verification complete. Overall score: {verification_result.get('overall_score', 'N/A')}")
            print(f"Approved: {verification_result.get('is_approved', False)}")
            
            return verification_result
            
        except Exception as e:
            print(f"Error during animation verification: {str(e)}")
            # Return a default verification result in case of error
            return {
                "overall_score": 0.5,
                "category_scores": {"accuracy": 0.5, "clarity": 0.5, "effectiveness": 0.5, "parameters": 0.5},
                "issues": [f"Verification error: {str(e)}"],
                "is_approved": False
            }
    
    def generate_narration(self, config: AnimationConfig, duration: float = 30.0) -> Dict[str, Any]:
        """
        Generate synchronized narration for the animation.
        
        Args:
            config: Animation configuration
            duration: Estimated animation duration in seconds
            
        Returns:
            Dict containing narration script and timing information
        """
        # Create a prompt for narration generation
        animation_type = config.visualization_type
        
        # Add type-specific details
        type_details = ""
        if animation_type in ["supply_demand", "supply_shift", "demand_shift"]:
            eq = config.supply_demand_config.calculate_equilibrium()
            type_details = f"""
            Supply equation: P = {config.supply_demand_config.supply_slope}Q + {config.supply_demand_config.supply_intercept}
            Demand equation: P = {config.supply_demand_config.demand_slope}Q + {config.supply_demand_config.demand_intercept}
            Equilibrium: Q = {eq['quantity']:.2f}, P = {eq['price']:.2f}
            """
            
            if animation_type == "supply_shift":
                new_supply_slope = config.supply_shift_config.new_supply_slope or config.supply_shift_config.supply_slope
                type_details += f"""
                New supply equation: P = {new_supply_slope}Q + {config.supply_shift_config.new_supply_intercept}
                """
                
            elif animation_type == "demand_shift":
                new_demand_slope = config.demand_shift_config.new_demand_slope or config.demand_shift_config.demand_slope
                type_details += f"""
                New demand equation: P = {new_demand_slope}Q + {config.demand_shift_config.new_demand_intercept}
                """
                
        elif animation_type == "perpetuity":
            pv = config.perpetuity_config.calculate_present_value()
            type_details = f"""
            Payment amount: ${config.perpetuity_config.payment}
            Interest rate: {config.perpetuity_config.interest_rate * 100:.1f}%
            Present value: ${pv:.2f}
            """
        
        prompt = f"""
        You are an expert economics educator creating narration for an educational animation.
        
        ANIMATION TYPE: {animation_type}
        TITLE: {config.title}
        DURATION: Approximately {duration} seconds
        
        TECHNICAL DETAILS:
        {type_details}
        
        Your task is to create:
        1. A complete narration script that explains this economic concept clearly
        2. Timing markers for synchronizing the narration with the animation
        
        The animation has these key segments:
        - 0-5 seconds: Introduction and setup
        - 5-15 seconds: Initial explanation of the concept
        - 15-25 seconds: Demonstration of the key relationships
        - 25-30 seconds: Conclusion and takeaways
        
        For supply_shift or demand_shift animations:
        - The shift occurs around 15 seconds into the animation
        
        Your narration should:
        - Be educational and clear, appropriate for an economics student
        - Explain the concept in simple terms first, then add technical details
        - Highlight the key relationships and insights
        - Match the pacing of the animation (introduce elements as they appear)
        - Have a total duration close to the animation duration
        
        Return your response as a JSON object with:
        - full_script: Complete narration text
        - segments: Array of objects with {{time: seconds, text: segment text}}
        - estimated_duration: Estimated narration duration in seconds
        """
        
        # Call the OpenAI API for narration generation
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            # Extract and parse the narration
            narration_text = response.choices[0].message.content
            narration_data = json.loads(narration_text)
            
            print(f"Generated narration with {len(narration_data.get('segments', []))} segments")
            print(f"Estimated duration: {narration_data.get('estimated_duration', 'N/A')} seconds")
            
            return narration_data
            
        except Exception as e:
            print(f"Error during narration generation: {str(e)}")
            # Return a default narration in case of error
            return {
                "full_script": f"This is a visualization of {animation_type}. The key insight is that economic variables are related through supply and demand.",
                "segments": [
                    {"time": 0, "text": f"This is a visualization of {animation_type}."},
                    {"time": 10, "text": "The key insight is that economic variables are related through supply and demand."}
                ],
                "estimated_duration": 15.0
            }
    
    def full_pipeline(self, question: str, explanation: str, output_format: str = "html") -> Dict[str, Any]:
        """
        Execute the full animation pipeline from question to final output.
        
        Args:
            question: Economic question
            explanation: Detailed explanation/answer
            output_format: Desired output format (html, video, image)
            
        Returns:
            Dict with all outputs from the pipeline
        """
        results = {
            "question": question,
            "explanation": explanation,
            "timestamps": {
                "start": time.time()
            }
        }
        
        # Step 1: Analyze question and generate configuration
        print("Step 1: Analyzing question and generating configuration...")
        config = self.analyze_question(question, explanation)
        results["configuration"] = config.dict()
        results["timestamps"]["configuration_generated"] = time.time()
        
        # Step 2: We would call the visualizer here
        # For now, we'll just return the configuration
        print("Step 2: Visualization would be generated here...")
        results["visualization"] = {
            "status": "pending_implementation",
            "expected_files": [
                f"output/{config.visualization_type}_animation.html",
                f"output/{config.visualization_type}_animation.mp3"
            ]
        }
        results["timestamps"]["visualization_placeholder"] = time.time()
        
        # Step 3: Verification would be done here
        print("Step 3: Verification would be performed here...")
        # We'll skip this for now, but include a placeholder
        results["verification"] = {
            "status": "pending_implementation",
            "expected_score": 0.9
        }
        results["timestamps"]["verification_placeholder"] = time.time()
        
        # Step 4: Generate narration
        print("Step 4: Generating narration...")
        narration = self.generate_narration(config)
        results["narration"] = narration
        results["timestamps"]["narration_generated"] = time.time()
        
        # Step 5: Final output would be created here
        print("Step 5: Final output would be created here...")
        results["output"] = {
            "status": "pending_implementation",
            "expected_files": [
                f"output/{config.visualization_type}_final.{output_format}",
                f"output/{config.visualization_type}_narration.mp3"
            ]
        }
        results["timestamps"]["end"] = time.time()
        
        # Calculate total time
        total_time = results["timestamps"]["end"] - results["timestamps"]["start"]
        results["total_time"] = total_time
        print(f"Pipeline completed in {total_time:.2f} seconds")
        
        return results

# Example of using the meta-agent
if __name__ == "__main__":
    # Example question and explanation
    question = "If the supply curve is P = 2 + 0.5Q and the demand curve is P = 10 - 0.5Q, what is the equilibrium price and quantity?"
    explanation = "To find the equilibrium, we set the supply and demand equations equal: 2 + 0.5Q = 10 - 0.5Q. Solving for Q, we get 0.5Q + 0.5Q = 10 - 2, so Q = 8. Substituting back, the equilibrium price is P = 2 + 0.5(8) = 6."
    
    # Create and use the meta-agent
    agent = MetaAgent()
    results = agent.full_pipeline(question, explanation)
    
    # Save the results to a file
    with open("meta_agent_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to meta_agent_results.json") 