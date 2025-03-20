#!/usr/bin/env python
"""
AI Economics Animation Generator

This script demonstrates the end-to-end process of taking an economic question,
analyzing it with AI, and generating a custom animation with explanation.
"""

import os
import sys
import json
import time
import logging
from config_models import AnimationConfig, SupplyShiftConfig
from enhanced_animation import EnhancedEconomicsAnimation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_econ_animation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AI_Econ_Animation")

class MockEconAgent:
    """
    Mock AI agent that simulates analyzing economic questions and generating
    animation configurations.
    
    In a real implementation, this would use LLM API calls to analyze the questions
    and generate the appropriate configurations.
    """
    
    def __init__(self):
        """Initialize the mock agent"""
        self.call_count = 0
        self.call_durations = []
    
    def analyze_question(self, question, explanation=None):
        """
        Analyze an economic question and determine what type of visualization is needed.
        
        Args:
            question: The economic question to analyze
            explanation: Optional detailed explanation/solution
            
        Returns:
            dict: Configuration for the animation
        """
        start_time = time.time()
        logger.info(f"AI Agent analyzing question: {question}")
        
        # In a real implementation, this would make an API call to an LLM
        # For this mock version, we'll simulate the analysis based on keywords
        
        # Log the call
        self.call_count += 1
        
        # Simulate thinking time
        time.sleep(1)
        
        # Mock analysis of the question - in reality this would use an LLM
        question_lower = question.lower()
        config = None
        
        # Check for supply shift keywords
        supply_keywords = ["supply shift", "supply shock", "supply decrease", "supply increase", 
                          "production cost", "technology improvement"]
        
        demand_keywords = ["demand shift", "demand shock", "demand decrease", "demand increase",
                          "consumer preference", "income change"]
        
        if any(keyword in question_lower for keyword in supply_keywords):
            # Create supply shift configuration
            config = self._create_supply_shift_config(question, explanation)
            analysis_type = "supply_shift"
        elif any(keyword in question_lower for keyword in demand_keywords):
            # Create demand shift configuration
            config = self._create_demand_shift_config(question, explanation)
            analysis_type = "demand_shift"
        else:
            # Default to basic supply-demand equilibrium
            config = self._create_equilibrium_config(question, explanation)
            analysis_type = "equilibrium"
        
        # Calculate duration
        duration = time.time() - start_time
        self.call_durations.append(duration)
        
        logger.info(f"AI analysis complete. Type: {analysis_type}, Duration: {duration:.2f}s")
        return config
    
    def _create_supply_shift_config(self, question, explanation=None):
        """Create a supply shift configuration based on the question"""
        # Extract parameters from the question or use defaults
        # In a real implementation, the LLM would parse the question for these values
        
        # Check if the shift is an increase or decrease in supply
        if "increase" in question.lower() or "improvement" in question.lower():
            # Supply increase (shift right/down) - lower intercept
            new_intercept = 1  # Lower than the default of 2
            title = "Impact of Supply Increase on Market Equilibrium"
            subtitle = "Simulation of a rightward shift in the supply curve"
        else:
            # Supply decrease (shift left/up) - higher intercept
            new_intercept = 4  # Higher than the default of 2
            title = "Impact of Supply Decrease on Market Equilibrium"
            subtitle = "Simulation of a leftward shift in the supply curve"
        
        # Create the configuration
        config = {
            "supply_slope": 0.5,
            "supply_intercept": 2,
            "demand_slope": -0.5,
            "demand_intercept": 8, 
            "new_supply_intercept": new_intercept,
            "title": title,
            "subtitle": subtitle
        }
        
        return config
    
    def _create_demand_shift_config(self, question, explanation=None):
        """Create a demand shift configuration based on the question"""
        # Extract parameters from the question or use defaults
        # In a real implementation, the LLM would parse the question for these values
        
        # Check if the shift is an increase or decrease in demand
        if "increase" in question.lower() or "preference" in question.lower():
            # Demand increase (shift right) - higher intercept
            new_intercept = 10  # Higher than the default of 8
            title = "Impact of Demand Increase on Market Equilibrium"
            subtitle = "Simulation of a rightward shift in the demand curve"
        else:
            # Demand decrease (shift left) - lower intercept
            new_intercept = 6  # Lower than the default of 8
            title = "Impact of Demand Decrease on Market Equilibrium"
            subtitle = "Simulation of a leftward shift in the demand curve"
        
        # Create the configuration
        config = {
            "supply_slope": 0.5,
            "supply_intercept": 2,
            "demand_slope": -0.5,
            "demand_intercept": 8,
            "new_demand_intercept": new_intercept,
            "title": title,
            "subtitle": subtitle,
            "is_demand_shift": True  # Flag to indicate this is a demand shift
        }
        
        return config
    
    def _create_equilibrium_config(self, question, explanation=None):
        """Create a basic equilibrium configuration"""
        # Default configuration
        config = {
            "supply_slope": 0.5,
            "supply_intercept": 2,
            "demand_slope": -0.5,
            "demand_intercept": 8,
            "title": "Supply and Demand Equilibrium Analysis",
            "subtitle": "Finding the market clearing price and quantity"
        }
        
        return config
    
    def get_statistics(self):
        """Get statistics on the AI agent's performance"""
        stats = {
            "total_calls": self.call_count,
            "total_time": sum(self.call_durations),
            "average_time": sum(self.call_durations) / max(1, len(self.call_durations)),
            "min_time": min(self.call_durations) if self.call_durations else 0,
            "max_time": max(self.call_durations) if self.call_durations else 0
        }
        return stats

def run_end_to_end(question, explanation=None, output_dir="./output/ai_generated"):
    """
    Run the end-to-end process from question to animation.
    
    Args:
        question: The economic question to analyze
        explanation: Optional detailed explanation/solution
        output_dir: Directory to save the generated files
        
    Returns:
        dict: Output files and statistics
    """
    logger.info(f"Starting end-to-end process for question: {question}")
    
    # Step 1: Create the AI agent
    agent = MockEconAgent()
    
    # Step 2: Analyze the question to determine animation type and config
    start_time = time.time()
    config = agent.analyze_question(question, explanation)
    analysis_time = time.time() - start_time
    
    # Step 3: Create the animation generator
    animator = EnhancedEconomicsAnimation(output_dir=output_dir)
    
    # Step 4: Generate the animation based on the configuration
    start_time = time.time()
    
    if "is_demand_shift" in config and config["is_demand_shift"]:
        # Special handling for demand shift
        # Now we have a proper demand shift method
        logger.info("Generating demand shift animation")
        
        # Generate using the demand shift method
        result = animator.create_demand_shift_animation(
            supply_slope=config["supply_slope"],
            supply_intercept=config["supply_intercept"],
            demand_slope=config["demand_slope"],
            demand_intercept=config["demand_intercept"],
            new_demand_intercept=config["new_demand_intercept"],
            title=config["title"],
            subtitle=config["subtitle"]
        )
    else:
        # Standard supply shift
        logger.info("Generating supply shift animation")
        result = animator.create_supply_demand_animation(**config)
    
    generation_time = time.time() - start_time
    
    # Step 5: Get agent statistics
    stats = agent.get_statistics()
    stats.update({
        "analysis_time": analysis_time,
        "generation_time": generation_time,
        "total_process_time": analysis_time + generation_time
    })
    
    # Step 6: Generate a report
    report_path = os.path.join(output_dir, "process_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    report = {
        "question": question,
        "explanation": explanation,
        "configuration": config,
        "statistics": stats,
        "output_files": result
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"End-to-end process complete. Results saved to {output_dir}")
    logger.info(f"Analysis time: {analysis_time:.2f}s, Generation time: {generation_time:.2f}s")
    
    return {
        "output_files": result,
        "statistics": stats,
        "report": report_path
    }

def main():
    """Run the AI economics animation demo"""
    # Example questions
    questions = [
        {
            "question": "How does an increase in production costs affect the market for smartphones?",
            "explanation": """
When production costs increase for smartphones, the supply curve shifts leftward (or upward).
This is because producers now require a higher price to provide the same quantity of smartphones.

Initial market:
- Supply: P = 2 + 0.5Q
- Demand: P = 8 - 0.5Q

After cost increase:
- Supply: P = 4 + 0.5Q (higher intercept)
- Demand: P = 8 - 0.5Q (unchanged)

This results in:
1. Higher equilibrium price
2. Lower equilibrium quantity
3. Decreased consumer surplus
4. Potential decrease in producer surplus (depends on elasticity)
5. Overall decrease in market efficiency
"""
        },
        {
            "question": "What happens to the market for coffee when consumer preferences shift toward tea?",
            "explanation": """
When consumer preferences shift away from coffee toward tea, the demand for coffee decreases.
This causes a leftward shift in the demand curve for coffee.

Initial market:
- Supply: P = 2 + 0.5Q
- Demand: P = 8 - 0.5Q

After preference change:
- Supply: P = 2 + 0.5Q (unchanged)
- Demand: P = 6 - 0.5Q (lower intercept)

This results in:
1. Lower equilibrium price for coffee
2. Lower equilibrium quantity of coffee sold
3. Decreased producer surplus in the coffee market
4. Increased consumer surplus for remaining coffee consumers
5. Potential growth in the tea market
"""
        }
    ]
    
    # Choose which question to process
    question_index = 1  # Change to 1 for the second question
    
    # Process the selected question
    result = run_end_to_end(
        questions[question_index]["question"],
        questions[question_index]["explanation"],
        output_dir=f"./output/ai_demo_question_{question_index+1}"
    )
    
    # Open the interactive visualization
    if "output_files" in result and "interactive_html" in result["output_files"]:
        interactive_html = result["output_files"]["interactive_html"]
        if os.path.exists(interactive_html):
            print(f"\nOpening interactive visualization: {interactive_html}")
            
            # Open the file in the default browser
            if sys.platform == 'darwin':  # macOS
                import subprocess
                subprocess.run(['open', interactive_html])
            elif sys.platform == 'win32':  # Windows
                os.startfile(interactive_html)
            else:  # Linux or other Unix
                import subprocess
                subprocess.run(['xdg-open', interactive_html])
    
    print("\nProcess statistics:")
    for key, value in result["statistics"].items():
        if "time" in key:
            print(f"- {key}: {value:.2f} seconds")
        else:
            print(f"- {key}: {value}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 