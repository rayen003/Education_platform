"""
Simple Supply and Demand Equilibrium Demo.

This script demonstrates how to use the simplified SupplyDemandPlot component
to visualize a market equilibrium example.
"""

import os
import matplotlib.pyplot as plt
from lib.simple_supply_demand import SupplyDemandPlot
from gtts import gTTS
import subprocess
import platform

def generate_audio(text, filename="explanation"):
    """
    Generate audio explanation using gTTS.
    
    Args:
        text: Text to convert to speech
        filename: Base filename for the audio file
        
    Returns:
        Path to the generated audio file
    """
    # Create directory if it doesn't exist
    output_dir = "media/audio"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate audio file
    tts = gTTS(text=text, lang='en', slow=False)
    filepath = f"{output_dir}/{filename}.mp3"
    tts.save(filepath)
    
    return filepath

def play_audio(file_path):
    """
    Play an audio file using the appropriate command for the platform.
    
    Args:
        file_path: Path to the audio file to play
    """
    try:
        if platform.system() == 'Darwin':  # macOS
            subprocess.run(['afplay', file_path])
        elif platform.system() == 'Windows':
            os.startfile(file_path)
        else:  # Linux or other
            subprocess.run(['mpg123', file_path], check=True)
    except Exception as e:
        print(f"Could not play audio: {e}")
        print(f"Audio file is available at: {file_path}")

def main():
    """Main function to run the demo."""
    # Create output directory for images
    output_dir = "media/images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure our supply and demand plot
    config = {
        "supply_config": {
            "slope": 0.5,
            "intercept": 2,
            "color": "#1f77b4",  # Blue
            "label": "Supply",
        },
        "demand_config": {
            "slope": -0.5,
            "intercept": 8,
            "color": "#d62728",  # Red
            "label": "Demand",
        },
        "axis_config": {
            "title": "Market Equilibrium Analysis",
        },
        "equilibrium_config": {
            "color": "#2ca02c",  # Green
        }
    }
    
    # Create the plot
    sd_plot = SupplyDemandPlot(config)
    
    # Set up narration
    intro_text = "In this example, we'll solve a market equilibrium problem using supply and demand curves."
    problem_text = "A market has a supply function P equals 2 plus 0.5 Q and demand function P equals 8 minus 0.5 Q. Let's find the equilibrium price and quantity."
    solution_text = "To solve for equilibrium, we set supply equal to demand and solve for quantity. Then we substitute this quantity back into either the supply or demand equation to find the price."
    conclusion_text = "At the equilibrium point, supply equals demand, so the quantity is 6 and the price is 5. This is the market equilibrium, where the quantity supplied equals the quantity demanded."
    
    # Generate audio files
    intro_audio = generate_audio(intro_text, "eq_intro")
    problem_audio = generate_audio(problem_text, "eq_problem")
    solution_audio = generate_audio(solution_text, "eq_solution")
    conclusion_audio = generate_audio(conclusion_text, "eq_conclusion")
    
    # Play introduction
    print("Playing introduction...")
    play_audio(intro_audio)
    
    # Play problem description
    print("\nPlaying problem description...")
    play_audio(problem_audio)
    
    # Create and display the plot
    print("\nCreating supply and demand plot...")
    sd_plot.create_supply_demand_plot(
        save_path=f"{output_dir}/equilibrium_plot.png",
        show=True
    )
    
    # Play solution explanation
    print("\nPlaying solution explanation...")
    play_audio(solution_audio)
    
    # Play conclusion
    print("\nPlaying conclusion...")
    play_audio(conclusion_audio)
    
    # Show where to find the saved image
    print(f"\nPlot saved to: {output_dir}/equilibrium_plot.png")

if __name__ == "__main__":
    main() 