#!/usr/bin/env python
"""
Test Animation Script for PlotlyVisualizer

This script demonstrates the animated, interactive features of the PlotlyVisualizer
with synchronized audio narration.
"""

import os
import sys
import json
from config_models import AnimationConfig, SupplyShiftConfig
from plotly_visualizer import PlotlyVisualizer

def main():
    """Main function to test the animated visualizations with audio synchronization"""
    
    print("Creating PlotlyVisualizer instance...")
    visualizer = PlotlyVisualizer(output_dir="./output/test_animation")
    
    # Create a supply shift animation config
    print("Creating supply shift animation configuration...")
    config_dict = {
        "visualization_type": "supply_shift",
        "title": "Impact of Production Cost Increase on Market Equilibrium",
        "subtitle": "Interactive supply curve shift with synchronized narration",
        "supply_shift_config": {
            "supply_slope": 0.5,
            "supply_intercept": 2,
            "demand_slope": -0.5,
            "demand_intercept": 8,
            "new_supply_intercept": 4,  # Higher intercept (decreased supply)
            "x_range": [0, 10],
            "y_range": [0, 10],
            "show_equilibrium": True,
            "equilibrium_color": "green",
            "supply_color": "blue",
            "demand_color": "red",
            "frames": 30,  # Number of animation frames
            "shift_style": "parallel"
        }
    }
    
    # Parse the configuration using Pydantic
    print("Validating configuration...")
    config = AnimationConfig.model_validate(config_dict)  # Updated to use model_validate instead of parse_obj
    
    # Create the visualization
    print("Generating animated supply shift visualization...")
    result = visualizer.create_visualization(config, output_format="html")
    
    # Print the result
    print("\nVisualization Generation Results:")
    print(json.dumps(result, indent=2))
    
    # Check if the visualization was successful
    if result["status"] == "success" and "output_files" in result:
        if "interactive_html" in result["output_files"]:
            # Open the interactive HTML file
            interactive_html = result["output_files"]["interactive_html"]
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
                
            print("\nInteractive visualization opened in your browser.")
            print("- You can play/pause the animation")
            print("- Audio narration is synchronized with the visual elements")
            print("- The explanation text updates as the animation progresses")
        else:
            print("\nWarning: No interactive HTML file was generated.")
            
            # If there's a regular HTML file, open that instead
            if "animated_html" in result["output_files"]:
                animated_html = result["output_files"]["animated_html"]
                print(f"Opening animated HTML file instead: {animated_html}")
                
                if sys.platform == 'darwin':  # macOS
                    import subprocess
                    subprocess.run(['open', animated_html])
            else:
                print("No animation files found in the output.")
    else:
        print(f"\nError: Visualization generation failed.")
        if "error" in result:
            print(f"Error message: {result['error']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 