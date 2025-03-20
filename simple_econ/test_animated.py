#!/usr/bin/env python
"""
Simple test script for animated supply shift visualization.
"""

import os
import sys
import json
from animated_example import AnimatedSupplyDemandPlot

def main():
    # Basic configuration
    config = {
        "supply_config": {
            "slope": 0.8,
            "intercept": 1,
            "color": "rgb(0, 0, 255)"  # Blue
        },
        "demand_config": {
            "slope": -0.7,
            "intercept": 8,
            "color": "rgb(255, 0, 0)"  # Red
        },
        "layout_config": {
            "title": "Animated Supply Shift",
            "width": 950,
            "height": 600
        }
    }
    
    # Create the animated plot
    print("Creating AnimatedSupplyDemandPlot...")
    sd_plot = AnimatedSupplyDemandPlot(config)
    
    # Define new configuration for the shift
    new_config = {
        "intercept": 3,  # Higher intercept (decreased supply)
        "color": "rgb(0, 0, 150)"  # Darker blue
    }
    
    # Create output directory
    output_dir = "./output/test_animated"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the animation
    print("Generating animated supply shift...")
    html_path = sd_plot.create_animated_shift(
        shift_type="supply",
        new_config=new_config,
        frames=30,
        output_dir=output_dir
    )
    
    print(f"Animation created: {html_path}")
    
    # Open the file
    if html_path and os.path.exists(html_path):
        print("Opening the animation...")
        if sys.platform == 'darwin':  # macOS
            import subprocess
            subprocess.run(['open', html_path])
        elif sys.platform == 'win32':  # Windows
            os.startfile(html_path)
        else:  # Linux or other Unix
            import subprocess
            subprocess.run(['xdg-open', html_path])
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 