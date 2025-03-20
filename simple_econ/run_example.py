#!/usr/bin/env python
"""
Simple Supply and Demand Example

This script demonstrates how to use the SupplyDemandPlot class to create
interactive visualizations of supply and demand curves with equilibrium analysis.
"""

import os
import sys
import argparse
from supply_demand import SupplyDemandPlot

def main():
    """Main function to run the supply and demand examples"""
    parser = argparse.ArgumentParser(description='Generate supply and demand visualizations')
    
    parser.add_argument('--scene', 
                        type=str, 
                        default='equilibrium',
                        choices=['equilibrium', 'supply_shift', 'demand_shift'],
                        help='Which scene to generate')
    
    parser.add_argument('--output_dir', 
                        type=str, 
                        default='./output',
                        help='Directory to save the output files')
    
    parser.add_argument('--open', 
                        action='store_true',
                        help='Open the HTML visualization after generating')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select the appropriate scene
    if args.scene == 'equilibrium':
        # Basic equilibrium scene
        print("Generating basic equilibrium scene...")
        generate_equilibrium(args.output_dir, args.open)
    
    elif args.scene == 'supply_shift':
        # Supply shift scene
        print("Generating supply shift scene...")
        generate_supply_shift(args.output_dir, args.open)
    
    elif args.scene == 'demand_shift':
        # Demand shift scene
        print("Generating demand shift scene...")
        generate_demand_shift(args.output_dir, args.open)
    
    else:
        print(f"Error: Unknown scene '{args.scene}'")
        return 1
    
    return 0

def generate_equilibrium(output_dir, open_html=False):
    """Generate a basic equilibrium visualization"""
    # Create a supply and demand plot with default parameters
    config = {
        "supply_config": {
            "slope": 0.5,
            "intercept": 2,
            "color": "blue"
        },
        "demand_config": {
            "slope": -0.5,
            "intercept": 8,
            "color": "red"
        },
        "layout_config": {
            "title": "Market Equilibrium Analysis",
            "width": 950,
            "height": 600
        }
    }
    
    sd_plot = SupplyDemandPlot(config)
    
    # Calculate equilibrium with step-by-step explanation
    eq_x, eq_y = sd_plot.solve_equilibrium()
    print(f"Equilibrium point: Q = {eq_x:.2f}, P = {eq_y:.2f}")
    
    # Create complete explanation (HTML, image, text, audio)
    output = sd_plot.create_explanation(os.path.join(output_dir, 'equilibrium'))
    
    print(f"Output files generated in {os.path.join(output_dir, 'equilibrium')}")
    print(f"- HTML: {output['html']}")
    print(f"- Image: {output['image']}")
    print(f"- Text: {output['text']}")
    print(f"- Audio: {output['audio']}")
    
    # Open HTML file if requested
    if open_html:
        open_file(output['html'])

def generate_supply_shift(output_dir, open_html=False):
    """Generate a supply shift visualization"""
    # Create initial supply and demand plot
    config = {
        "supply_config": {
            "slope": 0.8,
            "intercept": 1,
            "color": "blue",
            "name": "Supply (S₁)"
        },
        "demand_config": {
            "slope": -0.7,
            "intercept": 8,
            "color": "red"
        },
        "layout_config": {
            "title": "Supply Shift Analysis",
            "width": 950,
            "height": 600
        }
    }
    
    sd_plot = SupplyDemandPlot(config)
    
    # Calculate initial equilibrium
    initial_eq_x, initial_eq_y = sd_plot.solve_equilibrium()
    print(f"Initial equilibrium: Q = {initial_eq_x:.2f}, P = {initial_eq_y:.2f}")
    
    # Save initial state
    os.makedirs(os.path.join(output_dir, 'supply_shift', 'initial'), exist_ok=True)
    initial_output = sd_plot.create_explanation(os.path.join(output_dir, 'supply_shift', 'initial'))
    
    # Shift the supply curve
    new_supply_config = {
        "intercept": 3,  # Higher intercept (decreased supply)
        "color": "darkblue",
    }
    
    # Apply the shift
    new_eq_x, new_eq_y = sd_plot.shift_supply(new_supply_config, label="Supply (S₂)")
    
    # Recalculate equilibrium with the new curve
    sd_plot.solve_equilibrium()
    
    # Save the new state
    os.makedirs(os.path.join(output_dir, 'supply_shift', 'after'), exist_ok=True)
    new_output = sd_plot.create_explanation(os.path.join(output_dir, 'supply_shift', 'after'))
    
    print(f"New equilibrium: Q = {new_eq_x:.2f}, P = {new_eq_y:.2f}")
    print(f"Output files generated in {os.path.join(output_dir, 'supply_shift')}")
    
    # Create a comparison visualization showing both supply curves
    # (This would require an enhancement to our class to show multiple curves)
    
    # Open HTML file if requested
    if open_html:
        open_file(new_output['html'])

def generate_demand_shift(output_dir, open_html=False):
    """Generate a demand shift visualization"""
    # Create initial supply and demand plot
    config = {
        "supply_config": {
            "slope": 0.8,
            "intercept": 1,
            "color": "blue"
        },
        "demand_config": {
            "slope": -0.7,
            "intercept": 8,
            "color": "red",
            "name": "Demand (D₁)"
        },
        "layout_config": {
            "title": "Demand Shift Analysis",
            "width": 950,
            "height": 600
        }
    }
    
    sd_plot = SupplyDemandPlot(config)
    
    # Calculate initial equilibrium
    initial_eq_x, initial_eq_y = sd_plot.solve_equilibrium()
    print(f"Initial equilibrium: Q = {initial_eq_x:.2f}, P = {initial_eq_y:.2f}")
    
    # Save initial state
    os.makedirs(os.path.join(output_dir, 'demand_shift', 'initial'), exist_ok=True)
    initial_output = sd_plot.create_explanation(os.path.join(output_dir, 'demand_shift', 'initial'))
    
    # Shift the demand curve
    new_demand_config = {
        "intercept": 10,  # Higher intercept (increased demand)
        "color": "darkred",
    }
    
    # Apply the shift
    new_eq_x, new_eq_y = sd_plot.shift_demand(new_demand_config, label="Demand (D₂)")
    
    # Recalculate equilibrium with the new curve
    sd_plot.solve_equilibrium()
    
    # Save the new state
    os.makedirs(os.path.join(output_dir, 'demand_shift', 'after'), exist_ok=True)
    new_output = sd_plot.create_explanation(os.path.join(output_dir, 'demand_shift', 'after'))
    
    print(f"New equilibrium: Q = {new_eq_x:.2f}, P = {new_eq_y:.2f}")
    print(f"Output files generated in {os.path.join(output_dir, 'demand_shift')}")
    
    # Open HTML file if requested
    if open_html:
        open_file(new_output['html'])

def open_file(file_path):
    """Open a file with the default application"""
    if sys.platform == 'darwin':  # macOS
        import subprocess
        subprocess.run(['open', file_path])
    elif sys.platform == 'win32':  # Windows
        os.startfile(file_path)
    else:  # Linux or other Unix
        import subprocess
        subprocess.run(['xdg-open', file_path])

if __name__ == "__main__":
    sys.exit(main()) 