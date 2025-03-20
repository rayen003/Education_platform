#!/usr/bin/env python
"""
Animated Supply and Demand Example

This script demonstrates how to create animated supply and demand visualizations
with synchronized narration using Plotly.
"""

import os
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from supply_demand import SupplyDemandPlot
from gtts import gTTS
import json
import argparse

class AnimatedSupplyDemandPlot(SupplyDemandPlot):
    """
    Enhanced version of SupplyDemandPlot with animation capabilities.
    """
    
    def create_animated_shift(self, shift_type="supply", new_config=None, frames=30, duration=1000, output_dir="./output"):
        """
        Create an animation of shifting supply or demand curve.
        
        Args:
            shift_type (str): Type of shift - "supply" or "demand"
            new_config (dict): New parameters for the curve
            frames (int): Number of animation frames
            duration (int): Animation duration in milliseconds
            output_dir (str): Directory to save output files
            
        Returns:
            str: Path to the HTML file
        """
        # Set up default new configurations if none provided
        if new_config is None:
            if shift_type == "supply":
                new_config = {
                    "intercept": self.config["supply_config"]["intercept"] + 2,
                    "color": "darkblue"
                }
            else:  # demand
                new_config = {
                    "intercept": self.config["demand_config"]["intercept"] + 2,
                    "color": "darkred"
                }
        
        # Store original configuration
        orig_config = {}
        if shift_type == "supply":
            orig_config = self.config["supply_config"].copy()
        else:
            orig_config = self.config["demand_config"].copy()
        
        # Calculate original equilibrium
        orig_eq_x, orig_eq_y = self.eq_x, self.eq_y
        
        # Create the animated figure
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.7, 0.3],
            specs=[[{"type": "xy"}, {"type": "table"}]]
        )
        
        # Create initial curves
        x_min, x_max = self.config["x_range"]
        x_values = np.linspace(x_min, x_max, 100)
        
        # Supply curve
        y_supply = self.supply_function(x_values)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_supply,
                mode="lines",
                name=self.config["supply_config"]["name"],
                line=dict(color=self.config["supply_config"]["color"], width=3)
            ),
            row=1, col=1
        )
        
        # Demand curve
        y_demand = self.demand_function(x_values)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_demand,
                mode="lines",
                name=self.config["demand_config"]["name"],
                line=dict(color=self.config["demand_config"]["color"], width=3)
            ),
            row=1, col=1
        )
        
        # Add equilibrium point
        fig.add_trace(
            go.Scatter(
                x=[self.eq_x],
                y=[self.eq_y],
                mode="markers",
                name="Equilibrium",
                marker=dict(
                    color=self.config["layout_config"]["equilibrium_color"],
                    size=10,
                    symbol="circle"
                )
            ),
            row=1, col=1
        )
        
        # Equilibrium lines
        fig.add_trace(
            go.Scatter(
                x=[self.eq_x, self.eq_x],
                y=[0, self.eq_y],
                mode="lines",
                name="Eq. Quantity",
                line=dict(
                    color=self.config["layout_config"]["equilibrium_color"],
                    width=2,
                    dash="dash"
                ),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0, self.eq_x],
                y=[self.eq_y, self.eq_y],
                mode="lines",
                name="Eq. Price",
                line=dict(
                    color=self.config["layout_config"]["equilibrium_color"],
                    width=2,
                    dash="dash"
                ),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add table for calculation steps
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Equilibrium Analysis"],
                    fill_color="lightskyblue",
                    align="left",
                    font=dict(size=14)
                ),
                cells=dict(
                    values=[["Original equilibrium", f"Q = {self.eq_x:.2f}, P = {self.eq_y:.2f}"]],
                    fill_color="lavender",
                    align="left",
                    font=dict(size=12),
                    height=30
                )
            ),
            row=1, col=2
        )
        
        # Generate animation frames
        frames = []
        
        # First frame - initial state
        frames.append(go.Frame(data=fig.data, name="frame0"))
        
        # Create intermediate frames for curve shift
        for i in range(1, 31):
            # Calculate interpolation factor
            t = i / 30
            
            # Create a frame
            frame_data = []
            
            if shift_type == "supply":
                # Interpolate the supply curve
                new_intercept = (1-t) * orig_config["intercept"] + t * new_config["intercept"]
                temp_supply_function = lambda x: orig_config["slope"] * x + new_intercept
                
                # Supply curve
                new_supply_y = temp_supply_function(x_values)
                frame_data.append(
                    go.Scatter(
                        x=x_values,
                        y=new_supply_y,
                        mode="lines",
                        name=self.config["supply_config"]["name"],
                        line=dict(
                            color=self.config["supply_config"]["color"],
                            width=3
                        )
                    )
                )
                
                # Demand curve (unchanged)
                frame_data.append(
                    go.Scatter(
                        x=x_values,
                        y=y_demand,
                        mode="lines",
                        name=self.config["demand_config"]["name"],
                        line=dict(color=self.config["demand_config"]["color"], width=3)
                    )
                )
                
                # Calculate new equilibrium
                new_eq_x = (self.config["demand_config"]["intercept"] - new_intercept) / (orig_config["slope"] - self.config["demand_config"]["slope"])
                new_eq_y = temp_supply_function(new_eq_x)
                
            else:  # demand shift
                # Supply curve (unchanged)
                frame_data.append(
                    go.Scatter(
                        x=x_values,
                        y=y_supply,
                        mode="lines",
                        name=self.config["supply_config"]["name"],
                        line=dict(color=self.config["supply_config"]["color"], width=3)
                    )
                )
                
                # Interpolate the demand curve
                new_intercept = (1-t) * orig_config["intercept"] + t * new_config["intercept"]
                temp_demand_function = lambda x: orig_config["slope"] * x + new_intercept
                
                # Demand curve
                new_demand_y = temp_demand_function(x_values)
                frame_data.append(
                    go.Scatter(
                        x=x_values,
                        y=new_demand_y,
                        mode="lines",
                        name=self.config["demand_config"]["name"],
                        line=dict(
                            color=self.config["demand_config"]["color"],
                            width=3
                        )
                    )
                )
                
                # Calculate new equilibrium
                new_eq_x = (new_intercept - self.config["supply_config"]["intercept"]) / (self.config["supply_config"]["slope"] - orig_config["slope"])
                new_eq_y = self.supply_function(new_eq_x)
            
            # Add equilibrium point
            frame_data.append(
                go.Scatter(
                    x=[new_eq_x],
                    y=[new_eq_y],
                    mode="markers",
                    name="Equilibrium",
                    marker=dict(
                        color=self.config["layout_config"]["equilibrium_color"],
                        size=10,
                        symbol="circle"
                    )
                )
            )
            
            # Equilibrium lines
            frame_data.append(
                go.Scatter(
                    x=[new_eq_x, new_eq_x],
                    y=[0, new_eq_y],
                    mode="lines",
                    name="Eq. Quantity",
                    line=dict(
                        color=self.config["layout_config"]["equilibrium_color"],
                        width=2,
                        dash="dash"
                    ),
                    showlegend=False
                )
            )
            
            frame_data.append(
                go.Scatter(
                    x=[0, new_eq_x],
                    y=[new_eq_y, new_eq_y],
                    mode="lines",
                    name="Eq. Price",
                    line=dict(
                        color=self.config["layout_config"]["equilibrium_color"],
                        width=2,
                        dash="dash"
                    ),
                    showlegend=False
                )
            )
            
            # Add table for calculation steps
            message = "Shifting curve..." if i < 20 else f"New equilibrium: Q = {new_eq_x:.2f}, P = {new_eq_y:.2f}"
            
            frame_data.append(
                go.Table(
                    header=dict(
                        values=["Equilibrium Analysis"],
                        fill_color="lightskyblue",
                        align="left",
                        font=dict(size=14)
                    ),
                    cells=dict(
                        values=[["Original equilibrium", f"Q = {orig_eq_x:.2f}, P = {orig_eq_y:.2f}", 
                                message]],
                        fill_color="lavender",
                        align="left",
                        font=dict(size=12),
                        height=30
                    )
                )
            )
            
            frames.append(go.Frame(data=frame_data, name=f"frame{i}"))
        
        # Update layout with animation
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": duration / 30, "redraw": True}, 
                                       "fromcurrent": True}]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, 
                                         "mode": "immediate"}]
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "steps": [
                    {
                        "label": f"{i}",
                        "method": "animate",
                        "args": [[f"frame{i}"], {"frame": {"duration": 0, "redraw": True}, 
                                               "mode": "immediate"}]
                    }
                    for i in range(31)
                ],
                "x": 0.1,
                "y": 0,
                "len": 0.9,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"b": 10, "t": 50},
                "currentvalue": {
                    "visible": True,
                    "prefix": "Frame: ",
                    "xanchor": "center",
                    "font": {"size": 12, "color": "#666"}
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"}
            }]
        )
        
        # Update layout
        shift_title = "Supply Shift" if shift_type == "supply" else "Demand Shift"
        fig.update_layout(
            title=f"Animated {shift_title} Analysis",
            width=self.config["layout_config"]["width"],
            height=self.config["layout_config"]["height"],
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=50, r=50, t=80, b=100),
            xaxis=dict(
                title="Quantity",
                showgrid=True,
                zeroline=True,
                range=self.config["x_range"]
            ),
            yaxis=dict(
                title="Price",
                showgrid=True,
                zeroline=True,
                range=self.config["y_range"]
            )
        )
        
        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the interactive figure
        html_path = os.path.join(output_dir, f"animated_{shift_type}_shift.html")
        fig.write_html(
            html_path,
            include_plotlyjs=True,
            full_html=True,
            auto_open=False
        )
        
        # Generate narration
        self.generate_narration(shift_type, orig_eq_x, orig_eq_y, new_eq_x, new_eq_y, output_dir)
        
        return html_path
    
    def generate_narration(self, shift_type, orig_eq_x, orig_eq_y, new_eq_x, new_eq_y, output_dir):
        """
        Generate an audio narration for the animated shift.
        
        Args:
            shift_type (str): Type of shift - "supply" or "demand"
            orig_eq_x (float): Original equilibrium quantity
            orig_eq_y (float): Original equilibrium price
            new_eq_x (float): New equilibrium quantity
            new_eq_y (float): New equilibrium price
            output_dir (str): Directory to save audio file
            
        Returns:
            str: Path to the audio file
        """
        # Create narration text
        narration = [
            f"Welcome to the animated {shift_type} shift analysis.",
            f"We start with the original {shift_type} and demand curves, with an equilibrium quantity of {orig_eq_x:.1f} and price of {orig_eq_y:.1f}."
        ]
        
        if shift_type == "supply":
            narration.append("Now, let's see what happens when the supply curve shifts upward.")
            narration.append("This could be due to factors like increased production costs, resource constraints, or new regulations.")
            
            if new_eq_x < orig_eq_x and new_eq_y > orig_eq_y:
                narration.append(f"As the supply curve shifts upward, we can see that the equilibrium quantity decreases from {orig_eq_x:.1f} to {new_eq_x:.1f}.")
                narration.append(f"And the equilibrium price increases from {orig_eq_y:.1f} to {new_eq_y:.1f}.")
                narration.append("This confirms the economic principle that a decrease in supply leads to higher prices and lower quantities in the market.")
            else:
                narration.append(f"The new equilibrium occurs at quantity {new_eq_x:.1f} and price {new_eq_y:.1f}.")
        else:
            narration.append("Now, let's see what happens when the demand curve shifts.")
            narration.append("This could be due to factors like changes in consumer income, preferences, or population size.")
            
            if new_eq_x > orig_eq_x and new_eq_y > orig_eq_y:
                narration.append(f"As the demand curve shifts upward, we can see that the equilibrium quantity increases from {orig_eq_x:.1f} to {new_eq_x:.1f}.")
                narration.append(f"And the equilibrium price increases from {orig_eq_y:.1f} to {new_eq_y:.1f}.")
                narration.append("This confirms the economic principle that an increase in demand leads to higher prices and higher quantities in the market.")
            else:
                narration.append(f"The new equilibrium occurs at quantity {new_eq_x:.1f} and price {new_eq_y:.1f}.")
        
        # Join narration text
        narration_text = " ".join(narration)
        
        # Generate audio
        audio_path = os.path.join(output_dir, f"animated_{shift_type}_shift.mp3")
        tts = gTTS(text=narration_text, lang='en', slow=False)
        tts.save(audio_path)
        
        # Save narration text
        text_path = os.path.join(output_dir, f"animated_{shift_type}_shift.txt")
        with open(text_path, "w") as f:
            f.write(narration_text)
        
        return audio_path

def main():
    """Main function to run animated examples"""
    parser = argparse.ArgumentParser(description='Generate animated supply and demand visualizations')
    
    parser.add_argument('--shift', 
                        type=str, 
                        default='supply',
                        choices=['supply', 'demand'],
                        help='Which type of shift to animate')
    
    parser.add_argument('--output_dir', 
                        type=str, 
                        default='./output/animated',
                        help='Directory to save the output files')
    
    parser.add_argument('--open', 
                        action='store_true',
                        help='Open the HTML visualization after generating')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
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
            "title": f"Animated {args.shift.capitalize()} Shift",
            "width": 950,
            "height": 600
        }
    }
    
    # Create the animated plot
    print(f"Generating animated {args.shift} shift...")
    
    sd_plot = AnimatedSupplyDemandPlot(config)
    
    # Define new configuration for the shift
    if args.shift == "supply":
        new_config = {
            "intercept": 3,  # Higher intercept (decreased supply)
            "color": "rgb(0, 0, 150)"  # Darker blue
        }
    else:  # demand shift
        new_config = {
            "intercept": 10,  # Higher intercept (increased demand)
            "color": "rgb(150, 0, 0)"  # Darker red
        }
    
    # Create the animation
    html_path = sd_plot.create_animated_shift(
        shift_type=args.shift,
        new_config=new_config,
        output_dir=args.output_dir
    )
    
    print(f"Animation generated successfully!")
    print(f"Output file: {html_path}")
    
    # Open HTML file if requested
    if args.open:
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