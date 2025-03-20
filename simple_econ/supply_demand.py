"""
Simple Supply and Demand curve visualization using Plotly.
This module provides an easy way to create and visualize supply and demand curves
and calculate equilibrium points.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from gtts import gTTS

class SupplyDemandPlot:
    """
    A simple supply and demand curve visualization using Plotly.
    
    This class creates interactive supply and demand visualizations, calculates
    equilibrium points, and can generate explanations with step-by-step calculations.
    """
    
    def __init__(self, config=None):
        """
        Initialize a new supply and demand visualization.
        
        Args:
            config (dict): Configuration dictionary with parameters for the visualization
        """
        # Default configuration
        self.default_config = {
            "x_range": [0, 10],
            "y_range": [0, 10],
            "supply_config": {
                "slope": 0.5,
                "intercept": 2,
                "color": "blue",
                "name": "Supply"
            },
            "demand_config": {
                "slope": -0.5,
                "intercept": 8,
                "color": "red",
                "name": "Demand"
            },
            "layout_config": {
                "title": "Supply and Demand Analysis",
                "width": 950,
                "height": 600,
                "show_equilibrium": True,
                "show_steps": True,
                "equilibrium_color": "green"
            }
        }
        
        # Update configuration with provided values
        self.config = self.default_config.copy()
        if config:
            self._update_nested_dict(self.config, config)
        
        # Create supply and demand functions
        self.supply_function = lambda x: self.config["supply_config"]["slope"] * x + self.config["supply_config"]["intercept"]
        self.demand_function = lambda x: self.config["demand_config"]["slope"] * x + self.config["demand_config"]["intercept"]
        
        # Calculate equilibrium
        self._calculate_equilibrium()
        
        # Steps for calculating equilibrium (for display)
        self.calculation_steps = []
    
    def _update_nested_dict(self, d, u):
        """Helper method to update nested dictionaries"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def _calculate_equilibrium(self):
        """Calculate the equilibrium point where supply equals demand"""
        supply_slope = self.config["supply_config"]["slope"]
        supply_intercept = self.config["supply_config"]["intercept"]
        demand_slope = self.config["demand_config"]["slope"]
        demand_intercept = self.config["demand_config"]["intercept"]
        
        # Solve: supply_slope * x + supply_intercept = demand_slope * x + demand_intercept
        # (supply_slope - demand_slope) * x = demand_intercept - supply_intercept
        # x = (demand_intercept - supply_intercept) / (supply_slope - demand_slope)
        
        self.eq_x = (demand_intercept - supply_intercept) / (supply_slope - demand_slope)
        self.eq_y = self.supply_function(self.eq_x)
    
    def solve_equilibrium(self):
        """
        Calculate and return the steps to solve for equilibrium.
        
        Returns:
            tuple: The equilibrium quantity and price (x, y)
        """
        # Clear previous steps
        self.calculation_steps = []
        
        # Get config values
        supply_slope = self.config["supply_config"]["slope"]
        supply_intercept = self.config["supply_config"]["intercept"]
        demand_slope = self.config["demand_config"]["slope"]
        demand_intercept = self.config["demand_config"]["intercept"]
        
        # Step 1: Set up the equations
        self.calculation_steps.append({
            "title": "Finding Equilibrium:",
            "content": "",
            "color": "blue"
        })
        
        # Step 2: Write the equations
        self.calculation_steps.append({
            "title": "Step 1: Set up the equations",
            "content": f"Supply: P = {supply_intercept} + {supply_slope}Q",
            "color": self.config["supply_config"]["color"]
        })
        
        self.calculation_steps.append({
            "content": f"Demand: P = {demand_intercept} {demand_slope:+}Q",
            "color": self.config["demand_config"]["color"]
        })
        
        # Step 3: Set equal and solve
        equation_text = f"{supply_intercept} + {supply_slope}Q = {demand_intercept} {demand_slope:+}Q"
        self.calculation_steps.append({
            "title": "Step 2: Set supply equal to demand",
            "content": equation_text,
            "color": "black"
        })
        
        # Step 4: Rearrange to solve for Q
        combined_slope = supply_slope - demand_slope
        combined_intercept = demand_intercept - supply_intercept
        
        rearrange_text = f"{supply_slope}Q - ({demand_slope})Q = {demand_intercept} - {supply_intercept}"
        self.calculation_steps.append({
            "title": "Step 3: Rearrange to solve for Q",
            "content": rearrange_text,
            "color": "black"
        })
        
        # Step 5: Simplify
        simplify_text = f"{combined_slope}Q = {combined_intercept}"
        self.calculation_steps.append({
            "content": simplify_text,
            "color": "black"
        })
        
        # Step 6: Solve for Q
        q_text = f"Q = {combined_intercept}/{combined_slope} = {self.eq_x:.2f}"
        self.calculation_steps.append({
            "content": q_text,
            "color": "black"
        })
        
        # Step 7: Solve for P
        p_text = f"P = {supply_intercept} + {supply_slope} × {self.eq_x:.2f} = {self.eq_y:.2f}"
        self.calculation_steps.append({
            "title": "Step 4: Calculate the equilibrium price",
            "content": p_text,
            "color": "black"
        })
        
        # Step 8: Conclusion
        self.calculation_steps.append({
            "title": "Equilibrium",
            "content": f"Equilibrium Quantity: {self.eq_x:.2f}\nEquilibrium Price: {self.eq_y:.2f}",
            "color": self.config["layout_config"]["equilibrium_color"]
        })
        
        return self.eq_x, self.eq_y
    
    def create_plot(self):
        """
        Create and return an interactive Plotly figure with supply and demand curves.
        
        Returns:
            plotly.graph_objects.Figure: The Plotly figure object
        """
        # Create a subplot with a single plot and a side panel for calculations
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.7, 0.3],
            specs=[[{"type": "xy"}, {"type": "table"}]]
        )
        
        # X values for the curves
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
        
        # Add equilibrium point if configured to show it
        if self.config["layout_config"]["show_equilibrium"]:
            # Equilibrium point
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
            
            # Vertical line to x-axis
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
            
            # Horizontal line to y-axis
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
            
            # Labels for equilibrium values
            fig.add_annotation(
                x=self.eq_x, y=0,
                text=f"Q = {self.eq_x:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=self.config["layout_config"]["equilibrium_color"],
                ax=0, ay=30
            )
            
            fig.add_annotation(
                x=0, y=self.eq_y,
                text=f"P = {self.eq_y:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=self.config["layout_config"]["equilibrium_color"],
                ax=30, ay=0
            )
        
        # Create the table for calculation steps if we have any
        if self.config["layout_config"]["show_steps"] and self.calculation_steps:
            # Extract step information for the table
            headers = ["Calculation Steps"]
            cells = [[step.get("title", "") + " " + step.get("content", "") for step in self.calculation_steps]]
            
            # Add table trace
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=headers,
                        fill_color="lightskyblue",
                        align="left",
                        font=dict(size=14)
                    ),
                    cells=dict(
                        values=cells,
                        fill_color="lavender",
                        align="left",
                        font=dict(size=12),
                        height=30
                    )
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=self.config["layout_config"]["title"],
            width=self.config["layout_config"]["width"],
            height=self.config["layout_config"]["height"],
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=50, r=50, t=80, b=50),
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
        
        return fig
    
    def save_html(self, filename="supply_demand.html"):
        """
        Save the visualization as an interactive HTML file.
        
        Args:
            filename (str): Name of the output HTML file
        
        Returns:
            str: Path to the saved HTML file
        """
        # Create the figure
        fig = self.create_plot()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Save to HTML
        fig.write_html(
            filename,
            include_plotlyjs=True,
            full_html=True,
            auto_open=False
        )
        
        return filename
    
    def save_image(self, filename="supply_demand.png"):
        """
        Save the visualization as a static image file.
        
        Args:
            filename (str): Name of the output image file
        
        Returns:
            str: Path to the saved image file
        """
        # Create the figure
        fig = self.create_plot()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Save to image
        fig.write_image(filename)
        
        return filename
    
    def create_explanation(self, output_dir="./output"):
        """
        Create a complete explanation with visualization, calculations, and optional audio.
        
        Args:
            output_dir (str): Directory to save the output files
        
        Returns:
            dict: Paths to the generated files and calculation steps
        """
        # Make sure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate equilibrium and get steps
        self.solve_equilibrium()
        
        # Save the visualization
        html_path = os.path.join(output_dir, "supply_demand.html")
        self.save_html(html_path)
        
        # Save image version
        img_path = os.path.join(output_dir, "supply_demand.png")
        self.save_image(img_path)
        
        # Generate text explanation
        explanation_text = self._generate_text_explanation()
        
        # Save explanation text
        text_path = os.path.join(output_dir, "explanation.txt")
        with open(text_path, "w") as f:
            f.write(explanation_text)
        
        # Generate audio explanation
        audio_path = os.path.join(output_dir, "explanation.mp3")
        self._generate_audio_explanation(explanation_text, audio_path)
        
        # Save calculation steps as JSON for reference
        steps_path = os.path.join(output_dir, "calculation_steps.json")
        with open(steps_path, "w") as f:
            json.dump(self.calculation_steps, f, indent=2)
        
        # Return all paths
        return {
            "html": html_path,
            "image": img_path,
            "text": text_path,
            "audio": audio_path,
            "steps": steps_path,
            "calculation_steps": self.calculation_steps
        }
    
    def _generate_text_explanation(self):
        """
        Generate a complete textual explanation of the supply and demand analysis.
        
        Returns:
            str: The complete explanation text
        """
        # Title
        explanation = [self.config["layout_config"]["title"]]
        explanation.append("=" * len(explanation[0]))
        explanation.append("")
        
        # Equations
        explanation.append("Initial Equations:")
        explanation.append(f"Supply: P = {self.config['supply_config']['intercept']} + {self.config['supply_config']['slope']}Q")
        explanation.append(f"Demand: P = {self.config['demand_config']['intercept']} {self.config['demand_config']['slope']:+}Q")
        explanation.append("")
        
        # Calculation steps
        explanation.append("Step-by-Step Solution:")
        for step in self.calculation_steps:
            if "title" in step and step["title"]:
                explanation.append(f"{step['title']}")
            if "content" in step and step["content"]:
                explanation.append(f"  {step['content']}")
        explanation.append("")
        
        # Conclusion
        explanation.append("Conclusion:")
        explanation.append(f"At equilibrium, the market clears with a quantity of {self.eq_x:.2f} units and a price of {self.eq_y:.2f} dollars.")
        explanation.append("This is the point where supply equals demand, resulting in no shortage or surplus in the market.")
        
        return "\n".join(explanation)
    
    def _generate_audio_explanation(self, text, output_path):
        """
        Generate an audio file with the spoken explanation.
        
        Args:
            text (str): The text to convert to speech
            output_path (str): Path to save the audio file
        
        Returns:
            str: Path to the generated audio file
        """
        # Generate speech using gTTS
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_path)
        
        return output_path

    def shift_supply(self, new_config, label=None):
        """
        Shift the supply curve and recalculate equilibrium.
        
        Args:
            new_config (dict): New parameters for the supply curve
            label (str): Optional new label for the supply curve
        
        Returns:
            tuple: New equilibrium point (x, y)
        """
        # Store original supply config
        orig_supply = self.config["supply_config"].copy()
        
        # Update supply config
        self._update_nested_dict(self.config["supply_config"], new_config)
        
        # Update supply function
        self.supply_function = lambda x: self.config["supply_config"]["slope"] * x + self.config["supply_config"]["intercept"]
        
        # If label is provided, update it
        if label:
            self.config["supply_config"]["name"] = label
        
        # Recalculate equilibrium
        self._calculate_equilibrium()
        
        return self.eq_x, self.eq_y
    
    def shift_demand(self, new_config, label=None):
        """
        Shift the demand curve and recalculate equilibrium.
        
        Args:
            new_config (dict): New parameters for the demand curve
            label (str): Optional new label for the demand curve
        
        Returns:
            tuple: New equilibrium point (x, y)
        """
        # Store original demand config
        orig_demand = self.config["demand_config"].copy()
        
        # Update demand config
        self._update_nested_dict(self.config["demand_config"], new_config)
        
        # Update demand function
        self.demand_function = lambda x: self.config["demand_config"]["slope"] * x + self.config["demand_config"]["intercept"]
        
        # If label is provided, update it
        if label:
            self.config["demand_config"]["name"] = label
        
        # Recalculate equilibrium
        self._calculate_equilibrium()
        
        return self.eq_x, self.eq_y 