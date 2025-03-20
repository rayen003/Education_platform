"""
Simple Supply and Demand visualization using Matplotlib.
This module provides easier-to-use components for creating economic visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from gtts import gTTS
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib as mpl

# Set high-quality rendering defaults
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

class SupplyDemandPlot:
    """A simple supply and demand visualization using Matplotlib."""
    
    def __init__(self, config=None):
        """
        Initialize the supply and demand plot with configuration.
        
        Args:
            config: Dictionary with customization options for the plot
        """
        # Default configuration
        self.default_config = {
            "x_range": [0, 10, 1],
            "y_range": [0, 10, 1],
            "supply_config": {
                "slope": 0.5,
                "intercept": 2,
                "color": "#1f77b4",  # Blue
                "label": "Supply",
                "linestyle": "-",
                "linewidth": 2.5
            },
            "demand_config": {
                "slope": -0.5,
                "intercept": 8,
                "color": "#d62728",  # Red
                "label": "Demand",
                "linestyle": "-",
                "linewidth": 2.5
            },
            "equilibrium_config": {
                "color": "#2ca02c",  # Green
                "marker": "o",
                "markersize": 8,
                "show_lines": True,
                "line_style": "dashed",
                "line_width": 1.5,
                "show_labels": True,
                "label_fontsize": 11
            },
            "axis_config": {
                "x_label": "Quantity",
                "y_label": "Price",
                "title": "Supply and Demand Analysis",
                "grid": True,
                "show_axes": True,
                "tick_step": 1
            },
            "figure_config": {
                "figsize": (12, 7),
                "dpi": 120,
                "facecolor": "white",
                "tight_layout": True
            },
            "annotations_config": {
                "fontsize": 11,
                "text_x": 0.02,  # Position for text panel (% of figure width)
                "text_y": 0.95,  # Position for text panel (% of figure height)
                "text_width": 0.4,  # Width of text panel
                "text_color": "black",
                "text_box": True,
                "box_alpha": 0.1,
                "box_color": "gray"
            }
        }
        
        # Update with user config
        self.config = self.default_config.copy()
        if config:
            self._update_nested_dict(self.config, config)
            
        # Create the figure and axes
        self.setup_figure()
        
        # Initialize storage for elements
        self.elements = {
            "supply_curve": None,
            "demand_curve": None,
            "equilibrium_point": None,
            "eq_x_line": None,
            "eq_y_line": None,
            "annotations": [],
            "text_elements": []
        }
        
        # Store equilibrium values
        self.eq_x = None
        self.eq_y = None
        
        # Initialize annotation tracking
        self.last_annotation_y = self.config["annotations_config"]["text_y"]
    
    def _update_nested_dict(self, d, u):
        """Helper method to update nested dictionaries"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def setup_figure(self):
        """Set up the figure and axes for plotting."""
        fig_config = self.config["figure_config"]
        self.fig, self.ax = plt.subplots(
            figsize=fig_config["figsize"],
            dpi=fig_config["dpi"],
            facecolor=fig_config["facecolor"]
        )
        
        # Set axis labels and title
        axis_config = self.config["axis_config"]
        self.ax.set_xlabel(axis_config["x_label"], fontsize=12, fontweight='bold')
        self.ax.set_ylabel(axis_config["y_label"], fontsize=12, fontweight='bold')
        self.ax.set_title(axis_config["title"], fontsize=14, fontweight='bold')
        
        # Set axis ranges
        self.ax.set_xlim(self.config["x_range"][0], self.config["x_range"][1])
        self.ax.set_ylim(self.config["y_range"][0], self.config["y_range"][1])
        
        # Set grid
        self.ax.grid(axis_config["grid"], linestyle='--', alpha=0.7)
        
        # Set tick spacing
        tick_step = axis_config["tick_step"]
        self.ax.set_xticks(np.arange(self.config["x_range"][0], self.config["x_range"][1] + tick_step, tick_step))
        self.ax.set_yticks(np.arange(self.config["y_range"][0], self.config["y_range"][1] + tick_step, tick_step))
        
        # Ensure axes are visible
        self.ax.spines['left'].set_linewidth(1.5)
        self.ax.spines['bottom'].set_linewidth(1.5)
        self.ax.spines['right'].set_visible(True)
        self.ax.spines['top'].set_visible(True)
        self.ax.spines['right'].set_linewidth(1.5)
        self.ax.spines['top'].set_linewidth(1.5)
        
        # Use tight layout if configured
        if fig_config["tight_layout"]:
            plt.tight_layout(rect=[0, 0, 0.7, 1])  # Leave space for text panel
    
    def plot_supply_curve(self):
        """Plot the supply curve based on configuration."""
        supply_config = self.config["supply_config"]
        x_range = self.config["x_range"]
        
        # Generate x values
        x = np.linspace(x_range[0], x_range[1], 100)
        
        # Calculate y values using supply function
        y = supply_config["slope"] * x + supply_config["intercept"]
        
        # Plot the supply curve
        self.elements["supply_curve"] = self.ax.plot(
            x, y,
            linestyle=supply_config["linestyle"],
            color=supply_config["color"],
            linewidth=supply_config["linewidth"],
            label=supply_config["label"]
        )[0]
        
        # Add label to the curve at 80% of x-range
        x_pos = x_range[0] + 0.8 * (x_range[1] - x_range[0])
        y_pos = supply_config["slope"] * x_pos + supply_config["intercept"]
        
        # Add an offset to the label position
        self.ax.annotate(
            supply_config["label"],
            xy=(x_pos, y_pos),
            xytext=(10, 10),
            textcoords='offset points',
            color=supply_config["color"],
            fontsize=12,
            fontweight='bold'
        )
        
        return self.elements["supply_curve"]
    
    def plot_demand_curve(self):
        """Plot the demand curve based on configuration."""
        demand_config = self.config["demand_config"]
        x_range = self.config["x_range"]
        
        # Generate x values
        x = np.linspace(x_range[0], x_range[1], 100)
        
        # Calculate y values using demand function
        y = demand_config["slope"] * x + demand_config["intercept"]
        
        # Plot the demand curve
        self.elements["demand_curve"] = self.ax.plot(
            x, y,
            linestyle=demand_config["linestyle"],
            color=demand_config["color"],
            linewidth=demand_config["linewidth"],
            label=demand_config["label"]
        )[0]
        
        # Add label to the curve at 80% of x-range
        x_pos = x_range[0] + 0.8 * (x_range[1] - x_range[0])
        y_pos = demand_config["slope"] * x_pos + demand_config["intercept"]
        
        # Add an offset to the label position
        self.ax.annotate(
            demand_config["label"],
            xy=(x_pos, y_pos),
            xytext=(10, -20),
            textcoords='offset points',
            color=demand_config["color"],
            fontsize=12,
            fontweight='bold'
        )
        
        return self.elements["demand_curve"]
    
    def calculate_equilibrium(self):
        """
        Calculate the equilibrium point (where supply equals demand).
        
        Returns:
            tuple: (equilibrium_x, equilibrium_y)
        """
        supply_config = self.config["supply_config"]
        demand_config = self.config["demand_config"]
        
        # Equilibrium occurs when:
        # supply_slope * x + supply_intercept = demand_slope * x + demand_intercept
        # (supply_slope - demand_slope) * x = demand_intercept - supply_intercept
        # x = (demand_intercept - supply_intercept) / (supply_slope - demand_slope)
        
        self.eq_x = (demand_config["intercept"] - supply_config["intercept"]) / (supply_config["slope"] - demand_config["slope"])
        self.eq_y = supply_config["slope"] * self.eq_x + supply_config["intercept"]
        
        return (self.eq_x, self.eq_y)
    
    def plot_equilibrium(self):
        """Plot the equilibrium point and optional dashed lines to axes."""
        if self.eq_x is None or self.eq_y is None:
            self.calculate_equilibrium()
            
        eq_config = self.config["equilibrium_config"]
        
        # Plot the equilibrium point
        self.elements["equilibrium_point"] = self.ax.plot(
            [self.eq_x], [self.eq_y],
            marker=eq_config["marker"],
            markersize=eq_config["markersize"],
            color=eq_config["color"],
            label="Equilibrium"
        )[0]
        
        # Add dashed lines to axes if configured
        if eq_config["show_lines"]:
            # Vertical line to x-axis
            self.elements["eq_x_line"] = self.ax.plot(
                [self.eq_x, self.eq_x], [0, self.eq_y],
                linestyle=eq_config["line_style"],
                linewidth=eq_config["line_width"],
                color=eq_config["color"],
                alpha=0.7
            )[0]
            
            # Horizontal line to y-axis
            self.elements["eq_y_line"] = self.ax.plot(
                [0, self.eq_x], [self.eq_y, self.eq_y],
                linestyle=eq_config["line_style"],
                linewidth=eq_config["line_width"],
                color=eq_config["color"],
                alpha=0.7
            )[0]
        
        # Add labels for the equilibrium values
        if eq_config["show_labels"]:
            self.ax.annotate(
                f"Q = {self.eq_x:.1f}",
                xy=(self.eq_x, 0),
                xytext=(0, -20),
                textcoords='offset points',
                ha='center',
                fontsize=eq_config["label_fontsize"],
                color=eq_config["color"],
                fontweight='bold'
            )
            
            self.ax.annotate(
                f"P = {self.eq_y:.1f}",
                xy=(0, self.eq_y),
                xytext=(-40, 0),
                textcoords='offset points',
                va='center',
                fontsize=eq_config["label_fontsize"],
                color=eq_config["color"],
                fontweight='bold'
            )
            
        return self.elements["equilibrium_point"]
    
    def add_text_block(self, title, text, fontsize=None, y_offset=None):
        """
        Add a text block to the right side of the figure.
        
        Args:
            title: Title of the text block
            text: Content of the text block
            fontsize: Optional font size override
            y_offset: Optional vertical offset
            
        Returns:
            The text annotation object
        """
        ann_config = self.config["annotations_config"]
        fontsize = fontsize or ann_config["fontsize"]
        
        # Calculate position
        text_x = ann_config["text_x"] + 0.7  # Shift to right side
        if y_offset:
            text_y = self.last_annotation_y - y_offset
        else:
            # Automatically position below previous text blocks
            text_y = self.last_annotation_y - 0.05
        
        # Format the text with title
        formatted_text = f"{title}:\n{text}" if title else text
        
        # Add the text annotation
        text_obj = self.fig.text(
            text_x, text_y,
            formatted_text,
            fontsize=fontsize,
            color=ann_config["text_color"],
            verticalalignment='top',
            bbox=dict(
                facecolor=ann_config["box_color"],
                alpha=ann_config["box_alpha"],
                edgecolor='gray',
                boxstyle='round,pad=0.5'
            ) if ann_config["text_box"] else None
        )
        
        # Add to list of elements and update position tracker
        self.elements["text_elements"].append(text_obj)
        
        # Update the last annotation y position based on approx text height
        # This is a rough estimate; a more accurate approach would measure actual text height
        lines = text.count('\n') + 1
        text_height = 0.03 * lines  # Estimate 3% figure height per line
        self.last_annotation_y = text_y - text_height
        
        return text_obj
    
    def add_calculation_steps(self, animate=False):
        """
        Add step-by-step calculation of equilibrium to the plot.
        
        Args:
            animate: Whether to create separate frames for animation (not implemented yet)
        """
        supply_config = self.config["supply_config"]
        demand_config = self.config["demand_config"]
        
        # Reset text position
        self.last_annotation_y = self.config["annotations_config"]["text_y"]
        
        # Step 1: Display the equations
        self.add_text_block(
            "Equations",
            f"Supply: P = {supply_config['intercept']} + {supply_config['slope']}Q\n"
            f"Demand: P = {demand_config['intercept']} {demand_config['slope']:+}Q"
        )
        
        # Step 2: Set equations equal
        self.add_text_block(
            "Finding Equilibrium",
            f"Supply = Demand\n"
            f"{supply_config['intercept']} + {supply_config['slope']}Q = {demand_config['intercept']} {demand_config['slope']:+}Q"
        )
        
        # Step 3: Rearrange to solve for Q
        combined_slope = supply_config['slope'] - demand_config['slope']
        combined_intercept = demand_config['intercept'] - supply_config['intercept']
        
        self.add_text_block(
            "Calculation Steps",
            f"{supply_config['slope']}Q - ({demand_config['slope']})Q = {demand_config['intercept']} - {supply_config['intercept']}\n"
            f"{combined_slope}Q = {combined_intercept}\n"
            f"Q = {combined_intercept}/{combined_slope} = {self.eq_x:.1f}"
        )
        
        # Step 4: Calculate P by substituting into supply equation
        self.add_text_block(
            "Equilibrium Price",
            f"P = {supply_config['intercept']} + {supply_config['slope']} × {self.eq_x:.1f}\n"
            f"P = {self.eq_y:.1f}"
        )
        
        # Step 5: Conclusion
        self.add_text_block(
            "Conclusion",
            "At equilibrium, the market clears with no shortage or surplus.\n"
            f"Equilibrium: Q = {self.eq_x:.1f}, P = {self.eq_y:.1f}"
        )
    
    def shift_supply_curve(self, new_config, update_plot=True):
        """
        Shift the supply curve to a new position.
        
        Args:
            new_config: Dictionary with new supply curve parameters
            update_plot: Whether to update the plot immediately
            
        Returns:
            The new supply curve element
        """
        # Update the supply configuration
        old_config = self.config["supply_config"].copy()
        self._update_nested_dict(self.config["supply_config"], new_config)
        
        if update_plot:
            # Remove old supply curve
            if self.elements["supply_curve"]:
                self.elements["supply_curve"].remove()
            
            # Plot new supply curve
            self.plot_supply_curve()
            
            # Recalculate equilibrium
            self.calculate_equilibrium()
            
            # Update equilibrium visualization
            if self.elements["equilibrium_point"]:
                # Remove old equilibrium elements
                self.elements["equilibrium_point"].remove()
                if self.elements["eq_x_line"]:
                    self.elements["eq_x_line"].remove()
                if self.elements["eq_y_line"]:
                    self.elements["eq_y_line"].remove()
                
                # Add new equilibrium elements
                self.plot_equilibrium()
            
            # Clear text annotations and recalculate
            for text in self.elements["text_elements"]:
                text.remove()
            self.elements["text_elements"] = []
            
            # Add updated calculation steps
            self.add_calculation_steps()
            
            # Refresh the plot
            self.fig.canvas.draw()
        
        return self.elements["supply_curve"]
    
    def shift_demand_curve(self, new_config, update_plot=True):
        """
        Shift the demand curve to a new position.
        
        Args:
            new_config: Dictionary with new demand curve parameters
            update_plot: Whether to update the plot immediately
            
        Returns:
            The new demand curve element
        """
        # Update the demand configuration
        old_config = self.config["demand_config"].copy()
        self._update_nested_dict(self.config["demand_config"], new_config)
        
        if update_plot:
            # Remove old demand curve
            if self.elements["demand_curve"]:
                self.elements["demand_curve"].remove()
            
            # Plot new demand curve
            self.plot_demand_curve()
            
            # Recalculate equilibrium
            self.calculate_equilibrium()
            
            # Update equilibrium visualization
            if self.elements["equilibrium_point"]:
                # Remove old equilibrium elements
                self.elements["equilibrium_point"].remove()
                if self.elements["eq_x_line"]:
                    self.elements["eq_x_line"].remove()
                if self.elements["eq_y_line"]:
                    self.elements["eq_y_line"].remove()
                
                # Add new equilibrium elements
                self.plot_equilibrium()
            
            # Clear text annotations and recalculate
            for text in self.elements["text_elements"]:
                text.remove()
            self.elements["text_elements"] = []
            
            # Add updated calculation steps
            self.add_calculation_steps()
            
            # Refresh the plot
            self.fig.canvas.draw()
        
        return self.elements["demand_curve"]
    
    def generate_audio(self, text, filename="explanation"):
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
    
    def create_supply_demand_plot(self, save_path=None, show=True):
        """
        Create a complete supply and demand plot with all elements.
        
        Args:
            save_path: Optional path to save the figure
            show: Whether to show the plot
            
        Returns:
            The figure object
        """
        # Plot supply and demand curves
        self.plot_supply_curve()
        self.plot_demand_curve()
        
        # Calculate and plot equilibrium
        self.calculate_equilibrium()
        self.plot_equilibrium()
        
        # Add calculation steps
        self.add_calculation_steps()
        
        # Add legend
        self.ax.legend(loc='upper right')
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=self.config["figure_config"]["dpi"], bbox_inches='tight')
        
        # Show the plot if requested
        if show:
            plt.show()
        
        return self.fig 