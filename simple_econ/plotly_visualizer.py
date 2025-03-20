"""
Plotly-based visualizer for economic animations.

This module provides a visualizer that generates visualizations based on the
configurations produced by the meta-agent, with comprehensive logging.
"""

import os
import sys
import json
import time
import logging
import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Plotly package not installed. Please install with: pip install plotly")

try:
    from gtts import gTTS
except ImportError:
    print("gTTS package not installed. Please install with: pip install gtts")

# Import our models and existing components
from config_models import AnimationConfig, SupplyDemandConfig, SupplyShiftConfig, DemandShiftConfig
from config_models import TimeValueConfig, PerpetuityConfig
from supply_demand import SupplyDemandPlot
from animated_example import AnimatedSupplyDemandPlot

# Set up logging
logger = logging.getLogger("plotly_visualizer")
logger.setLevel(logging.DEBUG)

# Create a file handler
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"visualizer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatters and add them to handlers
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_format = logging.Formatter('%(levelname)s: %(message)s')
file_handler.setFormatter(file_format)
console_handler.setFormatter(console_format)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class PlotlyVisualizer:
    """
    Plotly-based visualizer for economic animations.
    
    This class creates visualizations based on configurations from the meta-agent.
    It includes support for different types of economic visualizations and
    logs detailed information about the generation process.
    """
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory where output files will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up a separate logger for LLM calls
        self.llm_logger = logging.getLogger("llm_calls")
        self.llm_logger.setLevel(logging.DEBUG)
        
        # Create a separate file for LLM call logs
        llm_log_file = os.path.join(log_dir, f"llm_calls_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        llm_handler = logging.FileHandler(llm_log_file)
        llm_handler.setLevel(logging.DEBUG)
        llm_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        llm_handler.setFormatter(llm_format)
        self.llm_logger.addHandler(llm_handler)
        
        logger.info(f"PlotlyVisualizer initialized with output directory: {output_dir}")
        logger.info(f"Logs will be saved to: {log_file} and {llm_log_file}")
    
    def log_llm_call(self, call_type: str, params: Dict, response: Any = None, error: str = None):
        """
        Log information about an LLM API call.
        
        Args:
            call_type: Type of LLM call (e.g., "analyze", "verify")
            params: Parameters sent to the API
            response: Response received (if successful)
            error: Error message (if failed)
        """
        # Format as JSON for easy parsing
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "call_type": call_type,
            "params": params,
            "response": response,
            "error": error
        }
        
        # Write to the LLM log
        self.llm_logger.debug(json.dumps(log_entry))
    
    def create_visualization(self, config: AnimationConfig, output_format: str = "html") -> Dict[str, Any]:
        """
        Create a visualization based on the provided configuration.
        
        Args:
            config: Configuration for the visualization
            output_format: Desired output format (html, image)
            
        Returns:
            Dict containing information about the rendered visualization
        """
        logger.info(f"Creating visualization of type: {config.visualization_type}")
        
        # Create output directory for this visualization
        vis_output_dir = os.path.join(self.output_dir, config.visualization_type)
        os.makedirs(vis_output_dir, exist_ok=True)
        
        start_time = time.time()
        render_info = {
            "visualization_type": config.visualization_type,
            "start_time": datetime.datetime.now().isoformat(),
            "output_format": output_format,
            "output_dir": vis_output_dir
        }
        
        try:
            # Dispatch to appropriate visualization method based on type
            if config.visualization_type == "supply_demand":
                output_files = self._create_supply_demand(config, vis_output_dir, output_format)
            elif config.visualization_type == "supply_shift":
                output_files = self._create_supply_shift(config, vis_output_dir, output_format)
            elif config.visualization_type == "demand_shift":
                output_files = self._create_demand_shift(config, vis_output_dir, output_format)
            elif config.visualization_type == "time_value":
                output_files = self._create_time_value(config, vis_output_dir, output_format)
            elif config.visualization_type == "perpetuity":
                output_files = self._create_perpetuity(config, vis_output_dir, output_format)
            else:
                raise ValueError(f"Unsupported visualization type: {config.visualization_type}")
            
            render_info["output_files"] = output_files
            render_info["status"] = "success"
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}", exc_info=True)
            render_info["status"] = "error"
            render_info["error"] = str(e)
        
        # Add timing information
        end_time = time.time()
        render_info["end_time"] = datetime.datetime.now().isoformat()
        render_info["duration"] = end_time - start_time
        
        logger.info(f"Visualization generation completed in {render_info['duration']:.2f} seconds")
        
        return render_info
    
    def _create_supply_demand(self, config: AnimationConfig, output_dir: str, output_format: str) -> Dict[str, str]:
        """Create a basic supply-demand equilibrium visualization"""
        logger.info("Creating supply-demand equilibrium visualization")
        
        # Extract the supply demand config
        sd_config = config.supply_demand_config
        
        # Create a configuration for the SupplyDemandPlot
        plot_config = {
            "supply_config": {
                "slope": sd_config.supply_slope,
                "intercept": sd_config.supply_intercept,
                "color": sd_config.supply_color
            },
            "demand_config": {
                "slope": sd_config.demand_slope,
                "intercept": sd_config.demand_intercept,
                "color": sd_config.demand_color
            },
            "layout_config": {
                "title": config.title,
                "width": config.width,
                "height": config.height
            }
        }
        
        # Log configuration
        logger.debug(f"Supply-demand configuration: {json.dumps(plot_config)}")
        
        # Create the plot
        sd_plot = SupplyDemandPlot(plot_config)
        
        # Calculate equilibrium
        eq_x, eq_y = sd_plot.solve_equilibrium()
        logger.info(f"Calculated equilibrium: Q = {eq_x:.2f}, P = {eq_y:.2f}")
        
        # Generate the complete explanation
        explanation_dir = os.path.join(output_dir, "equilibrium")
        os.makedirs(explanation_dir, exist_ok=True)
        
        output_files = sd_plot.create_explanation(explanation_dir)
        
        if output_format == "html":
            # Use the HTML output
            return {
                "html": output_files["html"],
                "image": output_files["image"],
                "text": output_files["text"],
                "audio": output_files["audio"]
            }
        else:
            # Use the image output
            return {
                "image": output_files["image"],
                "text": output_files["text"],
                "audio": output_files["audio"]
            }
    
    def _create_supply_shift(self, config: AnimationConfig, output_dir: str, output_format: str) -> Dict[str, str]:
        """Create a supply shift visualization"""
        logger.info("Creating supply shift visualization")
        
        # Extract the supply shift config
        shift_config = config.supply_shift_config
        
        # Create a configuration for the SupplyDemandPlot
        plot_config = {
            "supply_config": {
                "slope": shift_config.supply_slope,
                "intercept": shift_config.supply_intercept,
                "color": shift_config.supply_color,
                "name": "Initial Supply"
            },
            "demand_config": {
                "slope": shift_config.demand_slope,
                "intercept": shift_config.demand_intercept,
                "color": shift_config.demand_color
            },
            "layout_config": {
                "title": config.title,
                "width": config.width,
                "height": config.height
            }
        }
        
        # Log configuration
        logger.debug(f"Supply shift configuration: {json.dumps(plot_config)}")
        
        if output_format == "html" and shift_config.frames > 0:
            # Use the animated version
            return self._create_animated_supply_shift(config, output_dir)
        
        # Create the plot for static visualization
        sd_plot = SupplyDemandPlot(plot_config)
        
        # Calculate initial equilibrium
        initial_eq_x, initial_eq_y = sd_plot.solve_equilibrium()
        logger.info(f"Initial equilibrium: Q = {initial_eq_x:.2f}, P = {initial_eq_y:.2f}")
        
        # Save initial state
        os.makedirs(os.path.join(output_dir, "initial"), exist_ok=True)
        initial_output = sd_plot.create_explanation(os.path.join(output_dir, "initial"))
        
        # Prepare new supply config with the shifted parameters
        new_supply_config = {
            "intercept": shift_config.new_supply_intercept,
            "color": "darkblue",
        }
        
        # Add new slope if provided
        if shift_config.new_supply_slope is not None:
            new_supply_config["slope"] = shift_config.new_supply_slope
        
        # Apply the shift
        new_eq_x, new_eq_y = sd_plot.shift_supply(new_supply_config, label="New Supply")
        
        # Recalculate equilibrium with the new curve
        sd_plot.solve_equilibrium()
        
        # Save the new state
        os.makedirs(os.path.join(output_dir, "after"), exist_ok=True)
        new_output = sd_plot.create_explanation(os.path.join(output_dir, "after"))
        
        logger.info(f"New equilibrium: Q = {new_eq_x:.2f}, P = {new_eq_y:.2f}")
        
        return {
            "initial_html": initial_output["html"],
            "initial_image": initial_output["image"],
            "initial_text": initial_output["text"],
            "initial_audio": initial_output["audio"],
            "after_html": new_output["html"],
            "after_image": new_output["image"],
            "after_text": new_output["text"],
            "after_audio": new_output["audio"]
        }
    
    def _create_animated_supply_shift(self, config: AnimationConfig, output_dir: str) -> Dict[str, str]:
        """Create an animated supply shift visualization"""
        logger.info("Creating animated supply shift visualization")
        
        # Extract the supply shift config
        shift_config = config.supply_shift_config
        
        # Create a configuration for AnimatedSupplyDemandPlot
        plot_config = {
            "supply_config": {
                "slope": shift_config.supply_slope,
                "intercept": shift_config.supply_intercept,
                "color": shift_config.supply_color
            },
            "demand_config": {
                "slope": shift_config.demand_slope,
                "intercept": shift_config.demand_intercept,
                "color": shift_config.demand_color
            },
            "layout_config": {
                "title": config.title,
                "width": config.width,
                "height": config.height
            }
        }
        
        # Create the animated plot
        sd_plot = AnimatedSupplyDemandPlot(plot_config)
        
        # Prepare new supply config
        new_supply_config = {
            "intercept": shift_config.new_supply_intercept,
            "color": "darkblue",
        }
        
        # Add new slope if provided
        if shift_config.new_supply_slope is not None:
            new_supply_config["slope"] = shift_config.new_supply_slope
        
        # Create the animation
        animated_dir = os.path.join(output_dir, "animated")
        os.makedirs(animated_dir, exist_ok=True)
        
        html_path = sd_plot.create_animated_shift(
            shift_type="supply",
            new_config=new_supply_config,
            frames=shift_config.frames,
            output_dir=animated_dir
        )
        
        audio_path = os.path.join(animated_dir, "animated_supply_shift.mp3")
        
        # Create synchronized HTML
        interactive_dir = os.path.join(output_dir, "interactive")
        os.makedirs(interactive_dir, exist_ok=True)
        
        # Import create_synced_html from create_synced_animation
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from create_synced_animation import create_synced_html
        
        synced_html_path = create_synced_html(
            shift_type="supply",
            html_path=html_path,
            audio_path=audio_path,
            output_dir=interactive_dir
        )
        
        return {
            "animated_html": html_path,
            "audio": audio_path,
            "interactive_html": synced_html_path
        }
    
    def _create_demand_shift(self, config: AnimationConfig, output_dir: str, output_format: str) -> Dict[str, str]:
        """Create a demand shift visualization"""
        logger.info("Creating demand shift visualization")
        
        # Extract the demand shift config
        shift_config = config.demand_shift_config
        
        # Create a configuration for the SupplyDemandPlot
        plot_config = {
            "supply_config": {
                "slope": shift_config.supply_slope,
                "intercept": shift_config.supply_intercept,
                "color": shift_config.supply_color
            },
            "demand_config": {
                "slope": shift_config.demand_slope,
                "intercept": shift_config.demand_intercept,
                "color": shift_config.demand_color,
                "name": "Initial Demand"
            },
            "layout_config": {
                "title": config.title,
                "width": config.width,
                "height": config.height
            }
        }
        
        # Log configuration
        logger.debug(f"Demand shift configuration: {json.dumps(plot_config)}")
        
        if output_format == "html" and shift_config.frames > 0:
            # Use the animated version
            return self._create_animated_demand_shift(config, output_dir)
        
        # Create the plot for static visualization
        sd_plot = SupplyDemandPlot(plot_config)
        
        # Calculate initial equilibrium
        initial_eq_x, initial_eq_y = sd_plot.solve_equilibrium()
        logger.info(f"Initial equilibrium: Q = {initial_eq_x:.2f}, P = {initial_eq_y:.2f}")
        
        # Save initial state
        os.makedirs(os.path.join(output_dir, "initial"), exist_ok=True)
        initial_output = sd_plot.create_explanation(os.path.join(output_dir, "initial"))
        
        # Prepare new demand config with the shifted parameters
        new_demand_config = {
            "intercept": shift_config.new_demand_intercept,
            "color": "darkred",
        }
        
        # Add new slope if provided
        if shift_config.new_demand_slope is not None:
            new_demand_config["slope"] = shift_config.new_demand_slope
        
        # Apply the shift
        new_eq_x, new_eq_y = sd_plot.shift_demand(new_demand_config, label="New Demand")
        
        # Recalculate equilibrium with the new curve
        sd_plot.solve_equilibrium()
        
        # Save the new state
        os.makedirs(os.path.join(output_dir, "after"), exist_ok=True)
        new_output = sd_plot.create_explanation(os.path.join(output_dir, "after"))
        
        logger.info(f"New equilibrium: Q = {new_eq_x:.2f}, P = {new_eq_y:.2f}")
        
        return {
            "initial_html": initial_output["html"],
            "initial_image": initial_output["image"],
            "initial_text": initial_output["text"],
            "initial_audio": initial_output["audio"],
            "after_html": new_output["html"],
            "after_image": new_output["image"],
            "after_text": new_output["text"],
            "after_audio": new_output["audio"]
        }
    
    def _create_animated_demand_shift(self, config: AnimationConfig, output_dir: str) -> Dict[str, str]:
        """Create an animated demand shift visualization"""
        logger.info("Creating animated demand shift visualization")
        
        # Extract the demand shift config
        shift_config = config.demand_shift_config
        
        # Create a configuration for AnimatedSupplyDemandPlot
        plot_config = {
            "supply_config": {
                "slope": shift_config.supply_slope,
                "intercept": shift_config.supply_intercept,
                "color": shift_config.supply_color
            },
            "demand_config": {
                "slope": shift_config.demand_slope,
                "intercept": shift_config.demand_intercept,
                "color": shift_config.demand_color
            },
            "layout_config": {
                "title": config.title,
                "width": config.width,
                "height": config.height
            }
        }
        
        # Create the animated plot
        sd_plot = AnimatedSupplyDemandPlot(plot_config)
        
        # Prepare new demand config
        new_demand_config = {
            "intercept": shift_config.new_demand_intercept,
            "color": "darkred",
        }
        
        # Add new slope if provided
        if shift_config.new_demand_slope is not None:
            new_demand_config["slope"] = shift_config.new_demand_slope
        
        # Create the animation
        animated_dir = os.path.join(output_dir, "animated")
        os.makedirs(animated_dir, exist_ok=True)
        
        html_path = sd_plot.create_animated_shift(
            shift_type="demand",
            new_config=new_demand_config,
            frames=shift_config.frames,
            output_dir=animated_dir
        )
        
        audio_path = os.path.join(animated_dir, "animated_demand_shift.mp3")
        
        # Create synchronized HTML
        interactive_dir = os.path.join(output_dir, "interactive")
        os.makedirs(interactive_dir, exist_ok=True)
        
        # Import create_synced_html from create_synced_animation
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from create_synced_animation import create_synced_html
        
        synced_html_path = create_synced_html(
            shift_type="demand",
            html_path=html_path,
            audio_path=audio_path,
            output_dir=interactive_dir
        )
        
        return {
            "animated_html": html_path,
            "audio": audio_path,
            "interactive_html": synced_html_path
        }
    
    def _create_time_value(self, config: AnimationConfig, output_dir: str, output_format: str) -> Dict[str, str]:
        """Create a time value of money visualization"""
        logger.info("Creating time value of money visualization")
        
        # This would be implemented similar to the supply/demand visualizations
        # For now, return a placeholder
        return {
            "status": "placeholder",
            "message": "Time value visualization not yet implemented"
        }
    
    def _create_perpetuity(self, config: AnimationConfig, output_dir: str, output_format: str) -> Dict[str, str]:
        """Create a perpetuity visualization"""
        logger.info("Creating perpetuity visualization")
        
        # This would be implemented similar to the supply/demand visualizations
        # For now, return a placeholder
        return {
            "status": "placeholder",
            "message": "Perpetuity visualization not yet implemented"
        }
    
    def add_narration(self, render_result: Dict[str, Any], narration_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add narration to a visualization.
        
        Args:
            render_result: Information about the rendered visualization
            narration_data: Narration data (script and timing)
            
        Returns:
            Dict with info about the combined output
        """
        logger.info("Adding narration to visualization")
        
        # Get the output directory from the render result
        output_dir = render_result.get("output_dir", self.output_dir)
        
        # Get the full narration script
        script = narration_data.get("full_script", "")
        
        # Generate audio narration using gTTS
        audio_path = os.path.join(output_dir, "narration.mp3")
        
        try:
            tts = gTTS(text=script, lang='en', slow=False)
            tts.save(audio_path)
            logger.info(f"Narration audio saved to: {audio_path}")
        except Exception as e:
            logger.error(f"Error generating narration audio: {str(e)}")
            audio_path = None
        
        # If there's an HTML output, try to create a synchronized version
        if "interactive_html" in render_result:
            # Already synchronized, return as is
            return render_result
            
        elif "html" in render_result and audio_path:
            # We could synchronize the HTML and audio here
            # For now, just return both paths
            return {
                **render_result,
                "audio": audio_path,
                "narration_script": script
            }
        
        else:
            # Just return the render result with audio path added
            return {
                **render_result,
                "audio": audio_path,
                "narration_script": script
            }

# Main entry point for testing
if __name__ == "__main__":
    # Test the visualizer
    visualizer = PlotlyVisualizer()
    
    # Create a test configuration
    from config_models import AnimationConfig
    
    # Create a supply-demand config
    config_dict = {
        "visualization_type": "supply_demand",
        "title": "Test Supply-Demand Visualization",
        "supply_demand_config": {
            "supply_slope": 0.5,
            "supply_intercept": 2,
            "demand_slope": -0.5,
            "demand_intercept": 8,
            "x_range": [0, 10],
            "y_range": [0, 10],
            "show_equilibrium": True,
            "equilibrium_color": "green",
            "supply_color": "blue",
            "demand_color": "red"
        }
    }
    
    config = AnimationConfig.parse_obj(config_dict)
    
    # Create the visualization
    result = visualizer.create_visualization(config)
    
    # Print the result
    print(json.dumps(result, indent=2))
