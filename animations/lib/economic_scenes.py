"""
Economic scenes for supply and demand animations.

This module provides pre-configured scenes for common economic visualizations.
"""

from manim import (
    Scene, Axes, Text, Create, Write, FadeIn, FadeOut, Arrow, Line, Dot,
    ReplacementTransform, DOWN, RIGHT, UP, LEFT, ORIGIN, VGroup,
    BLUE, GREEN, RED, YELLOW, WHITE, PURPLE, ORANGE, GREY, GREY_D, GREEN_B, BLUE_D, RED_D,
    Rectangle, config
)
import numpy as np

from lib.supply_demand import SupplyDemandCurve

class SupplyDemandEquilibriumScene(Scene):
    def construct(self):
        """Construct the supply and demand equilibrium scene"""
        
        # Create a supply and demand curve with default configurations
        config = {
            "background_config": {
                "x_range": [0, 10, 1],
                "y_range": [0, 10, 1],
                "x_length": 6,
                "y_length": 6,
                "axis_config": {
                    "include_tip": False,
                    "include_numbers": True,
                    "numbers_to_exclude": []
                }
            },
            "supply_config": {
                "slope": 0.8,
                "intercept": 1,
                "color": BLUE,
                "stroke_width": 3
            },
            "demand_config": {
                "slope": -0.7,
                "intercept": 8,
                "color": RED,
                "stroke_width": 3
            },
            "labels_config": {
                "supply_label": "Supply",
                "demand_label": "Demand",
                "show_equilibrium_lines": True,
                "show_equilibrium_label": True,
                "equilibrium_color": GREEN,
                "eq_line_style": {
                    "stroke_width": 2,
                    "stroke_opacity": 0.8,
                    "stroke_color": GREEN_B
                }
            },
            "layout_config": {
                "graph_position": LEFT * 3.0,  # Position graph on left side
                "panel_width": 6.0,           # Width of the text panel
                "panel_height": 7.0,          # Height of the text panel
                "title_offset": UP * 3.2,      # Title offset from center
                "verify_layout": True,
            }
        }
        
        # Create supply and demand curve component
        sd_curve = SupplyDemandCurve(self, config=config)
        
        # Create and add all elements
        elements = sd_curve.add_all_elements(animate=True)
        
        # Wait before solving equilibrium
        self.wait(0.5)
        
        # Solve and show equilibrium
        sd_curve.solve_equilibrium(animate=True)
        
        # Wait a moment before concluding
        self.wait(1)
        
        # Add conclusion
        sd_curve.text_panel.add_text(
            "conclusions",
            "At equilibrium, the market clears with no shortage or surplus.",
            animate=True
        )
        
        # Wait before ending
        self.wait(1)

class ShiftingSupplyScene(Scene):
    def construct(self):
        """Construct a scene showing a shifting supply curve"""
        
        # Create a supply and demand curve with default configurations
        config = {
            "background_config": {
                "x_range": [0, 10, 1],
                "y_range": [0, 10, 1],
                "x_length": 6,
                "y_length": 6,
                "axis_config": {
                    "include_tip": False,
                    "include_numbers": True,
                    "numbers_to_exclude": []
                }
            },
            "supply_config": {
                "slope": 0.8,
                "intercept": 1,
                "color": BLUE,
                "stroke_width": 3
            },
            "demand_config": {
                "slope": -0.7,
                "intercept": 8,
                "color": RED,
                "stroke_width": 3
            },
            "labels_config": {
                "supply_label": "Supply (S₁)",
                "demand_label": "Demand",
                "show_equilibrium_lines": True,
                "show_equilibrium_label": True,
                "equilibrium_color": GREEN,
                "eq_line_style": {
                    "stroke_width": 2,
                    "stroke_opacity": 0.8,
                    "stroke_color": GREEN_B
                }
            },
            "layout_config": {
                "graph_position": LEFT * 3.0,  # Position graph on left side
                "panel_width": 6.0,           # Width of the text panel
                "panel_height": 7.0,          # Height of the text panel
                "title_offset": UP * 3.2,      # Title offset from center
                "verify_layout": True,
            }
        }
        
        # Create supply and demand curve component
        sd_curve = SupplyDemandCurve(self, config=config)
        
        # Create and add all elements
        elements = sd_curve.add_all_elements(animate=True)
        
        # Solve and show initial equilibrium
        self.wait(0.5)
        eq_x, eq_y = sd_curve.solve_equilibrium(animate=True)
        
        # Wait a bit before showing the shift
        self.wait(1)
        
        # Add explanation
        sd_curve.text_panel.add_text(
            "equations",
            "Initial Supply: P = 1 + 0.8Q\nDemand: P = 8 - 0.7Q",
            animate=True
        )
        
        # Wait before shifting the supply curve
        self.wait(1)
        
        # Defining the new supply curve parameters
        new_supply_config = {
            "slope": 0.8,  # Same slope
            "intercept": 3,  # Higher intercept
            "color": BLUE_D,
            "stroke_width": 3
        }
        
        # Add explanation for the shift
        sd_curve.text_panel.add_text(
            "conclusions",
            "Supply shifts up (decreases) due to:\n- Higher production costs\n- Resource constraints\n- New regulations",
            animate=True
        )
        
        # Wait before showing the shift
        self.wait(0.5)
        
        # Shift the supply curve with animation
        sd_curve.shift_supply_curve(
            new_supply_config,
            new_label="Supply (S₂)",
            shift_style="parallel",  # Options: parallel, pivot, or slope_change
            animate=True,
            run_time=2
        )
        
        # Wait a moment after the shift
        self.wait(0.5)
        
        # Solve for the new equilibrium
        sd_curve.text_panel.clear_section("equilibrium", animate=True, run_time=0.3)
        sd_curve.text_panel.clear_section("equations", animate=True, run_time=0.3)
        sd_curve.text_panel.clear_section("calculations", animate=True, run_time=0.3)
        
        # Update equation display
        sd_curve.text_panel.add_text(
            "equations",
            "New Supply: P = 3 + 0.8Q\nDemand: P = 8 - 0.7Q",
            animate=True
        )
        
        # Wait before solving for new equilibrium
        self.wait(0.5)
        
        # Calculate the new equilibrium
        new_eq_x, new_eq_y = sd_curve.solve_equilibrium(animate=True)
        
        # Wait a moment before adding the impact
        self.wait(1)
        
        # Add impact analysis
        sd_curve.text_panel.clear_section("conclusions", animate=True, run_time=0.5)
        sd_curve.text_panel.add_text(
            "conclusions",
            f"Impact:\n- ↓ Quantity: {eq_x:.1f} → {new_eq_x:.1f}\n- ↑ Price: {eq_y:.1f} → {new_eq_y:.1f}",
            animate=True,
            color=YELLOW
        )
        
        # Wait before ending
        self.wait(2)

class ShiftingDemandScene(Scene):
    def construct(self):
        """Construct a scene showing a shifting demand curve"""
        
        # Create a supply and demand curve with default configurations
        config = {
            "background_config": {
                "x_range": [0, 10, 1],
                "y_range": [0, 10, 1],
                "x_length": 6,
                "y_length": 6,
                "axis_config": {
                    "include_tip": False,
                    "include_numbers": True,
                    "numbers_to_exclude": []
                }
            },
            "supply_config": {
                "slope": 0.8,
                "intercept": 1,
                "color": BLUE,
                "stroke_width": 3
            },
            "demand_config": {
                "slope": -0.7,
                "intercept": 8,
                "color": RED,
                "stroke_width": 3
            },
            "labels_config": {
                "supply_label": "Supply",
                "demand_label": "Demand (D₁)",
                "show_equilibrium_lines": True,
                "show_equilibrium_label": True,
                "equilibrium_color": GREEN,
                "eq_line_style": {
                    "stroke_width": 2,
                    "stroke_opacity": 0.8,
                    "stroke_color": GREEN_B
                }
            },
            "layout_config": {
                "graph_position": LEFT * 3.0,  # Position graph on left side
                "panel_width": 6.0,           # Width of the text panel
                "panel_height": 7.0,          # Height of the text panel
                "title_offset": UP * 3.2,      # Title offset from center
                "verify_layout": True,
            }
        }
        
        # Create supply and demand curve component
        sd_curve = SupplyDemandCurve(self, config=config)
        
        # Create and add all elements
        elements = sd_curve.add_all_elements(animate=True)
        
        # Solve and show initial equilibrium
        self.wait(0.5)
        eq_x, eq_y = sd_curve.solve_equilibrium(animate=True)
        
        # Wait a bit before showing the shift
        self.wait(1)
        
        # Add explanation
        sd_curve.text_panel.add_text(
            "equations",
            "Supply: P = 1 + 0.8Q\nInitial Demand: P = 8 - 0.7Q",
            animate=True
        )
        
        # Wait before shifting the demand curve
        self.wait(1)
        
        # Defining the new demand curve parameters
        new_demand_config = {
            "slope": -0.7,  # Same slope
            "intercept": 10,  # Higher intercept (increased demand)
            "color": RED_D,
            "stroke_width": 3
        }
        
        # Add explanation for the shift
        sd_curve.text_panel.add_text(
            "conclusions",
            "Demand shifts right (increases) due to:\n- Increased income\n- Changes in preferences\n- Population growth",
            animate=True
        )
        
        # Wait before showing the shift
        self.wait(0.5)
        
        # Shift the demand curve with animation
        sd_curve.shift_demand_curve(
            new_demand_config,
            new_label="Demand (D₂)",
            animate=True,
            run_time=2
        )
        
        # Wait a moment after the shift
        self.wait(0.5)
        
        # Solve for the new equilibrium
        sd_curve.text_panel.clear_section("equilibrium", animate=True, run_time=0.3)
        sd_curve.text_panel.clear_section("equations", animate=True, run_time=0.3)
        sd_curve.text_panel.clear_section("calculations", animate=True, run_time=0.3)
        
        # Update equation display
        sd_curve.text_panel.add_text(
            "equations",
            "Supply: P = 1 + 0.8Q\nNew Demand: P = 10 - 0.7Q",
            animate=True
        )
        
        # Wait before solving for new equilibrium
        self.wait(0.5)
        
        # Calculate the new equilibrium
        new_eq_x, new_eq_y = sd_curve.solve_equilibrium(animate=True)
        
        # Wait a moment before adding the impact
        self.wait(1)
        
        # Add impact analysis
        sd_curve.text_panel.clear_section("conclusions", animate=True, run_time=0.5)
        sd_curve.text_panel.add_text(
            "conclusions",
            f"Impact:\n- ↑ Quantity: {eq_x:.1f} → {new_eq_x:.1f}\n- ↑ Price: {eq_y:.1f} → {new_eq_y:.1f}",
            animate=True,
            color=YELLOW
        )
        
        # Wait before ending
        self.wait(2) 