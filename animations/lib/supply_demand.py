"""
Supply and Demand curve component for Manim animations.
This module provides reusable components for creating and manipulating
supply and demand curve animations.
"""

from manim import (
    Scene, Axes, Text, Create, Write, FadeIn, FadeOut, Arrow, Line, Dot,
    ReplacementTransform, DOWN, RIGHT, UP, LEFT, ORIGIN, VGroup,
    BLUE, GREEN, RED, YELLOW, WHITE, PURPLE, ORANGE, GREY, GREY_D, GREEN_B, BLUE_D, RED_D,
    Rectangle, Tex, config, DashedLine, Animation
)
import numpy as np

class PositionableComponent:
    """
    Base class that provides an anchor-based positioning system.
    
    This allows elements to be positioned relative to predefined anchors,
    making layout more consistent and adaptable.
    """
    
    def __init__(self, scene, config=None):
        self.scene = scene
        self.anchors = {}
        self.positioned_elements = {}
        self.padding = 0.25  # Default padding between elements
        
        # Default configuration
        self.default_config = {
            "x_range": [0, 10, 1],
            "y_range": [0, 10, 1],
            "axes_config": {
                "color": BLUE,
                "x_length": 5,  # Slightly smaller to accommodate text panel
                "y_length": 5,
                "axis_config": {"include_tip": True},
            },
            "supply_config": {
                "color": GREEN,
                "slope": 0.5,
                "intercept": 2,
                "label_text": "Supply",
                "stroke_width": 3,
            },
            "demand_config": {
                "color": RED,
                "slope": -0.5,
                "intercept": 8,
                "label_text": "Demand",
                "stroke_width": 3,
            },
            "labels_config": {
                "x_label": "Quantity",
                "y_label": "Price",
                "title": None,
                "show_equilibrium": True,
                "equilibrium_color": YELLOW,
            },
            "layout_config": {
                "graph_position": LEFT * 3.0,  # Position graph on left side
                "panel_width": 6.0,           # Panel width
                "panel_height": 7.0,          # Panel height
                "title_offset": UP * 3.2,     # Offset from center for title
                "verify_layout": True,
            },
            "animation_config": {
                "axes_creation_time": 1.0,
                "labels_creation_time": 0.8,
                "title_creation_time": 1.0,
                "curve_creation_time": 1.2,
                "curve_label_creation_time": 0.8,
                "equilibrium_creation_time": 1.0,
                "wait_time_after_step": 0.5,
            }
        }
        
        # Set default colors with more contrast and improved visibility
        self.colors = {
            "axes": GREY_D,
            "supply": BLUE_D,  # Darker blue for better visibility
            "demand": RED_D,  # Darker red for better visibility
            "equilibrium": GREEN_B,
            "price": YELLOW,
            "quantity": PURPLE
        }
        
        # Update colors if provided in config
        if "colors" in self.config:
            self.colors.update(self.config["colors"])
        
        # Create the text panel
        self.text_panel = TextPanel(
            scene=self.scene, 
            position=RIGHT * 3.0,
            width=self.config["layout_config"]["panel_width"],
            height=self.config["layout_config"]["panel_height"]
        )
        
        # Create the axes and curves - initial elements only
        self._create_initial_elements()
        
        # Setup anchors for positioning
        self._setup_anchors()
        
        # Create the remaining elements
        self._create_remaining_elements()
        
        # Position elements
        self._position_elements()
        
        # Verify layout if configured to do so
        if self.config["layout_config"]["verify_layout"]:
            self.verify_layout()
        
    def _update_nested_dict(self, d, u):
        """Helper method to update nested dictionaries"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def _create_initial_elements(self):
        """Create the basic elements needed to establish anchors"""
        # Set the base position for the graph (left side)
        graph_position = self.config["layout_config"]["graph_position"]
        
        # Create axes
        self.axes = Axes(
            x_range=self.config["background_config"]["x_range"],
            y_range=self.config["background_config"]["y_range"],
            **self.config["axes_config"]
        ).move_to(graph_position)
        
        # Supply function: P = slope*Q + intercept
        supply_slope = self.config["supply_config"]["slope"]
        supply_intercept = self.config["supply_config"]["intercept"]
        supply_color = self.config["supply_config"]["color"]
        supply_stroke_width = self.config["supply_config"]["stroke_width"]
        
        self.supply_function = lambda x: supply_slope * x + supply_intercept
        self.supply_curve = self.axes.plot(
            self.supply_function, 
            color=supply_color,
            stroke_width=supply_stroke_width
        )
        
        # Create supply label
        supply_label_text = self.config["labels_config"].get("supply_label", "Supply")
        self.supply_label = Text(supply_label_text, color=supply_color).scale(0.7)
        # Position will be set in _position_elements
        
        # Demand function: P = slope*Q + intercept
        demand_slope = self.config["demand_config"]["slope"]
        demand_intercept = self.config["demand_config"]["intercept"]
        demand_color = self.config["demand_config"]["color"]
        demand_stroke_width = self.config["demand_config"]["stroke_width"]
        
        self.demand_function = lambda x: demand_slope * x + demand_intercept
        self.demand_curve = self.axes.plot(
            self.demand_function, 
            color=demand_color,
            stroke_width=demand_stroke_width
        )
        
        # Create demand label
        demand_label_text = self.config["labels_config"].get("demand_label", "Demand")
        self.demand_label = Text(demand_label_text, color=demand_color).scale(0.7)
        # Position will be set in _position_elements
        
        # Calculate equilibrium
        # Solve: supply_slope * x + supply_intercept = demand_slope * x + demand_intercept
        # (supply_slope - demand_slope) * x = demand_intercept - supply_intercept
        # x = (demand_intercept - supply_intercept) / (supply_slope - demand_slope)
        
        self.eq_x = (demand_intercept - supply_intercept) / (supply_slope - demand_slope)
        self.eq_y = self.supply_function(self.eq_x)
    
    def _setup_anchors(self):
        """Setup the anchor points for positioning elements"""
        # Basic graph anchors
        self.register_anchor("origin", lambda: self.axes.c2p(0, 0))
        self.register_anchor("graph_center", lambda: self.axes.get_center())
        self.register_anchor("graph_top", lambda: self.axes.get_top())
        self.register_anchor("graph_bottom", lambda: self.axes.get_bottom())
        self.register_anchor("graph_left", lambda: self.axes.get_left())
        self.register_anchor("graph_right", lambda: self.axes.get_right())
        
        # Axis endpoints
        self.register_anchor("x_axis_end", lambda: self.axes.x_axis.get_end())
        self.register_anchor("y_axis_end", lambda: self.axes.y_axis.get_end())
        
        # Equilibrium point
        self.register_anchor("equilibrium", lambda: self.axes.c2p(self.eq_x, self.eq_y))
        self.register_anchor("eq_x_axis", lambda: self.axes.c2p(self.eq_x, 0))
        self.register_anchor("eq_y_axis", lambda: self.axes.c2p(0, self.eq_y))
        
        # Title anchor (center top)
        title_offset = self.config["layout_config"]["title_offset"]
        self.register_anchor("title_position", lambda: ORIGIN + title_offset)
    
    def _create_remaining_elements(self):
        """Create the remaining elements that depend on anchors"""
        # Create axis labels
        x_label_text = self.config["labels_config"].get("x_label", "Quantity")
        y_label_text = self.config["labels_config"].get("y_label", "Price")
        label_color = self.config["labels_config"].get("axis_label_color", GREY_D)
        
        self.x_label = Text(x_label_text, color=label_color).scale(0.7)
        self.y_label = Text(y_label_text, color=label_color).scale(0.7)
        
        # Create title if configured
        if self.config["labels_config"]["title"]:
            self.title = Text(self.config["labels_config"]["title"]).scale(0.9)
        else:
            self.title = None
        
        # Create equilibrium elements
        self.eq_point = Dot(
            self.axes.c2p(self.eq_x, self.eq_y),
            color=self.config["labels_config"]["equilibrium_color"]
        )
        
        # Vertical line to x-axis
        self.eq_x_line = DashedLine(
            self.axes.c2p(self.eq_x, 0),
            self.axes.c2p(self.eq_x, self.eq_y),
            color=self.config["labels_config"]["equilibrium_color"],
            stroke_width=2,
            stroke_opacity=0.7
        )
        
        # Horizontal line to y-axis
        self.eq_y_line = DashedLine(
            self.axes.c2p(0, self.eq_y),
            self.axes.c2p(self.eq_x, self.eq_y),
            color=self.config["labels_config"]["equilibrium_color"],
            stroke_width=2,
            stroke_opacity=0.7
        )
        
        # Equilibrium labels
        self.eq_x_label = Text(f"Q = {self.eq_x:.1f}").scale(0.5)
        self.eq_y_label = Text(f"P = {self.eq_y:.1f}").scale(0.5)
        
        # Group equilibrium elements
        self.equilibrium_elements = [
            self.eq_point, 
            self.eq_x_line, 
            self.eq_y_line,
            self.eq_x_label,
            self.eq_y_label
        ]
    
    def _position_elements(self):
        """Position elements using the anchor system"""
        # Position axis labels
        self.position_at(self.x_label, "x_axis_end", DOWN, 0.3)
        self.position_at(self.y_label, "y_axis_end", LEFT, 0.3)
        
        # Position title if it exists (centered at top)
        if self.title:
            self.position_at(self.title, "title_position")
        
        # Position equilibrium labels
        self.position_at(self.eq_x_label, "eq_x_axis", DOWN, 0.2)
        self.position_at(self.eq_y_label, "eq_y_axis", LEFT, 0.2)
        
        # Position supply and demand curve labels
        # For supply, position near the top-right of the curve
        x_max = self.config["background_config"]["x_range"][1]
        supply_x_pos = x_max * 0.8
        supply_y_pos = self.supply_function(supply_x_pos)
        self.supply_label.move_to(self.axes.c2p(supply_x_pos, supply_y_pos) + UP * 0.3 + RIGHT * 0.3)
        
        # For demand, position near the top-right of the curve
        demand_x_pos = x_max * 0.8
        demand_y_pos = self.demand_function(demand_x_pos)
        self.demand_label.move_to(self.axes.c2p(demand_x_pos, demand_y_pos) + DOWN * 0.3 + RIGHT * 0.3)
    
    def register_anchor(self, name, position_func):
        """
        Register an anchor point with a callable that returns its position.
        
        Parameters:
        -----------
        name : str
            The name of the anchor
        position_func : callable
            A function that returns the position (as a np.array or manim.Point)
        """
        self.anchors[name] = position_func
        
    def get_anchor(self, name):
        """
        Get the current position of an anchor.
        
        Parameters:
        -----------
        name : str
            The name of the anchor
            
        Returns:
        --------
        position : np.array or None
            The position of the anchor, or None if not found
        """
        if name in self.anchors:
            return self.anchors[name]()
        return None
    
    def position_at(self, element, anchor_name, direction=None, buff=None):
        """
        Position an element at a specific anchor with a direction and buffer.
        
        Parameters:
        -----------
        element : Mobject
            The element to position
        anchor_name : str
            The name of the anchor to position relative to
        direction : np.array, optional
            The direction from the anchor (e.g., UP, DOWN, LEFT, RIGHT)
        buff : float, optional
            The buffer distance from the anchor
        """
        if anchor_name not in self.anchors:
            print(f"Warning: Anchor '{anchor_name}' not found")
            return element
            
        anchor_pos = self.get_anchor(anchor_name)
        if buff is None:
            buff = self.padding
            
        if direction is None:
            element.move_to(anchor_pos)
        else:
            element.next_to(anchor_pos, direction, buff=buff)
            
        # Store the positioning info for later adjustment
        self.positioned_elements[element] = {
            "anchor": anchor_name,
            "direction": direction,
            "buff": buff
        }
        
        return element
    
    def update_positions(self):
        """Update all positioned elements if anchors have moved"""
        for element, info in self.positioned_elements.items():
            self.position_at(element, info["anchor"], info["direction"], info["buff"])
    
    def check_for_collisions(self):
        """
        Check for potential collisions between elements.
        
        Returns:
        --------
        collisions : list of tuples
            List of (element1, element2) pairs that may be colliding
        """
        collisions = []
        elements = list(self.positioned_elements.keys())
        
        for i in range(len(elements)):
            for j in range(i+1, len(elements)):
                elem1 = elements[i]
                elem2 = elements[j]
                
                # Simple bounding box collision detection
                if (elem1.get_right()[0] > elem2.get_left()[0] and
                    elem1.get_left()[0] < elem2.get_right()[0] and
                    elem1.get_top()[1] > elem2.get_bottom()[1] and
                    elem1.get_bottom()[1] < elem2.get_top()[1]):
                    collisions.append((elem1, elem2))
                    
        return collisions
    
    def verify_layout(self, verbose=True):
        """
        Verify the layout has no obvious issues.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print detailed information
            
        Returns:
        --------
        is_valid : bool
            True if the layout appears valid
        """
        # Check for elements outside scene bounds
        out_of_bounds = []
        for elem in self.positioned_elements:
            # Using standard Manim scene dimensions (-7.5 to 7.5 horizontally, -4 to 4 vertically)
            if (elem.get_left()[0] < -7.5 or elem.get_right()[0] > 7.5 or
                elem.get_bottom()[1] < -4 or elem.get_top()[1] > 4):
                out_of_bounds.append(elem)
        
        # Check for collisions
        collisions = self.check_for_collisions()
        
        if verbose:
            if out_of_bounds:
                print(f"Warning: {len(out_of_bounds)} elements may be outside scene bounds")
                
            if collisions:
                print(f"Warning: {len(collisions)} potential collisions detected")
                
        return len(out_of_bounds) == 0 and len(collisions) == 0

class TextPanel:
    """
    A text panel component for displaying information and calculations.
    
    The panel is divided into sections, each with its own title and content.
    Sections can be dynamically updated with new content during animations.
    """
    
    def __init__(self, scene, position=RIGHT * 3, width=6, height=7, sections=None):
        self.scene = scene
        self.position = position
        self.width = width
        self.height = height
        
        # Default sections if none provided
        if sections is None:
            sections = {
                "title": {"title": "Supply and Demand Analysis", "height": 0.8},
                "equations": {"title": "Equations", "height": 1.5},
                "calculations": {"title": "Calculation Steps", "height": 3.0},
                "equilibrium": {"title": "Equilibrium", "height": 1.0},
                "conclusions": {"title": "Conclusions", "height": 2.0}
            }
        
        self.sections = sections
        self.content = {}  # Store content for each section
        self.elements = {}  # Store all visual elements
        
        # Set up the panel and sections
        self._setup_panel()

    def _setup_panel(self):
        # Create the main panel rectangle with a light gray fill
        self.panel = Rectangle(
            width=self.width,
            height=self.height,
            fill_color=GREY,
            fill_opacity=0.1,
            stroke_color=GREY_D,
            stroke_opacity=0.8,
            stroke_width=2
        ).move_to(self.position)
        
        # Store all elements in a group for easier manipulation
        self.elements["panel"] = self.panel
        self.elements["sections"] = {}
        self.elements["content"] = {}
        
        # Current vertical position (start from top)
        current_y = self.height / 2
        
        # Create section backgrounds and titles
        for section_id, section_info in self.sections.items():
            # Section title
            title_text = section_info.get("title", section_id.capitalize())
            title_height = 0.6  # Fixed height for title areas
            section_height = section_info.get("height", 1.0)
            
            # Section background (slightly darker than main panel)
            section_bg = Rectangle(
                width=self.width - 0.1,
                height=title_height,
                fill_color=GREY,
                fill_opacity=0.3,
                stroke_width=1,
                stroke_opacity=0.5,
                stroke_color=GREY_D
            ).move_to(self.panel.get_top() + DOWN * (current_y - title_height/2))
            
            # Section title text
            title = Text(
                title_text,
                color=WHITE,
                font_size=24
            ).move_to(section_bg.get_center())
            
            # Content area (no visible background, just tracking the area)
            content_height = section_height - title_height
            content_area = Rectangle(
                width=self.width - 0.2,
                height=content_height,
                fill_opacity=0,
                stroke_opacity=0
            ).move_to(self.panel.get_top() + DOWN * (current_y + content_height/2 + title_height/2))
            
            # Store created elements
            self.elements["sections"][section_id] = {
                "background": section_bg,
                "title": title,
                "content_area": content_area
            }
            
            # Initialize content tracking for this section
            self.content[section_id] = []
            self.elements["content"][section_id] = []
            
            # Update current position
            current_y += section_height
            
    def get_all_elements(self):
        """Return all panel elements as a flat list for adding to scene"""
        all_elements = [self.elements["panel"]]
        
        # Add all section backgrounds and titles
        for section_id, section_elements in self.elements["sections"].items():
            all_elements.append(section_elements["background"])
            all_elements.append(section_elements["title"])
        
        # Add all content elements
        for section_id, content_elements in self.elements["content"].items():
            all_elements.extend(content_elements)
            
        return all_elements
            
    def _wrap_text_if_needed(self, text_str, color=WHITE, scale=0.6, max_width=5.0):
        """Helper to wrap text if it would exceed the maximum width"""
        words = text_str.split()
        lines = []
        current_line = []
        
        # Simple word wrapping - could be improved with actual text measurement
        for word in words:
            test_line = ' '.join(current_line + [word])
            if len(test_line) * scale * 0.15 > max_width:  # Rough estimate
                if current_line:  # Avoid empty lines
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word itself is too long, just add it anyway
                    current_line.append(word)
            else:
                current_line.append(word)
                
        # Add the last line if there is one
        if current_line:
            lines.append(' '.join(current_line))
            
        # Join lines with newlines
        wrapped_text = '\n'.join(lines)
        
        # Create text object
        return Text(wrapped_text, color=color).scale(scale)

    def get_section_space(self, section):
        """Get available space in the given section"""
        if section not in self.sections:
            return 0
        return self.elements["sections"][section]["content_area"].height

    def get_max_items_in_section(self, section, item_height=0.4):
        """Calculate maximum number of items that can fit in a section"""
        space = self.get_section_space(section)
        return int(space / item_height)

    def add_text(self, section, text, color=WHITE, scale=0.6, animate=False, run_time=1):
        """Add text to a section, optionally with animation"""
        if section not in self.sections:
            print(f"Warning: Section '{section}' does not exist")
            return None
            
        # Wrap text if needed
        text_obj = self._wrap_text_if_needed(text, color, scale, self.width - 0.5)
        
        # Get content area
        content_area = self.elements["sections"][section]["content_area"]
        
        # Calculate vertical position based on existing content
        existing_content = self.elements["content"][section]
        y_offset = 0.2  # Start with a small top margin
        
        # Adjust for previous content (includes spacing between items)
        for item in existing_content:
            y_offset += item.height * scale + 0.25  # 0.25 spacing between items
        
        # Position text
        text_obj.move_to(
            content_area.get_top() + 
            DOWN * (y_offset + text_obj.height * scale / 2)
        ).align_to(content_area, LEFT).shift(RIGHT * 0.2)  # Small left margin
        
        # Store the content
        self.content[section].append(text)
        self.elements["content"][section].append(text_obj)
        
        # Add to scene with or without animation
        if animate:
            self.scene.play(Write(text_obj), run_time=run_time)
        else:
            self.scene.add(text_obj)
            
        return text_obj

    def clear_section(self, section, animate=False, run_time=1):
        """Clear all content from a section"""
        if section not in self.sections or not self.elements["content"][section]:
            return
            
        content_to_remove = self.elements["content"][section]
        
        if animate:
            self.scene.play(
                *[FadeOut(item) for item in content_to_remove],
                run_time=run_time
            )
        else:
            for item in content_to_remove:
                self.scene.remove(item)
                
        # Clear stored content
        self.content[section] = []
        self.elements["content"][section] = []

    def replace_text(self, section, text, color=WHITE, scale=0.6, animate=False, run_time=1):
        """Replace all text in a section with new text"""
        self.clear_section(section, animate=animate, run_time=run_time/2 if animate else 0)
        return self.add_text(section, text, color, scale, animate, run_time/2 if animate else run_time)
        
    def add_calculation_step(self, step_text, color=WHITE, scale=0.55, animate=True, run_time=1, fade_previous=False):
        """Add a calculation step to the calculations section"""
        # Keep track of steps
        steps = []
        
        # If we're fading the previous calculation, do that first
        if fade_previous and self.elements["content"]["calculations"]:
            prev_calculations = self.elements["content"]["calculations"]
            if animate:
                for item in prev_calculations:
                    self.scene.play(item.animate.set_opacity(0.3), run_time=run_time/2)
            else:
                for item in prev_calculations:
                    item.set_opacity(0.3)
        
        # Add the new calculation step
        return self.add_text("calculations", step_text, color, scale, animate, run_time)
    
    def introduce(self, run_time=1):
        """Introduce the panel with a simple animation"""
        # Create a list of all elements to introduce
        elements = self.get_all_elements()
        
        # Introduce the panel first
        self.scene.play(Create(self.panel), run_time=run_time/2)
        
        # Then introduce section backgrounds and titles
        section_elements = []
        for section_id, section in self.elements["sections"].items():
            section_elements.append(section["background"])
            section_elements.append(section["title"])
            
        self.scene.play(
            *[FadeIn(element) for element in section_elements],
            run_time=run_time/2
        )

class SupplyDemandCurve(PositionableComponent):
    """
    A reusable supply and demand curve component.
    
    This component creates and manages a complete supply and demand diagram,
    including axes, labels, curves, and equilibrium calculations.
    """
    
    def __init__(self, scene, config=None):
        # Set up default config first
        self.default_config = {
            "x_range": [0, 10, 1],
            "y_range": [0, 10, 1],
            "axes_config": {
                "color": GREY_D,  # Darker grey for better visibility
                "x_length": 5,
                "y_length": 5,
                "axis_config": {"include_tip": True},
            },
            "supply_config": {
                "color": BLUE_D,  # Darker blue for better visibility
                "slope": 0.5,
                "intercept": 2,
                "label_text": "Supply",
                "stroke_width": 3,
            },
            "demand_config": {
                "color": RED_D,  # Darker red for better visibility
                "slope": -0.5,
                "intercept": 8,
                "label_text": "Demand",
                "stroke_width": 3,
            },
            "labels_config": {
                "x_label": "Quantity",
                "y_label": "Price",
                "title": None,
                "show_equilibrium": True,
                "equilibrium_color": GREEN_B,  # Brighter green for visibility
            },
            "layout_config": {
                "graph_position": LEFT * 3.0,
                "panel_width": 6.0,
                "panel_height": 7.0,
                "title_offset": UP * 3.2,
                "verify_layout": True,
            },
            "animation_config": {
                "axes_creation_time": 1.0,
                "labels_creation_time": 0.8,
                "title_creation_time": 1.0,
                "curve_creation_time": 1.2,
                "curve_label_creation_time": 0.8,
                "equilibrium_creation_time": 1.0,
                "wait_time_after_step": 0.5,
            }
        }
        
        # Initialize config
        self.config = self.default_config.copy()
        if config:
            self._update_nested_dict(self.config, config)
        
        # Set scene
        self.scene = scene
        
        # Set up colors
        self.colors = {
            "axes": GREY_D,
            "supply": BLUE_D,
            "demand": RED_D,
            "equilibrium": GREEN_B,
            "price": YELLOW,
            "quantity": PURPLE
        }
        
        # Setup anchors and positioned elements from parent class
        self.anchors = {}
        self.positioned_elements = {}
        self.padding = 0.25
        
        # Create the text panel
        self.text_panel = TextPanel(
            scene=self.scene, 
            position=RIGHT * 3.0,
            width=self.config["layout_config"]["panel_width"],
            height=self.config["layout_config"]["panel_height"]
        )
        
        # Create elements
        self._create_initial_elements()
        self._setup_anchors()
        self._create_remaining_elements()
        self._position_elements()
        
        # Verify layout if configured to do so
        if self.config["layout_config"]["verify_layout"]:
            self.verify_layout()

    def _create_initial_elements(self):
        """Create the basic elements needed to establish anchors"""
        # Set the base position for the graph (left side)
        graph_position = self.config["layout_config"]["graph_position"]
        
        # Create axes
        self.axes = Axes(
            x_range=self.config["background_config"]["x_range"],
            y_range=self.config["background_config"]["y_range"],
            **self.config["axes_config"]
        ).move_to(graph_position)
        
        # Supply function: P = slope*Q + intercept
        supply_slope = self.config["supply_config"]["slope"]
        supply_intercept = self.config["supply_config"]["intercept"]
        supply_color = self.config["supply_config"]["color"]
        supply_stroke_width = self.config["supply_config"]["stroke_width"]
        
        self.supply_function = lambda x: supply_slope * x + supply_intercept
        self.supply_curve = self.axes.plot(
            self.supply_function, 
            color=supply_color,
            stroke_width=supply_stroke_width
        )
        
        # Create supply label
        supply_label_text = self.config["labels_config"].get("supply_label", "Supply")
        self.supply_label = Text(supply_label_text, color=supply_color).scale(0.7)
        # Position will be set in _position_elements
        
        # Demand function: P = slope*Q + intercept
        demand_slope = self.config["demand_config"]["slope"]
        demand_intercept = self.config["demand_config"]["intercept"]
        demand_color = self.config["demand_config"]["color"]
        demand_stroke_width = self.config["demand_config"]["stroke_width"]
        
        self.demand_function = lambda x: demand_slope * x + demand_intercept
        self.demand_curve = self.axes.plot(
            self.demand_function, 
            color=demand_color,
            stroke_width=demand_stroke_width
        )
        
        # Create demand label
        demand_label_text = self.config["labels_config"].get("demand_label", "Demand")
        self.demand_label = Text(demand_label_text, color=demand_color).scale(0.7)
        # Position will be set in _position_elements
        
        # Calculate equilibrium
        # Solve: supply_slope * x + supply_intercept = demand_slope * x + demand_intercept
        # (supply_slope - demand_slope) * x = demand_intercept - supply_intercept
        # x = (demand_intercept - supply_intercept) / (supply_slope - demand_slope)
        
        self.eq_x = (demand_intercept - supply_intercept) / (supply_slope - demand_slope)
        self.eq_y = self.supply_function(self.eq_x)

    def _create_remaining_elements(self):
        """Create the remaining elements: curves, labels, etc."""
        # Create axis labels
        x_label_text = self.config["labels_config"].get("x_label", "Quantity")
        y_label_text = self.config["labels_config"].get("y_label", "Price")
        label_color = self.config["labels_config"].get("axis_label_color", GREY_D)
        
        self.x_label = Text(x_label_text, color=label_color).scale(0.7)
        self.y_label = Text(y_label_text, color=label_color).scale(0.7)
        
        # Create title if configured
        if self.config["labels_config"]["title"]:
            self.title = Text(self.config["labels_config"]["title"]).scale(0.9)
        else:
            self.title = None
        
        # Create equilibrium elements
        self.eq_point = Dot(
            self.axes.c2p(self.eq_x, self.eq_y),
            color=self.config["labels_config"]["equilibrium_color"]
        )
        
        # Vertical line to x-axis
        self.eq_x_line = DashedLine(
            self.axes.c2p(self.eq_x, 0),
            self.axes.c2p(self.eq_x, self.eq_y),
            color=self.config["labels_config"]["equilibrium_color"],
            stroke_width=2,
            stroke_opacity=0.7
        )
        
        # Horizontal line to y-axis
        self.eq_y_line = DashedLine(
            self.axes.c2p(0, self.eq_y),
            self.axes.c2p(self.eq_x, self.eq_y),
            color=self.config["labels_config"]["equilibrium_color"],
            stroke_width=2,
            stroke_opacity=0.7
        )
        
        # Equilibrium labels
        self.eq_x_label = Text(f"Q = {self.eq_x:.1f}").scale(0.5)
        self.eq_y_label = Text(f"P = {self.eq_y:.1f}").scale(0.5)
        
        # Group equilibrium elements
        self.equilibrium_elements = [
            self.eq_point, 
            self.eq_x_line, 
            self.eq_y_line,
            self.eq_x_label,
            self.eq_y_label
        ]

    def _position_elements(self):
        """Position elements using the anchor system"""
        # Position axis labels
        self.position_at(self.x_label, "x_axis_end", DOWN, 0.3)
        self.position_at(self.y_label, "y_axis_end", LEFT, 0.3)
        
        # Position title if it exists (centered at top)
        if self.title:
            self.position_at(self.title, "title_position")
        
        # Position equilibrium labels
        self.position_at(self.eq_x_label, "eq_x_axis", DOWN, 0.2)
        self.position_at(self.eq_y_label, "eq_y_axis", LEFT, 0.2)
        
        # Position supply and demand curve labels
        # For supply, position near the top-right of the curve
        x_max = self.config["background_config"]["x_range"][1]
        supply_x_pos = x_max * 0.8
        supply_y_pos = self.supply_function(supply_x_pos)
        self.supply_label.move_to(self.axes.c2p(supply_x_pos, supply_y_pos) + UP * 0.3 + RIGHT * 0.3)
        
        # For demand, position near the top-right of the curve
        demand_x_pos = x_max * 0.8
        demand_y_pos = self.demand_function(demand_x_pos)
        self.demand_label.move_to(self.axes.c2p(demand_x_pos, demand_y_pos) + DOWN * 0.3 + RIGHT * 0.3)
    
    def add_calculation_step(self, step_text, color=WHITE, scale=0.55, animate=True, run_time=1, fade_previous=False):
        """Add a calculation step to the text panel"""
        if not hasattr(self, "text_panel"):
            return None
            
        return self.text_panel.add_calculation_step(
            step_text, 
            color=color, 
            scale=scale, 
            animate=animate, 
            run_time=run_time,
            fade_previous=fade_previous
        )

    def solve_equilibrium(self, animate=True):
        """Calculate and display the steps to solve for equilibrium"""
        # Get config values
        supply_slope = self.config["supply_config"]["slope"]
        supply_intercept = self.config["supply_config"]["intercept"]
        demand_slope = self.config["demand_config"]["slope"]
        demand_intercept = self.config["demand_config"]["intercept"]
        
        # Add explanation text if animating
        if animate:
            self.text_panel.clear_section("calculations", animate=False)
            self.text_panel.add_text(
                "calculations",
                "Finding Equilibrium:",
                color=BLUE,
                animate=True,
                run_time=0.8
            )
        
        # Step 1: Set up the equations
        self.add_calculation_step(
            f"Supply: P = {supply_intercept} + {supply_slope}Q",
            color=self.colors["supply"],
            animate=animate,
            run_time=0.8 if animate else 0
        )
        
        self.add_calculation_step(
            f"Demand: P = {demand_intercept} {demand_slope:+}Q",
            color=self.colors["demand"],
            animate=animate,
            run_time=0.8 if animate else 0
        )
        
        # Step 2: Set equal and solve
        equation_text = f"{supply_intercept} + {supply_slope}Q = {demand_intercept} {demand_slope:+}Q"
        self.add_calculation_step(
            equation_text,
            color=YELLOW,
            animate=animate,
            run_time=0.8 if animate else 0
        )
        
        # Step 3: Rearrange to solve for Q
        combined_slope = supply_slope - demand_slope
        combined_intercept = demand_intercept - supply_intercept
        
        rearrange_text = f"{supply_slope}Q - ({demand_slope})Q = {demand_intercept} - {supply_intercept}"
        self.add_calculation_step(
            rearrange_text,
            color=YELLOW,
            animate=animate,
            run_time=0.8 if animate else 0
        )
        
        # Step 4: Simplify
        simplify_text = f"{combined_slope}Q = {combined_intercept}"
        self.add_calculation_step(
            simplify_text,
            color=YELLOW,
            animate=animate,
            run_time=0.8 if animate else 0
        )
        
        # Step 5: Solve for Q
        q_text = f"Q = {combined_intercept}/{combined_slope} = {self.eq_x:.1f}"
        self.add_calculation_step(
            q_text,
            color=YELLOW,
            animate=animate,
            run_time=0.8 if animate else 0
        )
        
        # Step 6: Solve for P
        p_text = f"P = {supply_intercept} + {supply_slope} × {self.eq_x:.1f} = {self.eq_y:.1f}"
        self.add_calculation_step(
            p_text,
            color=YELLOW,
            animate=animate,
            run_time=0.8 if animate else 0
        )
        
        # Add to equilibrium section
        equilibrium_text = f"Initial Equilibrium: Q = {self.eq_x:.1f}, P = {self.eq_y:.1f}"
        self.text_panel.add_text(
            "equilibrium",
            equilibrium_text,
            color=GREEN_B,
            animate=animate,
            run_time=0.8 if animate else 0
        )
        
        # Add a conclusion to explain the meaning
        self.text_panel.add_text(
            "conclusions",
            "At equilibrium, the market clears with no shortage or surplus.",
            color=WHITE,
            animate=animate,
            run_time=0.8 if animate else 0
        )
        
        # Create the equilibrium point with visuals
        if animate:
            self.create_equilibrium_point(self.eq_x, self.eq_y, animate=True, run_time=1.0)
        
        return self.eq_x, self.eq_y

    def create_equilibrium_point(self, eq_x, eq_y, animate=True, run_time=1.0):
        """Create or update the equilibrium point with dashed lines and labels"""
        # Store the equilibrium values
        self.eq_x = eq_x
        self.eq_y = eq_y
        
        # Get the equilibrium color from config
        eq_color = self.config["labels_config"]["equilibrium_color"]
        
        # Create the equilibrium point
        equilibrium_point = Dot(
            self.axes.c2p(eq_x, eq_y),
            color=eq_color,
            radius=0.08
        )
        
        # Create dashed lines to axes
        x_line = DashedLine(
            start=self.axes.c2p(eq_x, 0),
            end=self.axes.c2p(eq_x, eq_y),
            color=eq_color,
            stroke_width=2,
            stroke_opacity=0.8
        )
        
        y_line = DashedLine(
            start=self.axes.c2p(0, eq_y),
            end=self.axes.c2p(eq_x, eq_y),
            color=eq_color,
            stroke_width=2,
            stroke_opacity=0.8
        )
        
        # Create labels with the equilibrium values
        # Use a slightly larger size for better visibility
        x_label = Text(f"Q = {eq_x:.1f}", color=eq_color).scale(0.55)
        y_label = Text(f"P = {eq_y:.1f}", color=eq_color).scale(0.55)
        
        # Position the labels with better spacing
        x_label.next_to(x_line, DOWN, buff=0.2)
        y_label.next_to(y_line, LEFT, buff=0.2)
        
        # Animate if requested
        if animate:
            # Remove old elements if they exist
            old_elements = []
            for attr in ["eq_point", "eq_x_line", "eq_y_line", "eq_x_label", "eq_y_label"]:
                old_element = getattr(self, attr, None)
                if old_element and old_element in self.scene.mobjects:
                    old_elements.append(old_element)
            
            # Fade out old elements if any exist
            if old_elements:
                self.scene.play(
                    *[FadeOut(elem) for elem in old_elements],
                    run_time=run_time/3
                )
            
            # Animate creation of new elements
            self.scene.play(
                Create(equilibrium_point),
                run_time=run_time/3
            )
            
            self.scene.play(
                Create(x_line),
                Create(y_line), 
                run_time=run_time/3
            )
            
            self.scene.play(
                Write(x_label),
                Write(y_label),
                run_time=run_time/3
            )
        else:
            # Just add the elements without animation
            # First remove any old elements
            for attr in ["eq_point", "eq_x_line", "eq_y_line", "eq_x_label", "eq_y_label"]:
                old_element = getattr(self, attr, None)
                if old_element and old_element in self.scene.mobjects:
                    self.scene.remove(old_element)
                    
            # Then add the new elements
            self.scene.add(equilibrium_point, x_line, y_line, x_label, y_label)
        
        # Store the new elements
        self.eq_point = equilibrium_point
        self.eq_x_line = x_line
        self.eq_y_line = y_line
        self.eq_x_label = x_label
        self.eq_y_label = y_label
        
        return equilibrium_point

    def add_all_elements(self, animate=True):
        """Add all elements to the scene in a sequential manner"""
        elements = []
        
        # Add text panel first
        if animate:
            self.text_panel.introduce(run_time=1.0)
        else:
            panel_elements = self.text_panel.get_all_elements()
            for element in panel_elements:
                self.scene.add(element)
                elements.append(element)
        
        # Add axes
        if animate:
            self.scene.play(Create(self.axes), run_time=0.8)
        else:
            self.scene.add(self.axes)
        elements.append(self.axes)
        
        # Add axis labels
        if animate:
            self.scene.play(
                Write(self.x_label),
                Write(self.y_label),
                run_time=0.6
            )
        else:
            self.scene.add(self.x_label, self.y_label)
        elements.extend([self.x_label, self.y_label])
        
        # Add title if it exists
        if hasattr(self, 'title') and self.title:
            if animate:
                self.scene.play(Write(self.title), run_time=0.6)
            else:
                self.scene.add(self.title)
            elements.append(self.title)
            
        # Add curves with separate animations for clarity
        if animate:
            # Add supply curve
            self.scene.play(
                Create(self.supply_curve),
                run_time=0.8
            )
            # Add demand curve
            self.scene.play(
                Create(self.demand_curve),
                run_time=0.8
            )
        else:
            self.scene.add(self.supply_curve, self.demand_curve)
        elements.extend([self.supply_curve, self.demand_curve])
        
        # Solve for equilibrium and show steps
        if animate:
            self.solve_equilibrium(animate=True)
        
        # Return all elements that were added
        return elements 

    def shift_supply_curve(self, new_config, new_label="S'", animate=True, run_time=1.0, update_equilibrium=True, shift_style="parallel"):
        """
        Shift the supply curve to a new position and recalculate equilibrium.
        
        Args:
            new_config (dict): Dictionary with new slope, intercept, and color values.
            new_label (str): New label for the supply curve.
            animate (bool): Whether to animate the change.
            run_time (float): Duration of the animation.
            update_equilibrium (bool): Whether to recalculate equilibrium after shifting.
            shift_style (str): Style of shift - "parallel", "pivot", or "slope_change".
        
        Returns:
            tuple: New equilibrium point (x, y) if update_equilibrium is True, else None.
        """
        # Store original supply config
        orig_supply = self.config["supply_config"].copy()
        
        # Create new supply curve
        x_min, x_max = self.config["background_config"]["x_range"][0], self.config["background_config"]["x_range"][1]
        
        # Create points for the new curve
        new_x_vals = np.linspace(x_min, x_max, 100)
        new_y_vals = new_config["intercept"] + new_config["slope"] * new_x_vals
        
        new_supply_curve = self.axes.plot_line_graph(
            x_values=new_x_vals,
            y_values=new_y_vals,
            line_color=new_config["color"],
            stroke_width=new_config.get("stroke_width", 3),
            add_vertex_dots=False
        )
        
        # Create new supply label
        new_supply_label = Text(new_label, color=new_config["color"]).scale(0.7)
        
        # Position the new label at the right end of the new supply curve
        x_pos = x_max * 0.8
        y_pos = new_config["intercept"] + new_config["slope"] * x_pos
        new_supply_label.move_to(self.axes.c2p(x_pos, y_pos) + UP * 0.3 + RIGHT * 0.3)
        
        if animate:
            # Animate the transition
            self.scene.play(
                ReplacementTransform(self.supply_curve, new_supply_curve),
                ReplacementTransform(self.supply_label, new_supply_label),
                run_time=run_time
            )
        else:
            # Remove old and add new without animation
            self.scene.remove(self.supply_curve, self.supply_label)
            self.scene.add(new_supply_curve, new_supply_label)
        
        # Update the supply curve and label references
        self.supply_curve = new_supply_curve
        self.supply_label = new_supply_label
        
        # Update the config with new values
        self.config["supply_config"] = new_config
        self.config["labels_config"]["supply_label"] = new_label
        
        # Calculate and show new equilibrium if requested
        if update_equilibrium:
            # Clear old equilibrium elements if they exist
            if hasattr(self, 'eq_point') and self.eq_point is not None:
                if animate:
                    self.scene.play(
                        FadeOut(self.eq_point),
                        FadeOut(self.eq_x_line),
                        FadeOut(self.eq_y_line),
                        FadeOut(self.eq_x_label),
                        FadeOut(self.eq_y_label),
                        run_time=0.7
                    )
                else:
                    self.scene.remove(
                        self.eq_point, self.eq_x_line, self.eq_y_line,
                        self.eq_x_label, self.eq_y_label
                    )
            
            # Calculate and display new equilibrium
            return self.solve_equilibrium(animate=animate)
        
        return None

    def shift_demand_curve(self, new_config, new_label="D'", animate=True, run_time=1.0, update_equilibrium=True, shift_style="parallel"):
        """
        Shift the demand curve to a new position and recalculate equilibrium.
        
        Args:
            new_config (dict): Dictionary with new slope, intercept, and color values.
            new_label (str): New label for the demand curve.
            animate (bool): Whether to animate the change.
            run_time (float): Duration of the animation.
            update_equilibrium (bool): Whether to recalculate equilibrium after shifting.
            shift_style (str): Style of shift - "parallel", "pivot", or "slope_change".
        
        Returns:
            tuple: New equilibrium point (x, y) if update_equilibrium is True, else None.
        """
        # Store original demand config
        orig_demand = self.config["demand_config"].copy()
        
        # Create new demand curve
        x_min, x_max = self.config["background_config"]["x_range"][0], self.config["background_config"]["x_range"][1]
        
        # Create points for the new curve
        new_x_vals = np.linspace(x_min, x_max, 100)
        new_y_vals = new_config["intercept"] + new_config["slope"] * new_x_vals
        
        new_demand_curve = self.axes.plot_line_graph(
            x_values=new_x_vals,
            y_values=new_y_vals,
            line_color=new_config["color"],
            stroke_width=new_config.get("stroke_width", 3),
            add_vertex_dots=False
        )
        
        # Create new demand label
        new_demand_label = Text(new_label, color=new_config["color"]).scale(0.7)
        
        # Position the new label at the right end of the new demand curve
        x_pos = x_max * 0.8
        y_pos = new_config["intercept"] + new_config["slope"] * x_pos
        new_demand_label.move_to(self.axes.c2p(x_pos, y_pos) + DOWN * 0.3 + RIGHT * 0.3)
        
        if animate:
            # Animate the transition
            self.scene.play(
                ReplacementTransform(self.demand_curve, new_demand_curve),
                ReplacementTransform(self.demand_label, new_demand_label),
                run_time=run_time
            )
        else:
            # Remove old and add new without animation
            self.scene.remove(self.demand_curve, self.demand_label)
            self.scene.add(new_demand_curve, new_demand_label)
        
        # Update the demand curve and label references
        self.demand_curve = new_demand_curve
        self.demand_label = new_demand_label
        
        # Update the config with new values
        self.config["demand_config"] = new_config
        self.config["labels_config"]["demand_label"] = new_label
        
        # Calculate and show new equilibrium if requested
        if update_equilibrium:
            # Clear old equilibrium elements if they exist
            if hasattr(self, 'eq_point') and self.eq_point is not None:
                if animate:
                    self.scene.play(
                        FadeOut(self.eq_point),
                        FadeOut(self.eq_x_line),
                        FadeOut(self.eq_y_line),
                        FadeOut(self.eq_x_label),
                        FadeOut(self.eq_y_label),
                        run_time=0.7
                    )
                else:
                    self.scene.remove(
                        self.eq_point, self.eq_x_line, self.eq_y_line,
                        self.eq_x_label, self.eq_y_label
                    )
            
            # Calculate and display new equilibrium
            return self.solve_equilibrium(animate=animate)
        
        return None 