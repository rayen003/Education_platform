"""
Supply Shift Demo using Maenigna components.

This animation demonstrates how to use the reusable SupplyDemandCurve component
to visualize what happens when there's a shift in the supply curve.
"""

try:
    from manim import Scene, Create, Write, FadeIn, FadeOut, ReplacementTransform, DOWN, RIGHT, UP, LEFT, ORIGIN
    from manim import BLUE, GREEN, RED, YELLOW, WHITE, Text
    import os
    import tempfile
    from gtts import gTTS
    from lib.supply_demand import SupplyDemandCurve
except ModuleNotFoundError:
    print("Error: Some modules are missing. Please install them using:")
    print("pip install manim gtts")
    exit()

class SupplyShiftDemo(Scene):
    def create_voice(self, text, filename):
        """Generate text-to-speech audio file"""
        tts = gTTS(text=text, lang='en', slow=False)
        sound_dir = "media/sounds"
        os.makedirs(sound_dir, exist_ok=True)
        
        filepath = f"{sound_dir}/{filename}.mp3"
        tts.save(filepath)
        return filepath
    
    def construct(self):
        # Problem description: 
        # "In the coffee market, the supply function is P = 3 + 0.4Q and demand function is P = 9 - 0.6Q.
        # Due to a drought, the supply curve shifts to P = 5 + 0.4Q. How does this affect the equilibrium?"
        
        # Generate speech audio files
        voice_intro = self.create_voice(
            "In this example, we'll analyze the effect of a supply shift on market equilibrium.",
            "shift_intro"
        )
        
        voice_problem = self.create_voice(
            "In the coffee market, the supply function is P equals 3 plus 0.4 Q, and demand function is P equals 9 minus 0.6 Q. "
            "Due to a drought, the supply curve shifts to P equals 5 plus 0.4 Q. How does this affect the equilibrium?",
            "shift_problem"
        )
        
        voice_initial = self.create_voice(
            "First, let's set up our initial supply and demand curves and find the initial equilibrium with step-by-step calculation.",
            "shift_initial"
        )
        
        voice_equilibrium = self.create_voice(
            "In the initial equilibrium, quantity is 6 units and price is 5 dollars and 40 cents.",
            "shift_equilibrium"
        )
        
        voice_shock = self.create_voice(
            "Now, let's see what happens when a drought causes the supply curve to shift upward. "
            "This means that at any given quantity, suppliers now require a higher price.",
            "shift_shock"
        )
        
        voice_newcalc = self.create_voice(
            "Let's calculate the new equilibrium using our step-by-step approach.",
            "shift_newcalc"
        )
        
        voice_newequilibrium = self.create_voice(
            "The new equilibrium occurs at quantity 4 units and price 6 dollars and 60 cents. "
            "We can see that the price has increased and quantity has decreased.",
            "shift_newequilibrium"
        )
        
        voice_conclusion = self.create_voice(
            "This demonstrates how a negative supply shock leads to higher prices and lower quantities in the market.",
            "shift_conclusion"
        )
        
        # Step 1: Introduce the problem - Audio only intro
        self.add_sound(voice_intro)
        self.wait(2)  # Wait for audio to play
        
        # Create custom configuration for our specific problem
        sd_config = {
            "supply_config": {
                "slope": 0.4,
                "intercept": 3,
                "label_text": "Supply",
                "stroke_width": 3
            },
            "demand_config": {
                "slope": -0.6,
                "intercept": 9,
                "label_text": "Demand",
                "stroke_width": 3
            },
            "labels_config": {
                "title": "Supply Shift in Coffee Market",
                "equilibrium_color": YELLOW,
                "show_equilibrium": True
            },
            "layout_config": {
                "graph_position": LEFT * 3.0,   # Position graph on left side
                "panel_width": 5.8,            # Width of the text panel (slightly narrower to fix alignment)
                "panel_height": 7.0,           # Height of the text panel
                "title_offset": UP * 3.2,       # Title offset from center
                "verify_layout": True
            },
            "animation_config": {
                "axes_creation_time": 1.0,
                "labels_creation_time": 0.8,
                "curve_creation_time": 1.2,
                "curve_label_creation_time": 0.8,
                "equilibrium_creation_time": 1.2,
                "wait_time_after_step": 0.5
            },
            "show_calculation_steps": True  # Enable the step-by-step calculation feature
        }
        
        # Add problem description as voiceover
        self.add_sound(voice_problem)
        
        # Create the supply-demand component
        sd_curve = SupplyDemandCurve(self, sd_config)
        
        # Check layout validity
        layout_valid = sd_curve.verify_layout(verbose=True)
        
        # Step 2: Create a supply-demand diagram using our component
        self.add_sound(voice_initial)
        
        # Introduce the supply-demand diagram with improved timing
        sd_curve.introduce_curves()
        
        # Step 3: Highlight the initial equilibrium with calculations
        self.add_sound(voice_equilibrium)
        
        # Emphasize the equilibrium point
        self.play(sd_curve.eq_point.animate.scale(1.5), run_time=1)
        self.wait(1)
        
        # Step 4: Shift the supply curve
        self.add_sound(voice_shock)
        
        # Shift the supply curve - now returns animations
        shift_anims = sd_curve.shift_supply(new_intercept=5)  # Same slope, new intercept
        self.play(*shift_anims)
        self.wait(1)
        
        # Add calculation steps for the new equilibrium
        self.add_sound(voice_newcalc)
        
        # Clear the calculation section and add new calculation steps for the shifted supply
        sd_curve.text_panel.clear_section("calculations", animate=True, run_time=0.5)
        
        # Step 1: Set up the new equations
        sd_curve.add_calculation_step(
            "Step 1: Set new supply equal to demand",
            color=WHITE
        )
        
        # Step 2: Substitute the equations
        sd_curve.add_calculation_step(
            "5 + 0.4Q = 9 - 0.6Q",
            color=YELLOW, 
            fade_previous=False
        )
        
        # Step 3: Rearrange
        sd_curve.add_calculation_step(
            "0.4Q + 0.6Q = 9 - 5",
            color=YELLOW,
            fade_previous=False
        )
        
        # Step 4: Simplify
        sd_curve.add_calculation_step(
            "1.0Q = 4",
            color=YELLOW,
            fade_previous=False
        )
        
        # Step 5: Solve for Q
        sd_curve.add_calculation_step(
            "Q = 4.0",
            color=YELLOW,
            fade_previous=False
        )
        
        # Step 6: Calculate P
        sd_curve.add_calculation_step(
            "P = 5 + 0.4 × 4.0 = 6.6",
            color=YELLOW,
            fade_previous=False
        )
        
        # Update the equilibrium point
        self.add_sound(voice_newequilibrium)
        update_eq_anims = sd_curve.get_updated_equilibrium()
        self.play(*update_eq_anims)
        self.wait(1)
        
        # Conclusion
        self.add_sound(voice_conclusion)
        
        # Add conclusion text to the right panel
        conclusion = sd_curve.add_conclusion_text(
            "Effect: ↑ Price, ↓ Quantity",
            color=YELLOW
        )
        
        # Add a more detailed explanation
        sd_curve.add_conclusion_text(
            "Higher costs → Lower supply → Higher prices",
            color=WHITE
        )
        
        self.wait(2) 