"""
Market Equilibrium Demo using Maenigna components.

This animation demonstrates how to use the reusable SupplyDemandCurve component
to visualize and solve a market equilibrium problem.
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

class MarketEquilibriumDemo(Scene):
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
        # "A market has a supply function P = 2 + 0.5Q and demand function P = 10 - 0.5Q.
        # Find the equilibrium price and quantity."
        
        # Generate speech audio files
        voice_intro = self.create_voice(
            "In this example, we'll solve a market equilibrium problem using supply and demand curves.",
            "market_intro"
        )
        
        voice_problem = self.create_voice(
            "A market has a supply function P equals 2 plus 0.5 Q and demand function P equals 10 minus 0.5 Q. "
            "Let's find the equilibrium price and quantity.",
            "market_problem"
        )
        
        voice_setup = self.create_voice(
            "First, we'll visualize the supply and demand curves on a graph.",
            "market_setup"
        )
        
        voice_solution = self.create_voice(
            "To find the equilibrium, we need to find where the supply and demand curves intersect. "
            "We'll solve this step by step.",
            "market_solution"
        )
        
        voice_equilibrium = self.create_voice(
            "At the equilibrium point, supply equals demand, so the quantity is 8 and the price is 6.",
            "market_equilibrium"
        )
        
        voice_conclusion = self.create_voice(
            "This is the market equilibrium, where the quantity supplied equals the quantity demanded.",
            "market_conclusion"
        )
        
        # Step 1: Introduce the problem - Audio only intro
        self.add_sound(voice_intro)
        self.wait(2)  # Wait for audio to play
        
        # Create custom configuration for our specific problem
        sd_config = {
            "supply_config": {
                "slope": 0.5,
                "intercept": 2,
                "label_text": "Supply",
                "stroke_width": 3
            },
            "demand_config": {
                "slope": -0.5,
                "intercept": 10,
                "label_text": "Demand",
                "stroke_width": 3
            },
            "labels_config": {
                "title": "Market Equilibrium Problem",
                "equilibrium_color": YELLOW
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
                "title_creation_time": 1.0,
                "curve_creation_time": 1.2,
                "curve_label_creation_time": 0.8,
                "equilibrium_creation_time": 1.0,
                "wait_time_after_step": 0.5
            },
            "show_calculation_steps": True  # Enable the step-by-step calculation feature
        }
        
        # Add problem description as voiceover
        self.add_sound(voice_problem)
        
        # Create the supply-demand component
        sd_curve = SupplyDemandCurve(self, sd_config)
        
        # Check layout validity before proceeding
        sd_curve.verify_layout(verbose=True)
        
        # Step 2: Create a supply-demand diagram using our component
        self.add_sound(voice_setup)
        
        # Introduce the supply-demand diagram with improved timing
        sd_curve.introduce_curves()
        
        # Step 3: Highlight the equilibrium solution with step-by-step calculations
        self.add_sound(voice_solution)
        
        # Emphasize the equilibrium point
        self.play(sd_curve.eq_point.animate.scale(1.5), run_time=1)
        self.wait(0.5)
        
        # Solve the equilibrium using the step-by-step method
        # This will automatically add the calculation steps to the panel
        sd_curve.solve_equilibrium(animate=True)
        
        # Add the equilibrium audio
        self.add_sound(voice_equilibrium)
        self.wait(2)
        
        # Step 4: Conclusion
        self.add_sound(voice_conclusion)
        
        # Add a visual conclusion using the component's text system
        conclusion = sd_curve.add_conclusion_text(
            "At equilibrium: Quantity Supplied = Quantity Demanded",
            color=YELLOW
        )
        
        self.wait(2) 