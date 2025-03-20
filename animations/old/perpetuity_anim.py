#!/usr/bin/env python3
from manim import *

class PerpetuityAnimation(Scene):
    def construct(self):
        # Define values for our perpetuity problem
        payment = 1000  # Annual payment of $1,000
        rate = 0.05     # Interest rate of 5%
        pv = payment / rate  # Present value = $20,000
        
        # Title
        title = Text("Perpetuity Valuation", font_size=40).to_edge(UP)
        self.play(Write(title))
        
        # Create a timeline for cash flows
        timeline = NumberLine(
            x_range=[0, 10, 1],
            length=8,
            include_numbers=True,
            label_direction=DOWN,
        ).shift(DOWN)
        
        timeline_label = Text("Years", font_size=24).next_to(timeline, DOWN, buff=0.5)
        
        # Show timeline
        self.play(Create(timeline), Write(timeline_label))
        
        # Create cash flow arrows and labels
        arrows = []
        labels = []
        
        for year in range(1, 6):
            # Create arrow pointing up
            arrow = Arrow(
                start=timeline.n2p(year) + DOWN * 0.2,
                end=timeline.n2p(year) + UP * 1.5,
                color=GREEN,
                stroke_width=3
            )
            
            # Create payment label
            label = Text(f"${payment}", font_size=20, color=GREEN).next_to(arrow, UP)
            
            arrows.append(arrow)
            labels.append(label)
        
        # Show the first 5 cash flows
        self.play(
            *[GrowArrow(arrow) for arrow in arrows],
            *[FadeIn(label) for label in labels],
            run_time=1.5
        )
        
        # Indicate continuation to infinity
        dots = Text("...", font_size=36).next_to(arrows[-1], RIGHT)
        infinity = Text("∞", font_size=36).next_to(dots, RIGHT, buff=0.5)
        
        self.play(Write(dots), Write(infinity))
        self.wait(0.5)
        
        # Formula for present value
        formula = MathTex(
            "PV = \\frac{C}{r}",
            font_size=40
        ).shift(UP * 1.5 + RIGHT * 3)
        
        self.play(Write(formula))
        
        # Add values
        formula_with_values = MathTex(
            "PV = \\frac{\\$1,000}{0.05} = \\$20,000",
            font_size=40
        ).next_to(formula, DOWN)
        
        self.play(Write(formula_with_values))
        
        # Highlight present value at time 0
        pv_arrow = Arrow(
            start=timeline.n2p(0) + DOWN * 0.2,
            end=timeline.n2p(0) + DOWN * 1.5,
            color=BLUE,
            stroke_width=4
        )
        
        pv_label = Text(f"PV = ${int(pv)}", font_size=24, color=BLUE).next_to(pv_arrow, DOWN)
        
        self.play(GrowArrow(pv_arrow), Write(pv_label))
        
        # Final explanation
        explanation = Text(
            "A perpetuity is a stream of equal payments\nthat continue indefinitely",
            font_size=24
        ).to_edge(DOWN)
        
        self.play(Write(explanation))
        self.wait(2)
        
        # Clean up for a simpler explanation
        self.play(
            FadeOut(explanation),
            FadeOut(title),
            FadeOut(dots),
            FadeOut(infinity),
            *[FadeOut(arrow) for arrow in arrows],
            *[FadeOut(label) for label in labels],
            FadeOut(formula),
            FadeOut(timeline_label)
        )
        
        # Simplified explanation
        summary = Text(
            "Present Value of Perpetuity",
            font_size=36
        ).to_edge(UP)
        
        formula_final = MathTex(
            "PV = \\frac{\\$1,000}{0.05} = \\$20,000",
            font_size=48
        ).next_to(summary, DOWN, buff=0.75)
        
        self.play(
            Write(summary),
            Transform(formula_with_values, formula_final),
            pv_label.animate.scale(1.5).shift(UP)
        )
        
        # Final beat
        self.wait(1)

# This will directly render the scene when run
if __name__ == "__main__":
    scene = PerpetuityAnimation()
    scene.render(quality="medium_quality", preview=True) 