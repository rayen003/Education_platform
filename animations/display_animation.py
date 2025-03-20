#!/usr/bin/env python3
"""
Script to display the perpetuity animation template code.
"""

import os
import sys
from pathlib import Path

def display_template_code():
    """Display the template animation code with line numbers."""
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    template_path = os.path.join(current_dir, "perpetuity_manim_template.py")
    
    if not os.path.exists(template_path):
        print(f"Template file not found at {template_path}")
        return
    
    # Read the template file
    with open(template_path, 'r') as f:
        template_code = f.read()
    
    print("\n" + "=" * 80)
    print("PERPETUITY ANIMATION TEMPLATE CODE")
    print("=" * 80)
    
    for i, line in enumerate(template_code.split('\n'), 1):
        print(f"{i:3d}: {line}")
    
    print("=" * 80 + "\n")
    
    # Also explain what the animation does
    print("\nAnimation Overview:")
    print("-------------------")
    print("1. Introduction to the concept of perpetuity")
    print("2. Explanation of the formula PV = C/r")
    print("3. Example problem: $5,000 annually at 5% interest")
    print("4. Step-by-step calculation: $5,000 / 0.05 = $100,000")
    print("5. Visual timeline showing infinite cash flows")
    print("6. Explanation of why $100,000 today equals $5,000 annually forever at 5%")
    
    print("\nKey Animation Sequences:")
    print("----------------------")
    print("1. Title sequence")
    print("2. Formula display and explanation")
    print("3. Problem statement")
    print("4. Calculation steps")
    print("5. Timeline visualization with arrows showing cash flows")
    print("6. Highlighting the result")
    print("7. Final recap of key points")
    
    print("\nThis animation would demonstrate to students how perpetuity calculations work,")
    print("showing both the mathematical formula and a visual representation of the concept.")
    print("The meta-agent would verify this animation for educational clarity and accuracy.")

if __name__ == "__main__":
    display_template_code() 