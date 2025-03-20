#!/usr/bin/env python

import os
import sys
from manim import config, WHITE, BLUE, RED, GREEN
import argparse

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.economic_scenes import (
    SupplyDemandEquilibriumScene,
    ShiftingSupplyScene,
    ShiftingDemandScene
)

def main():
    """
    Main function to generate economic animations.
    
    Usage:
        python generate_animations.py --scene [scene_name] --quality [quality]
    
    Scene options:
        - equilibrium: Show basic supply and demand equilibrium
        - supply_shift: Show effects of shifting supply curve
        - demand_shift: Show effects of shifting demand curve
        
    Quality options:
        - low: 480p
        - medium: 720p
        - high: 1080p
        - ultra_high: 1440p
    """
    parser = argparse.ArgumentParser(description='Generate economic animations')
    
    parser.add_argument('--scene', 
                        type=str, 
                        default='equilibrium',
                        choices=['equilibrium', 'supply_shift', 'demand_shift'],
                        help='Which scene to generate')
    
    parser.add_argument('--quality', 
                        type=str, 
                        default='medium',
                        choices=['low', 'medium', 'high', 'ultra_high'],
                        help='Quality of the output video')
    
    args = parser.parse_args()
    
    # Set up resolution based on quality parameter
    quality_settings = {
        'low': '480p',
        'medium': '720p',
        'high': '1080p',
        'ultra_high': '1440p'
    }
    
    # Configure Manim
    config.background_color = WHITE
    config.output_file = f"{args.scene}_animation"
    config.pixel_height = int(quality_settings[args.quality].replace('p', ''))
    config.pixel_width = int(config.pixel_height * 16 / 9)  # 16:9 aspect ratio
    config.frame_rate = 30
    config.preview = True  # Always preview the animation
    
    # Show progress bar
    config.progress_bar = 'display'  # Options: 'none', 'display', 'leave'
    
    # Configure media output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media')
    os.makedirs(output_dir, exist_ok=True)
    config.media_dir = output_dir
    
    # Map scene names to scene classes
    scene_map = {
        'equilibrium': SupplyDemandEquilibriumScene,
        'supply_shift': ShiftingSupplyScene,
        'demand_shift': ShiftingDemandScene
    }
    
    # Get the appropriate scene class
    scene_class = scene_map.get(args.scene)
    
    if scene_class:
        print(f"Generating {args.scene} animation at {quality_settings[args.quality]} quality...")
        
        # Instantiate and render the scene
        scene = scene_class()
        scene.render()
        
        output_path = os.path.join(output_dir, 'videos', f"{args.scene}_animation.mp4")
        print(f"Animation generated successfully! Output file: {output_path}")
    else:
        print(f"Error: Scene '{args.scene}' not found.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 