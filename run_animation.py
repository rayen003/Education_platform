#!/usr/bin/env python

import os
import sys
import subprocess
import argparse

def main():
    """
    Wrapper script to easily run the economic animations with proper configuration.
    
    This script automatically sets up the correct environment and runs the animation
    generation script with the provided parameters.
    """
    parser = argparse.ArgumentParser(description='Run economic animations')
    
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
    
    parser.add_argument('--preview', 
                        action='store_true',
                        help='Open the animation after rendering')
    
    args = parser.parse_args()
    
    # Build the command
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               'animations', 'generate_animations.py')
    
    cmd = [
        sys.executable,
        script_path,
        f"--scene={args.scene}",
        f"--quality={args.quality}"
    ]
    
    # Print what we're doing
    print(f"Generating {args.scene} animation...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     'animations', 'media', 'videos')
            output_file = os.path.join(output_dir, f"{args.scene}_animation.mp4")
            
            print(f"\nAnimation generated successfully!")
            print(f"Output file: {output_file}")
            
            # Open the video if preview is requested
            if args.preview and os.path.exists(output_file):
                print("Opening animation...")
                if sys.platform == 'darwin':  # macOS
                    subprocess.run(['open', output_file])
                elif sys.platform == 'win32':  # Windows
                    os.startfile(output_file)
                else:  # Linux or other Unix
                    subprocess.run(['xdg-open', output_file])
    except subprocess.CalledProcessError as e:
        print(f"Error generating animation: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 