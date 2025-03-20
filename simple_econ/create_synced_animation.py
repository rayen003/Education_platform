#!/usr/bin/env python
"""
Create Synchronized Animation with Narration

This script creates an HTML page that combines the Plotly animation with
synchronized audio narration for a complete interactive educational experience.
"""

import os
import sys
import argparse
import json
import shutil
from supply_demand import SupplyDemandPlot
from animated_example import AnimatedSupplyDemandPlot

def create_synced_html(shift_type, html_path, audio_path, output_dir):
    """
    Create an HTML page with synchronized animation and audio.
    
    Args:
        shift_type (str): Type of shift - "supply" or "demand"
        html_path (str): Path to the Plotly HTML file
        audio_path (str): Path to the audio narration file
        output_dir (str): Directory to save the output file
        
    Returns:
        str: Path to the synced HTML file
    """
    # Read the Plotly HTML file
    with open(html_path, 'r') as f:
        plotly_html = f.read()
    
    # Extract the plot data portion
    start_idx = plotly_html.find("Plotly.newPlot(")
    end_idx = plotly_html.find("</script>", start_idx)
    plot_data = plotly_html[start_idx:end_idx]
    
    # Create the synchronized HTML file
    shift_title = "Supply Shift" if shift_type == "supply" else "Demand Shift"
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive {shift_title} Analysis</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            h1, h2 {{
                color: #333;
                text-align: center;
            }}
            .container {{
                display: flex;
                flex-direction: column;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                padding: 20px;
                margin-bottom: 20px;
            }}
            .plot-container {{
                height: 600px;
                margin-bottom: 20px;
            }}
            .controls {{
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 10px;
                padding: 10px;
                background: #f0f0f0;
                border-radius: 4px;
                margin-bottom: 20px;
            }}
            .button {{
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 4px;
                transition: background-color 0.3s;
            }}
            .button:hover {{
                background-color: #45a049;
            }}
            .button:disabled {{
                background-color: #cccccc;
                cursor: not-allowed;
            }}
            .explanation {{
                background-color: #f0f0f0;
                border-left: 4px solid #4CAF50;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 0 4px 4px 0;
            }}
            .step {{
                margin-bottom: 10px;
                padding: 10px;
                background-color: #f8f8f8;
                border-left: 3px solid #ddd;
                transition: all 0.3s;
            }}
            .step.active {{
                background-color: #e6f7e6;
                border-left: 3px solid #4CAF50;
            }}
            #progress-bar {{
                width: 100%;
                background-color: #ddd;
                border-radius: 4px;
                margin-top: 10px;
            }}
            #progress {{
                height: 10px;
                width: 0%;
                background-color: #4CAF50;
                border-radius: 4px;
                transition: width 0.1s;
            }}
        </style>
    </head>
    <body>
        <h1>Interactive {shift_title} Analysis</h1>
        
        <div class="container">
            <div id="plotly-div" class="plot-container"></div>
            
            <div class="controls">
                <button id="play-btn" class="button">▶ Play with Narration</button>
                <button id="pause-btn" class="button" disabled>⏸ Pause</button>
                <button id="restart-btn" class="button">↻ Restart</button>
            </div>
            
            <div id="progress-bar">
                <div id="progress"></div>
            </div>
            
            <h2>Economic Explanation</h2>
            <div class="explanation">
                <div id="narration-text">
                    Click "Play with Narration" to start the animated explanation.
                </div>
            </div>
        </div>
        
        <audio id="narration" src="{os.path.basename(audio_path)}" preload="auto"></audio>
        
        <script>
            // Initialize the plot
            {plot_data}
            
            // Setup animation controls and synchronization
            const audio = document.getElementById('narration');
            const playBtn = document.getElementById('play-btn');
            const pauseBtn = document.getElementById('pause-btn');
            const restartBtn = document.getElementById('restart-btn');
            const narrationText = document.getElementById('narration-text');
            const progress = document.getElementById('progress');
            
            // Animation control
            let currentFrame = 0;
            const totalFrames = 30;
            let animationInterval;
            
            // Narration segments with timing
            const narrationSegments = [
                {{ text: "Welcome to the interactive {shift_type} shift analysis.", time: 0, frame: 0 }},
                {{ text: "This visualization shows how changes in {shift_type} affect market equilibrium.", time: 3, frame: 0 }},
                {{ text: "The initial equilibrium is shown by the green dot where supply and demand curves intersect.", time: 6, frame: 0 }},
                {{ text: "Now we'll see what happens when the {shift_type} curve shifts.", time: 10, frame: 5 }},
                {{ text: "The {shift_type} curve is moving, changing the market equilibrium.", time: 13, frame: 15 }},
                {{ text: "Notice how the equilibrium point is changing, with both price and quantity adjusting.", time: 16, frame: 20 }},
                {{ text: "The new equilibrium has been established, with different price and quantity values.", time: 20, frame: 30 }}
            ];
            
            // Update narration text based on current time
            audio.addEventListener('timeupdate', () => {{
                const currentTime = audio.currentTime;
                let activeSegment = narrationSegments[0];
                
                // Find the current narration segment
                for (let i = 0; i < narrationSegments.length; i++) {{
                    if (currentTime >= narrationSegments[i].time &&
                        (i === narrationSegments.length - 1 || currentTime < narrationSegments[i+1].time)) {{
                        activeSegment = narrationSegments[i];
                        break;
                    }}
                }}
                
                // Update text
                narrationText.innerText = activeSegment.text;
                
                // Update progress bar
                const progressPercent = (currentTime / audio.duration) * 100;
                progress.style.width = progressPercent + '%';
                
                // Update animation frame if not manually controlling
                if (animationInterval) {{
                    const targetFrame = Math.min(activeSegment.frame, totalFrames);
                    if (currentFrame !== targetFrame) {{
                        Plotly.animate('plotly-div', [{{name: `frame${{targetFrame}}`}}], {{
                            frame: {{ duration: 300, redraw: true }},
                            transition: {{ duration: 300 }},
                            mode: 'immediate'
                        }});
                        currentFrame = targetFrame;
                    }}
                }}
            }});
            
            // Play button
            playBtn.addEventListener('click', () => {{
                audio.play();
                playBtn.disabled = true;
                pauseBtn.disabled = false;
                
                // Start animation sync
                animationInterval = true;
            }});
            
            // Pause button
            pauseBtn.addEventListener('click', () => {{
                audio.pause();
                playBtn.disabled = false;
                pauseBtn.disabled = true;
                
                // Stop animation sync
                animationInterval = false;
            }});
            
            // Restart button
            restartBtn.addEventListener('click', () => {{
                audio.pause();
                audio.currentTime = 0;
                playBtn.disabled = false;
                pauseBtn.disabled = true;
                
                // Reset animation
                animationInterval = false;
                currentFrame = 0;
                Plotly.animate('plotly-div', [{{name: 'frame0'}}], {{
                    frame: {{ duration: 0, redraw: true }},
                    mode: 'immediate'
                }});
                
                narrationText.innerText = "Click \"Play with Narration\" to start the animated explanation.";
                progress.style.width = '0%';
            }});
            
            // When audio ends
            audio.addEventListener('ended', () => {{
                playBtn.disabled = false;
                pauseBtn.disabled = true;
                animationInterval = false;
            }});
        </script>
    </body>
    </html>
    """
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the HTML file
    output_path = os.path.join(output_dir, f"interactive_{shift_type}_shift.html")
    with open(output_path, "w") as f:
        f.write(html_template)
    
    # Copy the audio file to the output directory
    audio_output = os.path.join(output_dir, os.path.basename(audio_path))
    shutil.copy2(audio_path, audio_output)
    
    return output_path

def main():
    """Main function to create synchronized HTML pages"""
    parser = argparse.ArgumentParser(description='Create synchronized animations with narration')
    
    parser.add_argument('--shift', 
                        type=str, 
                        default='supply',
                        choices=['supply', 'demand', 'both'],
                        help='Which type of shift to create')
    
    parser.add_argument('--output_dir', 
                        type=str, 
                        default='./output/interactive',
                        help='Directory to save the output files')
    
    parser.add_argument('--open', 
                        action='store_true',
                        help='Open the HTML visualization after generating')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which shifts to create
    shifts_to_create = []
    if args.shift == 'both':
        shifts_to_create = ['supply', 'demand']
    else:
        shifts_to_create = [args.shift]
    
    # Process each shift type
    for shift_type in shifts_to_create:
        print(f"Creating interactive {shift_type} shift visualization...")
        
        # First, create the animated visualization if it doesn't exist
        animated_dir = os.path.join(os.path.dirname(args.output_dir), "animated")
        os.makedirs(animated_dir, exist_ok=True)
        
        html_path = os.path.join(animated_dir, f"animated_{shift_type}_shift.html")
        audio_path = os.path.join(animated_dir, f"animated_{shift_type}_shift.mp3")
        
        # If the animated files don't exist, create them
        if not (os.path.exists(html_path) and os.path.exists(audio_path)):
            print(f"Generating animated {shift_type} shift first...")
            
            # Basic configuration
            config = {
                "supply_config": {
                    "slope": 0.8,
                    "intercept": 1,
                    "color": "rgb(0, 0, 255)"  # Blue
                },
                "demand_config": {
                    "slope": -0.7,
                    "intercept": 8,
                    "color": "rgb(255, 0, 0)"  # Red
                },
                "layout_config": {
                    "title": f"Animated {shift_type.capitalize()} Shift",
                    "width": 950,
                    "height": 600
                }
            }
            
            # Create the animated plot
            sd_plot = AnimatedSupplyDemandPlot(config)
            
            # Define new configuration for the shift
            if shift_type == "supply":
                new_config = {
                    "intercept": 3,  # Higher intercept (decreased supply)
                    "color": "rgb(0, 0, 150)"  # Darker blue
                }
            else:  # demand shift
                new_config = {
                    "intercept": 10,  # Higher intercept (increased demand)
                    "color": "rgb(150, 0, 0)"  # Darker red
                }
                
            # Create the animation
            html_path = sd_plot.create_animated_shift(
                shift_type=shift_type,
                new_config=new_config,
                output_dir=animated_dir
            )
            
            audio_path = os.path.join(animated_dir, f"animated_{shift_type}_shift.mp3")
        
        # Create the synchronized HTML
        output_path = create_synced_html(
            shift_type=shift_type,
            html_path=html_path,
            audio_path=audio_path,
            output_dir=args.output_dir
        )
        
        print(f"Interactive visualization created: {output_path}")
        
        # Open the file if requested
        if args.open:
            if sys.platform == 'darwin':  # macOS
                import subprocess
                subprocess.run(['open', output_path])
            elif sys.platform == 'win32':  # Windows
                os.startfile(output_path)
            else:  # Linux or other Unix
                import subprocess
                subprocess.run(['xdg-open', output_path])
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 