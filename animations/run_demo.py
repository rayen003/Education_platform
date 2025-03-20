#!/usr/bin/env python3
"""
Run script for perpetuity demo with correct path setup.
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path

# Add the root directory to Python path
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

def run_template_animation():
    """Run the template animation directly."""
    template_path = os.path.join(current_dir, "perpetuity_manim_template.py")
    if not os.path.exists(template_path):
        print(f"Template file not found at {template_path}")
        return None
    
    # Read the template file
    with open(template_path, 'r') as f:
        template_code = f.read()
    
    # Create a temporary file with the uncommented render call
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
        modified_code = template_code.replace('# PerpetuityConcept().render()', 'PerpetuityConcept().render()')
        temp_file.write(modified_code.encode('utf-8'))
        temp_file_path = temp_file.name
    
    try:
        # Create output directory
        output_dir = os.path.join(current_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the Manim command
        print(f"Executing template animation: {temp_file_path}")
        cmd = [
            'manim',
            '-qm',  # Medium quality
            '-o', output_dir,
            temp_file_path
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Template animation rendered to: {output_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing template animation: {e}")
    except Exception as e:
        print(f"Error in template animation process: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

if __name__ == "__main__":
    print(f"Running from directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    # Run the template animation directly
    run_template_animation() 