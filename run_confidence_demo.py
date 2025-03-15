#!/usr/bin/env python3
"""
Run the confidence demo app.

This script runs the confidence indicators demo for math assessment.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

def main():
    """Run the confidence demo."""
    print("Starting Confidence Demo App")
    
    # Path to the demo script
    demo_script = os.path.join(
        project_root, 
        "app", 
        "math_services", 
        "examples", 
        "confidence_demo.py"
    )
    
    # Check if the script exists
    if not os.path.exists(demo_script):
        print(f"Error: Demo script not found at {demo_script}")
        return 1
    
    # Run the demo with Streamlit
    return subprocess.call(["streamlit", "run", demo_script])

if __name__ == "__main__":
    sys.exit(main()) 