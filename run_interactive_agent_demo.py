#!/usr/bin/env python3
"""
Run the Interactive Math Agent Demo.

This script executes the interactive math agent demo, showcasing
different user interaction patterns and agent capabilities.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).absolute().parent)
if project_root not in sys.path:
    sys.path.append(project_root)

def main():
    """Run the interactive agent demo."""
    print("Starting Interactive Math Agent Demo...")
    
    # Execute the demo script
    from app.math_services.examples.interactive_agent_demo import main as run_demo
    run_demo()

if __name__ == "__main__":
    main() 