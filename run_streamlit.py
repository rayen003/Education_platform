#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ["streamlit", "openai", "dotenv"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_environment():
    """Check if environment variables are properly set."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check if OpenAI API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY environment variable is not set.")
            print("You can set it in a .env file in the project root or in your environment.")
            print("The application will run, but some features might be limited.")
    except ImportError:
        # dotenv not available - skip this check
        pass

def main():
    """Run the Streamlit app."""
    # Check for dependencies
    if not check_dependencies():
        return 1
    
    # Check environment variables
    check_environment()
    
    streamlit_file = os.path.join(project_root, "app", "templates", "streamlit_app.py")
    
    if not os.path.exists(streamlit_file):
        print(f"Error: Streamlit app file not found at {streamlit_file}")
        return 1
    
    # Create necessary directories
    static_data_dir = os.path.join(project_root, "app", "static", "data")
    json_files_dir = os.path.join(static_data_dir, "json_files")
    os.makedirs(static_data_dir, exist_ok=True)
    os.makedirs(json_files_dir, exist_ok=True)
    
    # Run the Streamlit app
    print(f"Starting Streamlit app from: {streamlit_file}")
    return subprocess.call(["streamlit", "run", streamlit_file])

if __name__ == "__main__":
    sys.exit(main()) 