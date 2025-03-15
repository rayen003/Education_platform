# Educational Technology Platform

This project provides an educational technology platform with two main features:
1. Knowledge Graph Generation - Visualize learning material as an interactive knowledge graph
2. Math Assessment Tool - Get instant feedback on math problem solutions

## Getting Started

### Prerequisites

- Python 3.8 or later
- Required Python packages:
  - streamlit
  - openai
  - python-dotenv
  - flask (for the original web app)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Edtech_project_clean.git
   cd Edtech_project_clean
   ```

2. Install required packages:
   ```bash
   pip install streamlit openai python-dotenv flask
   ```

3. Set up your environment variables by creating a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Application

You can run either the Streamlit app or the original Flask app:

### Streamlit App (Recommended)

Run the Streamlit app with:

```bash
python run_streamlit.py
```

The app should open in your browser automatically. If not, navigate to the URL shown in the terminal (typically http://localhost:8501).

### Original Flask App

Run the Flask app with:

```bash
export FLASK_APP=app
export FLASK_ENV=development
flask run
```

Then navigate to http://localhost:5000 in your browser.

## Features

### Knowledge Graph Generation

Upload a syllabus or educational material text to visualize it as an interactive knowledge graph. The system will:
- Extract modules and concepts
- Map relationships between concepts
- Provide an interactive visualization
- Generate metadata and statistics

### Math Assessment Tool

Get instant feedback on math problems:
- Enter your math question
- Submit your answer
- Receive instant correctness feedback
- Get hints and guidance
- Track your progress

## Using Without OpenAI API Key

The platform includes a fallback mock mode when an OpenAI API key isn't available:
- Knowledge graph generation will use a simplified parsing algorithm
- Math assessment will provide basic feedback without advanced analysis
- Some features may have limited functionality

To use the full capabilities, provide a valid OpenAI API key in the `.env` file.
