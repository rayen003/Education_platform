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

## Development Journal

### May 22, 2023 - Confidence Metrics System

Added a comprehensive confidence metrics system to the math assessment tool:
- Implemented `ConfidenceMetrics` class to assess confidence in feedback, hints, analysis, and chat interactions
- Integrated machine learning-based confidence prediction with calibration
- Created data collection infrastructure for training and improving confidence models
- Fixed handling of different state object formats and potential None values in metrics assessment
- Added feature extraction methods for all assessment types
- Created enhanced finance demo showcasing confidence assessment capabilities

### May 23, 2023 - Modern UI Implementation

Enhanced the UI with a modern learning platform design:
- Created a grid-based component layout with cards for different learning resources
- Updated the UI to include Videos, Lessons, Questions, Mock Exams, Notes, and Flashcards sections
- Improved visual appearance with consistent styling, icons, and responsive design
- Integrated confidence visualization components into the assessment workflow
- Added the ability to show confidence levels visually through progress bars and badges

### May 24, 2023 - Confidence UI Integration

Enhanced the confidence metrics visualization in the UI:
- Created `confidence_display.py` module with reusable UI components for showing confidence levels
- Implemented three visualization types:
  1. Confidence bars: Progress bars with color coding based on confidence level
  2. Confidence badges: Compact inline indicators for showing confidence next to hints
  3. Confidence tooltips: Contextual tooltips explaining the confidence assessment
- Added explanatory tooltips that provide details on how confidence is calculated
- Integrated confidence visuals throughout the math assessment workflow for:
  - Problem solution confidence
  - Analysis of student work
  - Feedback confidence
  - Hint reliability
  - Reasoning step verification
  - Chat response accuracy
- Color coded all confidence indicators (green for high, orange for medium, red for low)
- Added confidence explanation expanders to help users understand the reliability of AI responses

### May 25, 2023 - UI Navigation and Layout Improvements

Redesigned the application interface for improved usability:
- Simplified the home page to focus on the two core services: Knowledge Graph and Math Assessment
- Implemented a navigation system that allows users to switch between services
- Created a page-based navigation model with "Back to Home" options for better user flow
- Enhanced the Knowledge Graph visualization:
  - Increased the vertical space allocated to the graph for better visibility
  - Moved resource information to a sidebar on the right
  - Improved the styling of resource panels with colored backgrounds instead of white
  - Added more detailed node information displays
- Unified the Math Assessment interface:
  - Combined structured and chat modes into a single chat-based interface
  - Retained the toggle for switching between modes while maintaining a consistent UI
  - Improved the chat history display with better formatting and timestamps
  - Enhanced the hint and follow-up question flow
- Added responsive layout adjustments for different screen sizes
- Improved accessibility with better color contrast and larger touch targets
