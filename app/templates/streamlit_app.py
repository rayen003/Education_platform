import streamlit as st
import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import datetime
from openai import OpenAI
import time
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Set up page config (MUST be the first Streamlit command)
st.set_page_config(
    page_title="Educational Technology Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/Edtech_project_clean',
        'Report a bug': 'https://github.com/yourusername/Edtech_project_clean/issues',
        'About': 'Educational Technology Platform with Knowledge Graph and Math Assessment capabilities.'
    }
)

# Add the project directory to the Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Import required modules
from app.math_services.services.llm.openai_service import OpenAILLMService
from app.math_services.agent.math_agent import MathAgent
from app.knowledge_graph.api_adapter import KnowledgeGraphService
from app.math_services.models.state import MathState, InteractionMode, ChatMessage
from app.math_services.services.service_container import ServiceContainer
from app.math_services.services.llm.openai_service import OpenAILLMService
from app.math_services.agent.meta_agent import MetaAgent
from app.math_services.commands.solve_command import MathSolveSymbolicallyCommand
from app.math_services.commands.analyze_command import MathAnalyzeCalculationCommand
from app.math_services.commands.feedback_command import MathGenerateFeedbackCommand
from app.math_services.commands.hint_command import MathGenerateHintCommand
from app.math_services.commands.reasoning_command import MathGenerateReasoningCommand
from app.math_services.commands.chat_command import MathChatFollowUpCommand

# Import our custom confidence display components
from app.math_services.ui.confidence_display import (
    display_confidence_bar,
    display_confidence_badge,
    display_confidence_tooltip,
    confidence_explanation
)

# Load environment variables
load_dotenv()

# Initialize the services
try:
    # Check if OpenAI API key is available and valid
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Clean up API key if it contains prefix
    if api_key and "OPENAI_API_KEY=" in api_key:
        api_key = api_key.replace("OPENAI_API_KEY=", "").strip()
        # Update environment variable with cleaned key
        os.environ["OPENAI_API_KEY"] = api_key
    
    if not api_key:
        # Create default API key file if not exists
        env_file = os.path.join(project_root, '.env')
        if not os.path.exists(env_file):
            with open(env_file, 'w') as f:
                f.write('OPENAI_API_KEY=\n')
            st.sidebar.error("⚠️ API key file created. Please add your OpenAI API key to the .env file and restart the app.")
        else:
            st.sidebar.warning("⚠️ OpenAI API key not found. Using mock mode.")
        
        use_mock = True
    else:
        # Verify the API key format
        if api_key.startswith('your_api') or len(api_key) < 20:
            st.sidebar.error("⚠️ Invalid OpenAI API key format. Please update your .env file with a valid key.")
            use_mock = True
        else:
            # Test the API key with a simple request
            try:
                test_client = OpenAI(api_key=api_key)
                test_client.models.list()
                st.sidebar.success("✅ OpenAI API key is valid!")
                use_mock = False
            except Exception as e:
                error_message = str(e)
                if "401" in error_message:
                    st.sidebar.error(f"⚠️ API key validation failed: Invalid API key. Please check your API key.")
                else:
                    st.sidebar.error(f"⚠️ API key validation failed: {error_message}")
                use_mock = True
    
    # Initialize services based on mock mode
    llm_service = OpenAILLMService(model="gpt-4o-mini", mock_mode=use_mock)
    meta_agent = MetaAgent(model="gpt-4o-mini", llm_service=llm_service)
    service_container = ServiceContainer(llm_service=llm_service, meta_agent=meta_agent)
    math_agent = MathAgent(model="gpt-4o-mini")
    kg_service = KnowledgeGraphService(llm_service, mock_mode=use_mock)
except Exception as e:
    st.error(f"Error initializing services: {str(e)}")
    st.warning("Some functionality may be limited due to initialization errors.")
    # Fallback initialization
    llm_service = OpenAILLMService(model="gpt-4o-mini", mock_mode=True)
    math_agent = MathAgent(model="gpt-4o-mini")
    kg_service = KnowledgeGraphService(llm_service, mock_mode=True)  # Fallback to mock mode

# Helper functions for displaying math content with confidence indicators
def display_math_feedback(feedback, confidence=None):
    """Display math feedback with confidence indicator if available."""
    st.write(feedback)
    
    if confidence is not None:
        # Add a visual confidence indicator
        display_confidence_bar(confidence, "Feedback Confidence")
        
        # Add an explanation of what the confidence means
        with st.expander("What does this confidence level mean?"):
            st.write(confidence_explanation(confidence))

def display_math_hints(hints, confidence=None):
    """Display math hints with confidence indicators if available."""
    for i, hint in enumerate(hints, 1):
        st.markdown(f'<div class="hint"><strong>Hint #{i}:</strong> {hint}</div>', 
                   unsafe_allow_html=True)
        
        # If we have confidence for this hint
        if confidence is not None:
            display_confidence_badge(confidence)

def display_math_analysis(analysis, confidence=None):
    """Display math problem analysis with confidence indicator if available."""
    if analysis.get("is_correct", False):
        st.success("Your answer is correct!")
    else:
        st.error("Your answer needs some work.")
    
    if "error_type" in analysis and analysis["error_type"]:
        st.write(f"**Error Type:** {analysis['error_type']}")
    
    if "misconception" in analysis and analysis["misconception"]:
        st.write(f"**Misconception:** {analysis['misconception']}")
    
    if "calculation_steps" in analysis and analysis["calculation_steps"]:
        with st.expander("Calculation Steps"):
            for i, step in enumerate(analysis["calculation_steps"], 1):
                st.write(f"{i}. {step}")
    
    if confidence is not None:
        display_confidence_bar(confidence, "Analysis Confidence")

# Create sidebar for settings
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/knowledge-sharing.png", width=80)
    st.title("EdTech Platform")
    st.markdown("---")
    
    # Theme selection
    theme = st.selectbox(
        "Theme",
        ["Light", "Dark", "Blue", "Green"],
        index=0
    )
    
    # Settings
    st.subheader("Settings")
    show_labels = st.checkbox("Show node labels", value=True)
    enable_physics = st.checkbox("Enable physics simulation", value=True)
    
    st.markdown("---")
    st.info("Drag nodes to rearrange the graph. Zoom with scroll wheel.")

# Custom CSS with dynamic theming based on selection
if theme == "Dark":
    bg_color = "#121212"
    text_color = "#f0f0f0"
    card_bg = "#1e1e1e"
    accent_color = "#BB86FC"
    correct_color = "#28a745"  # Green for high confidence
    incorrect_color = "#dc3545"  # Red for low confidence
    warning_color = "#ff9800"  # Orange for medium confidence
    module_color = "#3498db"  # Blue for modules
    concept_color = "#2ecc71"  # Green for concepts
    graph_bg_color = "#1A1A1A"  # Lighter background for graph
    node_label_color = "#FFFFFF"  # Bright white for dark backgrounds
elif theme == "Blue":
    bg_color = "#E8F0F8"
    text_color = "#333333"
    card_bg = "#FFFFFF"
    accent_color = "#1976D2"
    correct_color = "#28a745"  # Green for high confidence
    incorrect_color = "#dc3545"  # Red for low confidence
    warning_color = "#ff9800"  # Orange for medium confidence
    module_color = "#3498db"  # Blue for modules
    concept_color = "#2ecc71"  # Green for concepts
    graph_bg_color = "#F5F9FF"  # Light blue tint for graph
    node_label_color = "#333333"
elif theme == "Green":
    bg_color = "#F1F8E9"
    text_color = "#333333"
    card_bg = "#FFFFFF"
    accent_color = "#388E3C"
    correct_color = "#28a745"  # Green for high confidence
    incorrect_color = "#dc3545"  # Red for low confidence
    warning_color = "#ff9800"  # Orange for medium confidence
    module_color = "#3498db"  # Blue for modules
    concept_color = "#2ecc71"  # Green for concepts
    graph_bg_color = "#F5FFF5"  # Light green tint for graph
    node_label_color = "#333333"
else:  # Light
    bg_color = "#FFFFFF"
    text_color = "#333333"
    card_bg = "#F8F9FA"
    accent_color = "#2C3E50"
    correct_color = "#28a745"  # Green for high confidence
    incorrect_color = "#dc3545"  # Red for low confidence
    warning_color = "#ff9800"  # Orange for medium confidence
    module_color = "#3498db"  # Blue for modules
    concept_color = "#2ecc71"  # Green for concepts
    graph_bg_color = "#FFFFFF"  # White background for graph
    node_label_color = "#333333"

# Custom CSS to enhance aesthetics
st.markdown(f"""
<style>
    /* Main container styling */
    .main .block-container {{
        background-color: {bg_color};
        color: {text_color};
        padding: 1rem;
        border-radius: 20px;
        max-width: 100% !important;
        margin: 0 auto;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
    }}
    
    /* Common content padding */
    .st-emotion-cache-16txtl3 {{
        padding: 1.2rem;
    }}
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {{
        color: {accent_color};
        font-weight: 700;
        margin-bottom: 1.2rem;
        letter-spacing: -0.02em;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    h1 {{
        font-size: 3rem;
        text-align: center;
        margin-bottom: 2.5rem;
        background: linear-gradient(135deg, {accent_color}, #8BC34A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0.5rem 0;
    }}
    
    h2 {{
        font-size: 2.2rem;
        margin-top: 1.5rem;
        margin-bottom: 1.2rem;
        position: relative;
        padding-bottom: 0.5rem;
    }}
    
    h2::after {{
        content: '';
        position: absolute;
        left: 0;
        bottom: 0;
        height: 3px;
        width: 80px;
        background: linear-gradient(90deg, {accent_color}, {accent_color}33);
        border-radius: 3px;
    }}
    
    p {{
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }}
    
    /* Tab navigation */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
        background-color: {bg_color};
        padding: 1.2rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.07);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 60px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 15px;
        gap: 1px;
        padding: 10px 32px;
        font-weight: 600;
        color: {text_color};
        transition: all 0.3s ease;
        border: 1px solid {accent_color}33;
        font-size: 1.05rem;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {accent_color}22;
        color: {accent_color};
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-3px);
    }}
    
    /* Card styling */
    .card {{
        padding: 28px;
        border-radius: 20px;
        background-color: {card_bg};
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        margin-bottom: 25px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid {accent_color}15;
    }}
    
    .card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.18);
    }}
    
    /* Button styling */
    .stButton button {{
        border-radius: 12px;
        font-weight: 600;
        background-color: {accent_color};
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.95rem;
    }}
    
    .stButton button:hover {{
        background-color: {accent_color}ee;
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
    }}
    
    .stButton button:active {{
        transform: translateY(-1px);
    }}
    
    /* Form elements */
    .stTextInput input, .stTextArea textarea {{
        border-radius: 12px;
        border: 1px solid #ddd;
        padding: 15px;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.25s ease;
        font-size: 1rem;
    }}
    
    .stTextInput input:focus, .stTextArea textarea:focus {{
        border-color: {accent_color};
        box-shadow: 0 0 0 3px {accent_color}33;
    }}
    
    /* Progress bar */
    .stProgress .st-emotion-cache-1943zmx {{
        background-color: {accent_color}33;
        border-radius: 15px;
        height: 12px;
    }}
    
    .stProgress .st-emotion-cache-1lpbw6j {{
        background-color: {accent_color};
        border-radius: 15px;
        height: 12px;
    }}
    
    /* Slider */
    .stSlider [data-baseweb="slider"] {{
        margin-top: 1rem;
    }}
    
    .stSlider [data-testid="stThumbValue"] {{
        background-color: {accent_color} !important;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }}
    
    /* Selectbox */
    .stSelectbox [data-baseweb="select"] {{
        border-radius: 12px;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
    }}
    
    .stSelectbox [data-baseweb="select"]:focus-within {{
        border-color: {accent_color};
        box-shadow: 0 0 0 3px {accent_color}33;
    }}
    
    /* Footer */
    footer {{
        text-align: center;
        margin-top: 4rem;
        padding: 2rem;
        border-top: 1px solid {accent_color}22;
        font-size: 0.9rem;
        background: linear-gradient(180deg, {bg_color} 0%, {bg_color}dd 100%);
        position: relative;
    }}
    
    footer:before {{
        content: "";
        position: absolute;
        top: 1px;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, {accent_color}55, transparent);
    }}
    
    /* Feedback status */
    .correct-answer {{
        color: {correct_color};
        font-weight: bold;
        padding: 1rem;
        border-radius: 12px;
        background-color: {correct_color}15;
        display: inline-block;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
        border-left: 4px solid {correct_color};
    }}
    
    .incorrect-answer {{
        color: {incorrect_color};
        font-weight: bold;
        padding: 1rem;
        border-radius: 12px;
        background-color: {incorrect_color}15;
        display: inline-block;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
        border-left: 4px solid {incorrect_color};
    }}
    
    .warning-message {{
        color: {warning_color};
        font-weight: bold;
        padding: 1rem;
        border-radius: 12px;
        background-color: {warning_color}15;
        display: inline-block;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
        border-left: 4px solid {warning_color};
    }}
    
    .hint {{
        padding: 20px;
        margin-top: 20px;
        background-color: {accent_color}11;
        border-left: 4px solid {accent_color};
        border-radius: 12px;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
    }}
    
    /* Graph container */
    .graph-container {{
        height: 1200px;
        width: 100%;
        border: 1px solid {accent_color}33;
        border-radius: 20px;
        overflow: hidden;
        background-color: #FFFFFF; /* Set explicit white background */
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.18);
        position: relative;
        transition: all 0.5s ease;
    }}
    
    /* Improved layout for side-by-side display with even larger graph area */
    .graph-resources-container {{
        display: flex;
        width: 100%;
        position: relative;
        gap: 25px;
        margin-top: 20px;
        max-width: 100%;
    }}
    
    .graph-area {{
        flex: 7; /* Increased from 5 to 7 for much larger graph area */
        position: relative;
    }}
    
    .right-sidebar {{
        flex: 2;
        max-width: 350px;
        display: flex;
        flex-direction: column;
        gap: 25px;
        height: 1200px;
        overflow-y: auto;
        position: sticky;
        top: 20px;
    }}
    
    /* Resources panel with better visibility */
    .resources-panel {{
        background-color: #FFFFFF; /* Explicit white background */
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        border: 1px solid {accent_color}22;
        overflow-y: auto;
        max-height: 400px;
    }}
    
    /* Node details panel with better visibility */
    .node-details-panel {{
        background-color: #FFFFFF; /* Explicit white background */
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        z-index: 100;
        overflow-y: auto;
        max-height: 500px;
        opacity: 0.97;
        transition: transform 0.3s ease, opacity 0.3s ease;
        border: 1px solid {accent_color}22;
    }}
    
    /* Navigation panel with better visibility */
    .navigation-panel {{
        background-color: #FFFFFF; /* Explicit white background */
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        border: 1px solid {accent_color}22;
    }}
    
    /* Resource text styles with enhanced visibility */
    .resource-link {{
        display: block;
        margin-bottom: 15px;
        color: #1E2A38; /* Darker text color for better contrast */
        text-decoration: none;
        padding: 16px 20px;
        background-color: #F8F9FA; /* Light gray background instead of transparent */
        border-radius: 15px;
        transition: all 0.3s ease;
        font-weight: 600; /* Increased weight for better visibility */
        border-left: 4px solid {accent_color};
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
        display: flex;
        align-items: center;
    }}
    
    .module-resource {{
        border-left: 4px solid {module_color};
        background-color: #F0F8FF; /* Light blue background for module resources */
    }}
    
    .concept-resource {{
        border-left: 4px solid {concept_color};
        background-color: #F0FFF0; /* Light green background for concept resources */
    }}
    
    /* Node details content with enhanced visibility */
    .node-details-item .value {{
        background-color: #F8F9FA; /* Light gray background */
        padding: 15px;
        border-radius: 12px;
        word-break: break-word;
        border-left: 3px solid {accent_color}44;
        line-height: 1.5;
        color: #1E2A38; /* Ensure text is dark for visibility */
    }}
    
    .node-details-item.type .value {{
        display: inline-block;
        padding: 8px 15px;
        border-radius: 25px;
        background-color: #F0F8FF; /* Light blue background */
        color: #0A4B7D; /* Darker blue text for contrast */
        font-weight: 600;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1);
        border: none;
    }}
    
    .node-details-item.type .value.concept {{
        background-color: #F0FFF0; /* Light green background */
        color: #0A7D4B; /* Darker green text for contrast */
    }}
    
    /* Resources section */
    .node-details-resources {{
        margin-top: 30px;
        border-top: 2px solid {accent_color}33;
        padding-top: 25px;
        animation: fadeIn 0.8s ease;
    }}
    
    .node-details-resources h4 {{
        font-size: 1.5rem;
        margin-bottom: 20px;
        color: {accent_color};
        font-weight: 600;
        letter-spacing: -0.01em;
    }}
    
    .resource-link {{
        display: block;
        margin-bottom: 15px;
        color: {text_color};
        text-decoration: none;
        padding: 16px 20px;
        background-color: {accent_color}11;
        border-radius: 15px;
        transition: all 0.3s ease;
        font-weight: 500;
        border-left: 4px solid {accent_color};
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
    }}
    
    .module-resource {{
        border-left: 4px solid {module_color};
        background-color: {module_color}11;
    }}
    
    .concept-resource {{
        border-left: 4px solid {concept_color};
        background-color: {concept_color}11;
    }}
    
    .resource-link:hover {{
        background-color: {accent_color}22;
        transform: translateX(8px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }}
    
    /* Graph controls */
    .graph-controls {{
        position: absolute;
        bottom: 20px;
        left: 20px;
        z-index: 100;
        display: flex;
        gap: 12px;
        background-color: {card_bg}dd;
        padding: 12px;
        border-radius: 30px;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }}
    
    .graph-controls button {{
        width: 45px;
        height: 45px;
        border-radius: 50%;
        background-color: {accent_color}cc;
        color: white;
        border: none;
        font-size: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    
    .graph-controls button:hover {{
        background-color: {accent_color};
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.25);
    }}
    
    /* Graph legend */
    .graph-legend {{
        position: absolute;
        bottom: 20px;
        right: 20px;
        background: {card_bg}dd;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }}
    
    /* Stats cards */
    .stats-card {{
        padding: 20px;
        background-color: {card_bg};
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        margin-bottom: 20px;
        border: 1px solid {accent_color}22;
        position: relative;
        overflow: hidden;
    }}
    
    .stats-card:before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, {accent_color}, {accent_color}55);
        border-radius: 4px 4px 0 0;
    }}
    
    .stats-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.15);
        border-color: {accent_color}55;
    }}
    
    .stats-card h3 {{
        font-size: 2.5rem;
        margin: 0;
        color: {accent_color};
        font-weight: 700;
        background: linear-gradient(135deg, {accent_color}, {accent_color}77);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .stats-card p {{
        margin: 5px 0 0 0;
        opacity: 0.8;
        font-weight: 500;
        font-size: 1rem;
    }}
    
    /* Search input */
    #nodeSearch {{
        width: 100%;
        padding: 12px 15px;
        border-radius: 12px;
        border: 1px solid {accent_color}33;
        background-color: {bg_color};
        color: {text_color};
        font-size: 1rem;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.25s ease;
    }}
    
    #nodeSearch:focus {{
        border-color: {accent_color};
        box-shadow: 0 0 0 3px {accent_color}33;
        outline: none;
    }}
    
    /* Module navigation buttons */
    .module-nav-button {{
        display: block;
        width: 100%;
        padding: 12px 15px;
        margin: 8px 0;
        background-color: {module_color}15;
        color: {text_color};
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.25s ease;
        text-align: left;
        font-weight: 500;
        font-size: 0.95rem;
        position: relative;
        padding-left: 15px;
    }}
    
    .module-nav-button:before {{
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background-color: {module_color};
        border-radius: 4px 0 0 4px;
        opacity: 0.7;
    }}
    
    .module-nav-button:hover {{
        background-color: {module_color}25;
        transform: translateX(5px);
    }}
    
    /* Tooltips */
    .tooltip {{
        padding: 10px 15px;
        background-color: {card_bg}ee;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
        border-radius: 10px;
        font-size: 0.9rem;
        pointer-events: none;
        z-index: 1000;
        border: 1px solid {accent_color}22;
        max-width: 250px;
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }}
    
    /* Fullscreen mode */
    .graph-container:fullscreen {{
        padding: 20px;
        background-color: {graph_bg_color};
    }}
    
    .graph-container:fullscreen .graph-controls {{
        bottom: 30px;
        left: 30px;
    }}
    
    .graph-container:fullscreen .graph-legend {{
        bottom: 30px;
        right: 30px;
    }}
    
    /* Math assessment enhancement */
    .math-topics-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin-top: 25px;
    }}
    
    .math-topic {{
        flex: 1;
        min-width: 120px;
        padding: 15px;
        background-color: {accent_color}15;
        border-radius: 12px;
        text-align: center;
        transition: all 0.25s ease;
        cursor: pointer;
        font-weight: 500;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
    }}
    
    .math-topic:hover {{
        background-color: {accent_color}25;
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }}
    
    /* Sample problem cards */
    .sample-problem {{
        padding: 15px;
        background-color: {accent_color}10;
        border-radius: 12px;
        border-left: 4px solid {accent_color};
        transition: all 0.25s ease;
        cursor: pointer;
    }}
    
    .sample-problem:hover {{
        background-color: {accent_color}20;
        transform: translateX(5px);
    }}
    
    /* Loading animations */
    @keyframes pulse {{
        0% {{ opacity: 0.6; }}
        50% {{ opacity: 0.8; }}
        100% {{ opacity: 0.6; }}
    }}
    
    .loading-pulse {{
        animation: pulse 1.5s infinite;
    }}
    
    /* Responsive adjustments */
    @media (max-width: 992px) {{
        .graph-resources-container {{
            flex-direction: column;
        }}
        
        .right-sidebar {{
            max-width: 100%;
            height: auto;
        }}
        
        .graph-container {{
            height: 800px;
        }}
    }}
    
    /* Confidence badges */
    .confidence-high {{
        color: {correct_color};
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 12px;
        background-color: {correct_color}15;
        display: inline-block;
        border: 1px solid {correct_color};
    }}
    
    .confidence-medium {{
        color: {warning_color};
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 12px;
        background-color: {warning_color}15;
        display: inline-block;
        border: 1px solid {warning_color};
    }}
    
    .confidence-low {{
        color: {incorrect_color};
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 12px;
        background-color: {incorrect_color}15;
        display: inline-block;
        border: 1px solid {incorrect_color};
    }}
</style>
""", unsafe_allow_html=True)

# App header with animation
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1><span style="display: inline-block; animation: bounce 2s infinite;">🧠</span> Educational Technology Platform</h1>
    <p style="font-size: 1.2rem; opacity: 0.8;">Visualize knowledge graphs and practice math skills</p>
</div>
<style>
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
</style>
""", unsafe_allow_html=True)

# Create tabs
tab_knowledge_graph, tab_math_assessment = st.tabs(["📊 Knowledge Graph", "🔢 Math Assessment"])

# Knowledge Graph Tab
with tab_knowledge_graph:
    st.header("Knowledge Graph Visualization")
    
    if kg_service is None:
        st.error("Knowledge Graph service is not available. Please check your configuration.")
    else:
        # Input section
        with st.expander("Upload Syllabus", expanded=True):
            syllabus_text = st.text_area(
                "Paste your syllabus text below:",
                height=200,
                placeholder="Course Title: Introduction to Mathematics...",
                value=st.session_state.get('syllabus_text', '')
            )
            
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("🔍 Generate Knowledge Graph", key="generate_graph_btn"):
                    if syllabus_text:
                        with st.spinner("Generating knowledge graph... This may take a minute."):
                            try:
                                # Create a container to show the progress
                                status_container = st.empty()
                                status_container.info("Step 1/4: Processing syllabus text...")
                                
                                # Process the syllabus and generate a graph
                                result = kg_service.process_syllabus(syllabus_text, user_id="streamlit_user")
                                
                                if result.get('status') == 'success':
                                    status_container.info("Step 2/4: Graph generated, retrieving data...")
                                    st.success("Knowledge graph generated successfully!")
                                    
                                    try:
                                        # Get the generated graph data
                                        graph_data = kg_service.get_graph(result.get('graph_id'))
                                        status_container.info("Step 3/4: Graph data retrieved...")
                                        
                                        if not graph_data or not isinstance(graph_data, dict) or not graph_data.get('nodes'):
                                            status_container.warning("Generated graph is empty or invalid. Using demo graph...")
                                            # Log the actual data for debugging
                                            print(f"DEBUG: Retrieved graph data: {type(graph_data)}")
                                            if isinstance(graph_data, dict):
                                                print(f"DEBUG: Keys: {graph_data.keys()}")
                                            
                                            # Create a demo graph if the generated one is empty
                                            from app.knowledge_graph.api_adapter import KnowledgeGraphService
                                            temp_service = KnowledgeGraphService(mock_mode=True)
                                            graph_data = temp_service._generate_mock_graph(syllabus_text)
                                        
                                        # Ensure we have required fields
                                        if 'nodes' not in graph_data:
                                            graph_data['nodes'] = []
                                        if 'links' not in graph_data:
                                            graph_data['links'] = []
                                        
                                        # Add basic metadata if missing
                                        if 'metadata' not in graph_data:
                                            graph_data['metadata'] = {
                                                'course_title': 'Untitled Course',
                                                'timestamp': datetime.datetime.now().isoformat(),
                                                'generated_by': 'KnowledgeGraphService',
                                                'syllabus_length': len(syllabus_text)
                                            }
                                        
                                        # Store in session state for later
                                        st.session_state['current_graph'] = graph_data
                                        status_container.info("Step 4/4: Graph data processed successfully!")
                                        
                                        # Force a rerun to ensure the graph is displayed
                                        st.rerun()
                                        
                                    except Exception as e:
                                        st.error(f"Error processing graph data: {str(e)}")
                                        # Create a demo graph in case of error
                                        status_container.warning("Error occurred. Using demo graph instead...")
                                        from app.knowledge_graph.api_adapter import KnowledgeGraphService
                                        temp_service = KnowledgeGraphService(mock_mode=True)
                                        graph_data = temp_service._generate_mock_graph(syllabus_text)
                                        st.session_state['current_graph'] = graph_data
                                else:
                                    st.error(f"Error: {result.get('message', 'Failed to generate knowledge graph')}")
                                    status_container.warning("Error occurred. Using demo graph instead...")
                                    # Create a demo graph in case of error
                                    from app.knowledge_graph.api_adapter import KnowledgeGraphService
                                    temp_service = KnowledgeGraphService(mock_mode=True)
                                    graph_data = temp_service._generate_mock_graph(syllabus_text)
                                    st.session_state['current_graph'] = graph_data
                            except Exception as e:
                                st.error(f"Error processing syllabus: {str(e)}")
                                # Create a demo graph in case of error
                                from app.knowledge_graph.api_adapter import KnowledgeGraphService
                                temp_service = KnowledgeGraphService(mock_mode=True)
                                graph_data = temp_service._generate_mock_graph(syllabus_text)
                                st.session_state['current_graph'] = graph_data
                    else:
                        st.warning("Please enter a syllabus text")
            
            with col2:
                # Sample syllabus button
                if st.button("📋 Load Sample Syllabus", key="load_sample_btn"):
                    sample_path = os.path.join(project_root, 'app', 'static', 'data', 'sample_syllabus.txt')
                    if os.path.exists(sample_path):
                        with open(sample_path, 'r') as f:
                            sample_syllabus = f.read()
                        st.session_state['syllabus_text'] = sample_syllabus
                        st.rerun()
        
        # Display graph if available in session state
        if 'current_graph' in st.session_state:
            graph_data = st.session_state['current_graph']
            
            # Display statistics
            stat1, stat2, stat3, stat4 = st.columns(4)
            with stat1:
                st.markdown(f"""
                <div class="stats-card">
                    <h3>{len(graph_data.get('nodes', []))}</h3>
                    <p>Total Nodes</p>
                </div>
                """, unsafe_allow_html=True)
            with stat2:
                st.markdown(f"""
                <div class="stats-card">
                    <h3>{len(graph_data.get('links', []))}</h3>
                    <p>Connections</p>
                </div>
                """, unsafe_allow_html=True)
            with stat3:
                module_count = len([n for n in graph_data.get('nodes', []) if n.get('type') == 'module'])
                st.markdown(f"""
                <div class="stats-card">
                    <h3>{module_count}</h3>
                    <p>Modules</p>
                </div>
                """, unsafe_allow_html=True)
            with stat4:
                concept_count = len([n for n in graph_data.get('nodes', []) if n.get('type') == 'concept'])
                st.markdown(f"""
                <div class="stats-card">
                    <h3>{concept_count}</h3>
                    <p>Concepts</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Save graph data to JSON for visualization
            graph_json = json.dumps(graph_data)
            
            # Create a more dynamic visualization with D3.js
            st.subheader("Interactive Knowledge Graph")
            
            # Add debug information for troubleshooting
            if st.checkbox("Show debug info", key="show_debug", value=False):
                st.write("Graph data structure:")
                st.write(f"- Type: {type(graph_data)}")
                st.write(f"- Has nodes: {'nodes' in graph_data}")
                st.write(f"- Node count: {len(graph_data.get('nodes', []))}")
                st.write(f"- Has links: {'links' in graph_data}")
                st.write(f"- Link count: {len(graph_data.get('links', []))}")
                
                with st.expander("View raw graph data"):
                    st.json(graph_data)
            
            # Add debug mode to help troubleshoot JavaScript issues
            debug_mode = "false"
            if st.checkbox("Enable JavaScript debug mode", value=False):
                debug_mode = "true"
            
            # Generate resources JS function
            resources_js = """
            function getNodeResources(nodeId, nodeType) {
                // More realistic resources based on node type
                if (nodeType === "module") {
                    return [
                        { title: "Practice Quiz", url: "#" },
                        { title: "Recommended Reading", url: "#" },
                        { title: "Video Lecture Series", url: "#" }
                    ];
                } else {
                    return [
                        { title: `Introduction Video for ${nodeId}`, url: "#" },
                        { title: `Practice Quiz for ${nodeId}`, url: "#" },
                        { title: `Further Reading on ${nodeId}`, url: "#" }
                    ];
                }
            }
            """
            
            # Create HTML content with integrated resources panel
            html_content = f"""
            <div class="graph-resources-container">
                <!-- Left side: Graph visualization (much larger) -->
                <div class="graph-area">
                    <div class="graph-container">
                        <script src="https://d3js.org/d3.v7.min.js"></script>
                        <div id="graph-message" style="display:none; padding: 20px; text-align:center; color:#1E2A38; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: white; border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.2); z-index: 100;">
                            <h3>Loading graph visualization...</h3>
                            <p>If the graph doesn't appear, please check the console for errors.</p>
                        </div>
                        <svg id="graph" width="100%" height="100%"></svg>
                        
                        <div class="graph-controls">
                            <button onclick="zoomIn()" title="Zoom In">+</button>
                            <button onclick="zoomOut()" title="Zoom Out">-</button>
                            <button onclick="resetZoom()" title="Reset View">⟳</button>
                            <button onclick="toggleFullscreen()" title="Fullscreen Toggle">⛶</button>
                        </div>
                        
                        <div class="graph-legend" style="position: absolute; bottom: 15px; right: 15px; background: #FFFFFFEE; padding: 10px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);">
                            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {module_color}; margin-right: 8px;"></div>
                                <span style="color: #1E2A38; font-size: 0.9rem; font-weight: 500;">Module</span>
                            </div>
                            <div style="display: flex; align-items: center;">
                                <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {concept_color}; margin-right: 8px;"></div>
                                <span style="color: #1E2A38; font-size: 0.9rem; font-weight: 500;">Concept</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Right side: Sidebar with details and resources -->
                <div class="right-sidebar">
                    <!-- Node Details Panel -->
                    <div class="panel node-details-panel" id="nodeDetailsPanel" style="display: none; background-color: #FFFFFF;">
                        <button class="close-btn" onclick="hideNodeDetails()">&times;</button>
                        <h3 id="nodeDetailsTitle" style="color: #1E2A38;">Node Details</h3>
                        <div id="nodeDetailsContent">
                            <p style="color: #1E2A38;">Click on a node in the graph to view its details.</p>
                        </div>
                    </div>
                    
                    <!-- Resources Panel -->
                    <div class="panel resources-panel" id="resourcesPanel" style="background-color: #FFFFFF;">
                        <h3 style="color: {accent_color}; margin-bottom: 15px; border-bottom: 2px solid {accent_color}33; padding-bottom: 10px; font-size: 1.3rem;">📚 Learning Resources</h3>
                        <div id="resourcesContent">
                            <div style="text-align: center; padding: 20px 0;">
                                <div style="width: 60px; height: 60px; margin: 0 auto; background-color: #F0F8FF; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                                    <span style="font-size: 24px;">📚</span>
                                </div>
                                <p style="margin-top: 15px; color: #1E2A38; font-weight: 500;">Click on a node to view related learning resources</p>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Graph Navigation -->
                    <div class="panel navigation-panel" style="background-color: #FFFFFF;">
                        <h3 style="color: {accent_color}; margin-bottom: 15px; border-bottom: 2px solid {accent_color}33; padding-bottom: 10px; font-size: 1.3rem;">🧭 Graph Navigation</h3>
                        <div id="navigationContent">
                            <div style="margin-bottom: 15px;">
                                <input type="text" id="nodeSearch" placeholder="Search for a node..." style="background-color: #F8F9FA; color: #1E2A38;">
                            </div>
                            <div id="moduleList" style="margin-top: 15px;"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- D3.js graph implementation -->
            <script>
                // Parse the graph data from Python
                const graphData = {graph_json};
                const showLabels = {str(show_labels).lower()};
                const enablePhysics = {str(enable_physics).lower()};
                const debugMode = {debug_mode};
                
                // Colors from Streamlit themes
                const moduleColor = "{module_color}";
                const conceptColor = "{concept_color}";
                const accentColor = "{accent_color}";
                const textColor = "{text_color}";
                const bgColor = "{graph_bg_color}";
                const nodeLabelColor = "{node_label_color}";
                
                // Debug helper function
                function debug(message) {{
                    if (debugMode) {{
                        console.log(`DEBUG: ${{message}}`);
                    }}
                }}
                
                // Initialize the graph visualization
                function initGraph() {{
                    try {{
                        debug("Initializing graph visualization");
                        debug(`Graph data: ${{graphData.nodes.length}} nodes, ${{graphData.links.length}} links`);
                        
                        // Select the SVG element
                        const svg = d3.select("#graph");
                        const width = svg.node().parentNode.clientWidth;
                        const height = svg.node().parentNode.clientHeight;
                        
                        debug(`SVG dimensions: ${{width}}x${{height}}`);
                        
                        // Clear any existing content
                        svg.selectAll("*").remove();
                        
                        // Set explicit white background for the SVG
                        svg.append("rect")
                           .attr("width", "100%")
                           .attr("height", "100%")
                           .attr("fill", "#FFFFFF");
                        
                        // Create a group for the graph
                        const g = svg.append("g");
                        
                        // Create zoom behavior
                        const zoom = d3.zoom()
                            .scaleExtent([0.1, 4])
                            .on("zoom", (event) => {{
                                g.attr("transform", event.transform);
                            }});
                        
                        svg.call(zoom);
                        
                        // Define arrow markers for links
                        svg.append("defs").selectAll("marker")
                            .data(["arrow"])
                            .enter().append("marker")
                            .attr("id", d => d)
                            .attr("viewBox", "0 -5 10 10")
                            .attr("refX", 25)
                            .attr("refY", 0)
                            .attr("markerWidth", 6)
                            .attr("markerHeight", 6)
                            .attr("orient", "auto")
                            .append("path")
                            .attr("fill", `${{accentColor}}99`)
                            .attr("d", "M0,-5L10,0L0,5");
                        
                        // Process the graph data
                        const nodes = graphData.nodes;
                        const links = graphData.links.map(link => {{
                            // Ensure source and target references are correct
                            const source = nodes.find(node => node.id === link.source) || link.source;
                            const target = nodes.find(node => node.id === link.target) || link.target;
                            return {{...link, source, target}};
                        }});
                        
                        debug(`Processed ${{nodes.length}} nodes and ${{links.length}} links`);
                        
                        // Create force simulation
                        const simulation = d3.forceSimulation(nodes)
                            .force("link", d3.forceLink(links).id(d => d.id).distance(100))
                            .force("charge", d3.forceManyBody().strength(-400))
                            .force("center", d3.forceCenter(width / 2, height / 2))
                            .force("collide", d3.forceCollide().radius(60));
                        
                        if (!enablePhysics) {{
                            debug("Physics simulation disabled by user");
                            simulation.stop();
                        }}
                        
                        // Create links
                        const link = g.append("g")
                            .attr("class", "links")
                            .selectAll("line")
                            .data(links)
                            .enter().append("line")
                            .attr("stroke", d => `${{accentColor}}55`)
                            .attr("stroke-width", d => Math.max(1, d.strength * 3 || 1))
                            .attr("marker-end", "url(#arrow)")
                            .attr("stroke-dasharray", d => d.type === "prerequisite" ? "5,5" : null)
                            .on("mouseover", function(event, d) {{
                                try {{
                                    d3.select(this)
                                        .attr("stroke", accentColor)
                                        .attr("stroke-width", Math.max(2, d.strength * 4 || 2));
                                        
                                    // Show tooltip with link type
                                    const tooltip = d3.select("body").append("div")
                                        .attr("class", "tooltip")
                                        .style("position", "absolute")
                                        .style("padding", "8px")
                                        .style("background", `${{bgColor}}EE`)
                                        .style("color", textColor)
                                        .style("border-radius", "4px")
                                        .style("font-size", "12px")
                                        .style("box-shadow", "0 4px 8px rgba(0,0,0,0.2)")
                                        .style("pointer-events", "none")
                                        .style("opacity", 0)
                                        .style("z-index", 1000);
                                        
                                    tooltip.html(`<strong>Relationship:</strong> ${{d.type || "connected"}}`)
                                        .style("left", (event.pageX + 10) + "px")
                                        .style("top", (event.pageY - 28) + "px")
                                        .transition()
                                        .duration(200)
                                        .style("opacity", 0.9);
                                        
                                    // Store the tooltip reference
                                    this._tooltip = tooltip;
                                }} catch (e) {{
                                    console.error("Error in link mouseover:", e);
                                }}
                            }})
                            .on("mouseout", function() {{
                                try {{
                                    const d = d3.select(this).datum();
                                    d3.select(this)
                                        .attr("stroke", `${{accentColor}}55`)
                                        .attr("stroke-width", Math.max(1, d.strength * 3 || 1));
                                        
                                    // Remove tooltip
                                    if (this._tooltip) {{
                                        this._tooltip.transition()
                                            .duration(200)
                                            .style("opacity", 0)
                                            .remove();
                                    }}
                                }} catch (e) {{
                                    console.error("Error in link mouseout:", e);
                                }}
                            }});
                        
                        // Create nodes
                        const node = g.append("g")
                            .attr("class", "nodes")
                            .selectAll(".node")
                            .data(nodes)
                            .enter().append("g")
                            .attr("class", "node")
                            .attr("data-id", d => d.id)
                            .call(d3.drag()
                                .on("start", dragstarted)
                                .on("drag", dragged)
                                .on("end", dragended));
                        
                        // Add circles for nodes with different colors based on type
                        node.append("circle")
                            .attr("r", d => d.type === "module" ? 25 : 18)
                            .attr("fill", d => d.type === "module" ? moduleColor : conceptColor)
                            .attr("stroke", accentColor)
                            .attr("stroke-width", 2)
                            .attr("stroke-opacity", 0.6)
                            .attr("filter", "drop-shadow(0px 2px 3px rgba(0,0,0,0.2))"); // Add drop shadow for better visibility
                        
                        // Add text labels to nodes
                        if (showLabels) {{
                            node.append("text")
                                .attr("dy", d => d.type === "module" ? 40 : 30)
                                .attr("text-anchor", "middle")
                                .text(d => d.name || d.id)
                                .style("fill", "#1a1a1a") // Darker label color
                                .style("font-weight", d => d.type === "module" ? "bold" : "normal")
                                .style("font-size", d => d.type === "module" ? "14px" : "12px")
                                .style("pointer-events", "none")
                                .style("text-shadow", "0 0 4px #FFFFFF, 0 0 4px #FFFFFF, 0 0 4px #FFFFFF, 0 0 4px #FFFFFF"); // Enhanced white outline for visibility
                        }}
                        
                        // Populate the module list for navigation
                        const moduleList = document.getElementById("moduleList");
                        moduleList.innerHTML = '';
                        
                        const moduleNodes = nodes.filter(n => n.type === "module");
                        moduleNodes.forEach(module => {{
                            const button = document.createElement("button");
                            button.className = "module-nav-button";
                            button.innerHTML = module.name || module.id;
                            
                            button.addEventListener("click", () => {{
                                // Find the node in the visualization and focus on it
                                const moduleNode = d3.select(`[data-id="${{module.id}}"]`).datum();
                                if (moduleNode) {{
                                    // Center on the module
                                    centerOnNode(moduleNode);
                                    // Trigger the node details display
                                    showNodeDetails(moduleNode);
                                }}
                            }});
                            
                            moduleList.appendChild(button);
                        }});
                        
                        // Add search functionality
                        const nodeSearch = document.getElementById("nodeSearch");
                        nodeSearch.addEventListener("input", function() {{
                            const searchText = this.value.toLowerCase();
                            
                            // Filter nodes that match the search text
                            const matchingNodes = nodes.filter(node => 
                                (node.name && node.name.toLowerCase().includes(searchText)) || 
                                node.id.toLowerCase().includes(searchText)
                            );
                            
                            // Highlight matching nodes
                            node.selectAll("circle")
                                .attr("stroke-width", d => matchingNodes.includes(d) ? 4 : 2)
                                .attr("stroke", d => matchingNodes.includes(d) ? "#FF5733" : accentColor)
                                .attr("stroke-opacity", d => matchingNodes.includes(d) ? 1 : 0.6);
                                
                            // If only one node matches and search is non-empty, center on it
                            if (matchingNodes.length === 1 && searchText !== "") {{
                                centerOnNode(matchingNodes[0]);
                            }}
                        }});
                        
                        // Handle node click to show details
                        node.on("click", function(event, d) {{
                            showNodeDetails(d);
                            event.stopPropagation();
                        }});
                        
                        // Handle node mouseover to highlight connections
                        node.on("mouseover", function(event, d) {{
                            try {{
                                // Highlight the current node
                                d3.select(this).select("circle")
                                    .attr("stroke-width", 4)
                                    .attr("stroke-opacity", 1);
                                
                                // Find connected nodes
                                const connectedNodeIds = new Set();
                                links.forEach(link => {{
                                    if (link.source.id === d.id || link.source === d.id) {{
                                        connectedNodeIds.add(typeof link.target === 'object' ? link.target.id : link.target);
                                    }} else if (link.target.id === d.id || link.target === d.id) {{
                                        connectedNodeIds.add(typeof link.source === 'object' ? link.source.id : link.source);
                                    }}
                                }});
                                
                                // Highlight connected links
                                link.attr("stroke", linkData => {{
                                    if ((linkData.source.id === d.id || linkData.target.id === d.id)) {{
                                        return accentColor;
                                    }} else {{
                                        return `${{accentColor}}55`;
                                    }}
                                }})
                                .attr("stroke-width", linkData => {{
                                    if ((linkData.source.id === d.id || linkData.target.id === d.id)) {{
                                        return Math.max(2, linkData.strength * 4 || 2);
                                    }} else {{
                                        return Math.max(1, linkData.strength * 3 || 1);
                                    }}
                                }});
                                
                                // Highlight connected nodes
                                node.select("circle")
                                    .attr("stroke", nodeData => {{
                                        if (nodeData.id === d.id) {{
                                            return "#FF5733"; // Current node
                                        }} else if (connectedNodeIds.has(nodeData.id)) {{
                                            return accentColor; // Connected node
                                        }} else {{
                                            return accentColor; // Other nodes
                                        }}
                                    }})
                                    .attr("stroke-width", nodeData => {{
                                        if (nodeData.id === d.id) {{
                                            return 4; // Current node
                                        }} else if (connectedNodeIds.has(nodeData.id)) {{
                                            return 3; // Connected node
                                        }} else {{
                                            return 2; // Other nodes
                                        }}
                                    }})
                                    .attr("stroke-opacity", nodeData => {{
                                        if (nodeData.id === d.id || connectedNodeIds.has(nodeData.id)) {{
                                            return 1; // Current or connected nodes
                                        }} else {{
                                            return 0.2; // Other nodes
                                        }}
                                    }});
                                    
                                // Show tooltip with node name
                                const tooltip = d3.select("body").append("div")
                                    .attr("class", "tooltip")
                                    .style("position", "absolute")
                                    .style("padding", "8px")
                                    .style("background", `${{bgColor}}EE`)
                                    .style("color", textColor)
                                    .style("border-radius", "4px")
                                    .style("font-size", "12px")
                                    .style("box-shadow", "0 4px 8px rgba(0,0,0,0.2)")
                                    .style("pointer-events", "none")
                                    .style("opacity", 0)
                                    .style("z-index", 1000);
                                
                                tooltip.html(`<strong>${{d.name || d.id}}</strong><br>${{d.type}}`)
                                    .style("left", (event.pageX + 10) + "px")
                                    .style("top", (event.pageY - 28) + "px")
                                    .transition()
                                    .duration(200)
                                    .style("opacity", 0.9);
                                
                                // Store the tooltip reference
                                this._tooltip = tooltip;
                            }} catch (e) {{
                                console.error("Error in node mouseover:", e);
                            }}
                        }});
                        
                        // Handle node mouseout to reset highlights
                        node.on("mouseout", function() {{
                            try {{
                                // Reset all node colors
                                node.select("circle")
                                    .attr("stroke", accentColor)
                                    .attr("stroke-width", 2)
                                    .attr("stroke-opacity", 0.6);
                                
                                // Reset all link colors
                                link.attr("stroke", `${{accentColor}}55`)
                                    .attr("stroke-width", d => Math.max(1, d.strength * 3 || 1));
                                
                                // Remove tooltip
                                if (this._tooltip) {{
                                    this._tooltip.transition()
                                        .duration(200)
                                        .style("opacity", 0)
                                        .remove();
                                }}
                            }} catch (e) {{
                                console.error("Error in node mouseout:", e);
                            }}
                        }});
                        
                        // Start with one module expanded if graph just loaded
                        if (moduleNodes.length > 0) {{
                            setTimeout(() => {{
                                const firstModule = moduleNodes[0];
                                showNodeDetails(firstModule);
                                centerOnNode(firstModule);
                            }}, 500);
                        }}
                        
                        // Update function for simulation
                        simulation.on("tick", () => {{
                            link
                                .attr("x1", d => d.source.x)
                                .attr("y1", d => d.source.y)
                                .attr("x2", d => d.target.x)
                                .attr("y2", d => d.target.y);
                            
                            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
                        }});
                        
                        // Force simulation drag functions
                        function dragstarted(event, d) {{
                            try {{
                                if (!event.active) simulation.alphaTarget(0.3).restart();
                                d.fx = d.x;
                                d.fy = d.y;
                            }} catch (e) {{
                                console.error("Error in dragstarted:", e);
                            }}
                        }}
                        
                        function dragged(event, d) {{
                            try {{
                                d.fx = event.x;
                                d.fy = event.y;
                            }} catch (e) {{
                                console.error("Error in dragged:", e);
                            }}
                        }}
                        
                        function dragended(event, d) {{
                            try {{
                                if (!event.active) simulation.alphaTarget(0);
                                d.fx = null;
                                d.fy = null;
                            }} catch (e) {{
                                console.error("Error in dragended:", e);
                            }}
                        }}
                        
                        // Auto-center and zoom to fit the graph
                        resetZoom();
                        
                        debug("Graph visualization initialized successfully");
                    }} catch (e) {{
                        console.error("Error initializing graph:", e);
                        document.getElementById("graph-message").style.display = "block";
                        document.getElementById("graph-message").innerHTML = `
                            <h3>Error Loading Graph</h3>
                            <p>${{e.message}}</p>
                            <p>Please try refreshing the page or contact support.</p>
                        `;
                    }}
                }}
                
                // Center on a specific node
                function centerOnNode(d) {{
                    try {{
                        const svg = d3.select("#graph");
                        const width = svg.node().parentNode.clientWidth;
                        const height = svg.node().parentNode.clientHeight;
                        
                        const transform = d3.zoomIdentity
                            .translate(width / 2, height / 2)
                            .scale(1.2)
                            .translate(-d.x, -d.y);
                        
                        svg.transition()
                            .duration(750)
                            .call(d3.zoom().transform, transform);
                    }} catch (e) {{
                        console.error("Error centering on node:", e);
                    }}
                }}
                
                // Show node details and resources
                function showNodeDetails(d) {{
                    try {{
                        const nodeDetailsPanel = document.getElementById("nodeDetailsPanel");
                        const nodeDetailsTitle = document.getElementById("nodeDetailsTitle");
                        const nodeDetailsContent = document.getElementById("nodeDetailsContent");
                        const resourcesContent = document.getElementById("resourcesContent");
                        
                        // Set display to block (make it visible)
                        nodeDetailsPanel.style.display = "block";
                        
                        // Update the title with better visibility
                        nodeDetailsTitle.innerText = d.name || d.id;
                        nodeDetailsTitle.style.color = "#1E2A38";
                        
                        // Update the content with better visibility
                        let detailsHtml = `
                            <div class="node-details-item type">
                                <div class="label" style="color: #1E2A38;">Type:</div>
                                <div class="value ${{d.type}}" style="background-color: ${{d.type === 'module' ? '#F0F8FF' : '#F0FFF0'}}; color: #1E2A38;">
                                    ${{d.type.charAt(0).toUpperCase() + d.type.slice(1)}}
                                </div>
                            </div>
                            <div class="node-details-item">
                                <div class="label" style="color: #1E2A38;">ID:</div>
                                <div class="value" style="background-color: #F8F9FA; color: #1E2A38;">${{d.id}}</div>
                            </div>
                        `;
                        
                        // Find connected nodes
                        const connectedNodes = [];
                        graphData.links.forEach(link => {{
                            if (link.source === d.id || (typeof link.source === 'object' && link.source.id === d.id)) {{
                                const target = typeof link.target === 'object' ? link.target : 
                                    graphData.nodes.find(n => n.id === link.target);
                                if (target) {{
                                    connectedNodes.push({{
                                        node: target,
                                        relationship: link.type || "connected to"
                                    }});
                                }}
                            }} else if (link.target === d.id || (typeof link.target === 'object' && link.target.id === d.id)) {{
                                const source = typeof link.source === 'object' ? link.source : 
                                    graphData.nodes.find(n => n.id === link.source);
                                if (source) {{
                                    connectedNodes.push({{
                                        node: source,
                                        relationship: link.type ? `has ${{link.type}} from` : "connected to"
                                    }});
                                }}
                            }}
                        }});
                        
                        if (connectedNodes.length > 0) {{
                            detailsHtml += `
                                <div class="node-details-item">
                                    <div class="label">Connected Nodes (${{connectedNodes.length}}):</div>
                                    <div class="value">
                                        <ul style="padding-left: 20px; margin: 0;">
                            `;
                            
                            // Show first 5 connected nodes
                            const displayLimit = 5;
                            const displayNodes = connectedNodes.slice(0, displayLimit);
                            
                            displayNodes.forEach(connection => {{
                                detailsHtml += `
                                    <li>
                                        <a href="#" onclick="centerOnNode({{x: ${{connection.node.x}}, y: ${{connection.node.y}}}}); return false;" 
                                           style="color: ${{connection.node.type === 'module' ? moduleColor : conceptColor}}; text-decoration: none; font-weight: 500;">
                                            ${{connection.node.name || connection.node.id}}
                                        </a>
                                        <span style="opacity: 0.6; font-size: 0.9em;"> (${{connection.relationship}})</span>
                                    </li>
                                `;
                            }});
                            
                            // Show count of additional nodes if there are more
                            if (connectedNodes.length > displayLimit) {{
                                const additionalCount = connectedNodes.length - displayLimit;
                                detailsHtml += `
                                    <li style="list-style-type: none; margin-top: 8px;">
                                        <span style="font-style: italic; opacity: 0.7;">+ ${{additionalCount}} more connected node${{additionalCount > 1 ? 's' : ''}}</span>
                                    </li>
                                `;
                            }}
                            
                            detailsHtml += `
                                        </ul>
                                    </div>
                                </div>
                            `;
                        }}
                        
                        nodeDetailsContent.innerHTML = detailsHtml;
                        
                        // Update the resources
                        let resourcesHtml = `
                            <h4 style="margin-top: 0; color: #1E2A38; font-weight: 600; font-size: 1.1rem;">Learning Resources for ${{d.name || d.id}}</h4>
                            <div style="margin-top: 15px; background-color: #FFFFFF;">`;
                        
                        // Different resources based on node type with better styling
                        let resources = [];
                        if (d.type === "module") {{
                            resources = [
                                {{ icon: "📋", title: `Module Overview: ${{d.name || d.id}}`, url: "#", type: "module" }},
                                {{ icon: "📝", title: "Practice Quiz", url: "#", type: "module" }},
                                {{ icon: "📚", title: "Recommended Reading", url: "#", type: "module" }},
                                {{ icon: "🎬", title: "Video Lecture Series", url: "#", type: "module" }},
                                {{ icon: "💬", title: "Discussion Forum", url: "#", type: "module" }}
                            ];
                        }} else {{
                            resources = [
                                {{ icon: "🔍", title: `Introduction to ${{d.name || d.id}}`, url: "#", type: "concept" }},
                                {{ icon: "📝", title: `Practice Quiz for ${{d.name || d.id}}`, url: "#", type: "concept" }},
                                {{ icon: "📚", title: `Further Reading on ${{d.name || d.id}}`, url: "#", type: "concept" }},
                                {{ icon: "🧩", title: "Interactive Tutorial", url: "#", type: "concept" }}
                            ];
                        }}
                        
                        resources.forEach(resource => {{
                            const resourceClass = resource.type === "module" ? "module-resource" : "concept-resource";
                            resourcesHtml += `
                                <a href="${{resource.url}}" class="resource-link ${{resourceClass}}">
                                    <span class="resource-icon">${{resource.icon}}</span>
                                    <span>${{resource.title}}</span>
                                </a>
                            `;
                        }});
                        
                        resourcesHtml += `</div>`;
                        resourcesContent.innerHTML = resourcesHtml;
                    }} catch (e) {{
                        console.error("Error showing node details:", e);
                    }}
                }}
                
                // Hide node details panel
                function hideNodeDetails() {{
                    document.getElementById("nodeDetailsPanel").style.display = "none";
                }}
                
                // Zoom control functions
                function zoomIn() {{
                    const svg = d3.select("#graph");
                    svg.transition().duration(500).call(
                        d3.zoom().scaleBy, 1.5
                    );
                }}
                
                function zoomOut() {{
                    const svg = d3.select("#graph");
                    svg.transition().duration(500).call(
                        d3.zoom().scaleBy, 0.75
                    );
                }}
                
                function resetZoom() {{
                    try {{
                        const svg = d3.select("#graph");
                        const width = svg.node().parentNode.clientWidth;
                        const height = svg.node().parentNode.clientHeight;
                        
                        // Create a temporary g element to calculate bounds
                        const tempG = svg.append("g").attr("class", "temp");
                        tempG.selectAll("circle")
                            .data(graphData.nodes)
                            .enter()
                            .append("circle")
                            .attr("cx", d => d.x || 0)
                            .attr("cy", d => d.y || 0)
                            .attr("r", 1);
                        
                        // Get bounds of nodes
                        const bounds = tempG.node().getBBox();
                        tempG.remove();
                        
                        // Calculate scaling factor to fit all nodes
                        const scale = 0.9 / Math.max(
                            bounds.width / width,
                            bounds.height / height
                        );
                        
                        // Calculate center of the graph
                        const centerX = bounds.x + bounds.width / 2;
                        const centerY = bounds.y + bounds.height / 2;
                        
                        // Create a transform that centers and scales
                        const transform = d3.zoomIdentity
                            .translate(width / 2, height / 2)
                            .scale(scale)
                            .translate(-centerX, -centerY);
                        
                        // Apply the transform with transition
                        svg.transition()
                            .duration(750)
                            .call(d3.zoom().transform, transform);
                    }} catch (e) {{
                        console.error("Error resetting zoom:", e);
                        
                        // Fallback to a simpler zoom reset
                        const svg = d3.select("#graph");
                        svg.transition()
                            .duration(750)
                            .call(d3.zoom().transform, d3.zoomIdentity);
                    }}
                }}
                
                // Toggle fullscreen for the graph
                function toggleFullscreen() {{
                    const container = document.querySelector(".graph-container");
                    
                    if (!document.fullscreenElement) {{
                        if (container.requestFullscreen) {{
                            container.requestFullscreen();
                        }} else if (container.webkitRequestFullscreen) {{
                            container.webkitRequestFullscreen();
                        }} else if (container.msRequestFullscreen) {{
                            container.msRequestFullscreen();
                        }}
                    }} else {{
                        if (document.exitFullscreen) {{
                            document.exitFullscreen();
                        }} else if (document.webkitExitFullscreen) {{
                            document.webkitExitFullscreen();
                        }} else if (document.msExitFullscreen) {{
                            document.msExitFullscreen();
                        }}
                    }}
                }}
                
                // Initialize the graph when the document is loaded
                document.addEventListener('DOMContentLoaded', initGraph);
                
                // Also initialize immediately in case DOMContentLoaded already fired
                initGraph();
            </script>
            """
            
            # Increase component height for the taller graph
            st.components.v1.html(html_content, height=1350)

# Math Assessment Tab
with tab_math_assessment:
    st.header("Math Problem Assessment")
    
    if math_agent is None:
        st.error("Math Assessment service is not available. Please check your configuration.")
    else:
        # Left column: Input
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🔢 Math Problem")
            
            # Form for math problem
            with st.form(key="math_form"):
                math_question = st.text_area(
                    "Question:",
                    placeholder="Enter a math problem or question...",
                    height=100,
                    value=st.session_state.get('math_question', '')
                )
                
                student_answer = st.text_input(
                    "Your Answer:",
                    placeholder="Enter your answer..."
                )
                
                # Make correct answer field hidden by default
                show_correct = st.checkbox("I want to provide a correct answer (optional)", value=False)
                
                if show_correct:
                    correct_answer = st.text_input(
                        "Correct Answer (Optional):",
                        placeholder="For verification purposes...",
                        value=st.session_state.get('correct_answer', '')
                    )
                else:
                    correct_answer = st.session_state.get('correct_answer', '')
                
                submit_button = st.form_submit_button(label="✓ Check Answer")
            
            # Sample problems
            st.markdown("### Try These Sample Problems:")
            sample_problems_container = st.container()
            
            with sample_problems_container:
                sample_cols = st.columns(3)
                sample_problems = {
                    "Basic Addition": {
                        "question": "What is 5 + 3?",
                        "answer": "8"
                    },
                    "Equation Solving": {
                        "question": "Solve for x in the equation 2x + 5 = 15",
                        "answer": "x = 5"
                    },
                    "Derivative": {
                        "question": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
                        "answer": "3x^2 + 4x - 5"
                    }
                }
                
                for i, (name, problem) in enumerate(sample_problems.items()):
                    with sample_cols[i % 3]:
                        if st.button(f"📝 {name}", key=f"sample_{i}"):
                            st.session_state['math_question'] = problem['question']
                            st.session_state['correct_answer'] = problem['answer']
                            st.rerun()
            
            st.markdown('<div class="math-topics-container" style="margin-top: 20px;">', unsafe_allow_html=True)
            st.markdown("### Topic Areas:")
            topics = ["Algebra", "Calculus", "Geometry", "Statistics", "Trigonometry", "Number Theory"]
            topic_cols = st.columns(3)
            
            for i, topic in enumerate(topics):
                with topic_cols[i % 3]:
                    st.markdown(f"""
                    <div style="padding: 10px; background-color: {accent_color}22; border-radius: 5px; margin-bottom: 10px; text-align: center;">
                        {topic}
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Right column: Feedback and progress
        with right_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Feedback")
            
            # Process the math problem if form is submitted
            if submit_button and math_question and student_answer:
                with st.spinner("Analyzing your answer..."):
                    try:
                        # Initialize state for the math agent
                        state = {
                            "question": math_question,
                            "student_answer": student_answer,
                            "correct_answer": correct_answer if correct_answer else None,
                            "analysis": {},
                            "feedback": {},
                            "hint_count": 0,
                            "hints": [],
                            "needs_hint": False
                        }
                        
                        # Process the math question
                        result_state = math_agent.analyze(state)
                        
                        # Get the results
                        is_correct = result_state.get("feedback", {}).get("math", {}).get("is_correct", False)
                        feedback = result_state.get("feedback", {}).get("math", {}).get("assessment", "No feedback available")
                        feedback_confidence = result_state.get("feedback", {}).get("math", {}).get("confidence", 0.7)
                        analysis = result_state.get("analysis", {})
                        analysis_confidence = 0.8  # Default confidence if not provided
                        
                        # Check if there's a confidence value in the analysis
                        if "confidence" in analysis:
                            analysis_confidence = analysis["confidence"]
                        
                        hints = result_state.get("hints", [])
                        hint_confidence = 0.75  # Default confidence for hints
                        if "hint_confidence" in result_state:
                            hint_confidence = result_state["hint_confidence"]
                        
                        proximity_score = result_state.get("feedback", {}).get("math", {}).get("proximity_score")
                        
                        # Store in session state
                        if 'problems_attempted' not in st.session_state:
                            st.session_state['problems_attempted'] = 0
                        if 'problems_correct' not in st.session_state:
                            st.session_state['problems_correct'] = 0
                        
                        st.session_state['problems_attempted'] += 1
                        if is_correct:
                            st.session_state['problems_correct'] += 1
                        
                        st.session_state['current_hints'] = hints
                        st.session_state['hint_index'] = 0
                        st.session_state['current_problem'] = {
                            "question": math_question,
                            "student_answer": student_answer,
                            "correct_answer": correct_answer,
                            "is_correct": is_correct,
                            "feedback": feedback,
                            "feedback_confidence": feedback_confidence,
                            "proximity_score": proximity_score,
                            "analysis": analysis,
                            "analysis_confidence": analysis_confidence,
                            "hint_confidence": hint_confidence
                        }
                    except Exception as e:
                        st.error(f"Error analyzing math problem: {str(e)}")
                        st.info("This might be due to a configuration issue or an API service limitation.")
            
            # Display feedback if available
            if 'current_problem' in st.session_state:
                problem = st.session_state['current_problem']
                
                # Display question and answer summary
                st.markdown(f"""
                <div style="margin-bottom: 15px; padding: 15px; background-color: {accent_color}11; border-radius: 8px;">
                    <strong>Question:</strong> {problem['question']}<br>
                    <strong>Your answer:</strong> {problem['student_answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Display correctness status with animation
                if problem['is_correct']:
                    st.markdown("""
                    <div class="correct-answer" style="animation: fadeIn 0.5s ease-in-out;">
                        <span style="font-size: 24px;">✓</span> Correct!
                    </div>
                    <style>
                        @keyframes fadeIn {
                            from { opacity: 0; transform: translateY(-10px); }
                            to { opacity: 1; transform: translateY(0); }
                        }
                    </style>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="incorrect-answer" style="animation: shake 0.5s ease-in-out;">
                        <span style="font-size: 24px;">✗</span> Not quite right
                    </div>
                    <style>
                        @keyframes shake {
                            0%, 100% { transform: translateX(0); }
                            25% { transform: translateX(-5px); }
                            75% { transform: translateX(5px); }
                        }
                    </style>
                    """, unsafe_allow_html=True)
                
                # Display analysis if available
                if 'analysis' in problem and problem['analysis']:
                    st.markdown("### Analysis:")
                    display_math_analysis(problem['analysis'], problem['analysis_confidence'])
                
                # Display feedback
                st.markdown("### Feedback:")
                display_math_feedback(problem['feedback'], problem['feedback_confidence'])
                
                # Display proximity score if available
                if problem['proximity_score'] is not None:
                    score_value = min(problem['proximity_score'] / 10, 1.0)
                    st.markdown("### How Close You Were:")
                    st.progress(score_value)
                    
                    # Add labels to the progress bar
                    progress_label = "Far Off" if score_value < 0.3 else ("Getting There" if score_value < 0.7 else "Very Close")
                    st.caption(f"Proximity Score: {problem['proximity_score']}/10 - {progress_label}")
                
                # Display hints with an expandable section
                if 'current_hints' in st.session_state and st.session_state['current_hints']:
                    st.markdown("### Need Help?")
                    
                    # Show any previously displayed hints
                    if st.session_state['hint_index'] > 0:
                        hints_shown = st.session_state['current_hints'][:st.session_state['hint_index']]
                        hint_confidence = problem.get('hint_confidence', None)
                        display_math_hints(hints_shown, hint_confidence)
                    
                    # Button to get more hints
                    if st.session_state['hint_index'] < len(st.session_state['current_hints']):
                        hint_col1, hint_col2 = st.columns([3, 1])
                        with hint_col1:
                            hint_text = f"Get hint #{st.session_state['hint_index']+1} of {len(st.session_state['current_hints'])}"
                            st.write(hint_text)
                        with hint_col2:
                            if st.button("💡 Get Hint", key="hint_button"):
                                hint = st.session_state['current_hints'][st.session_state['hint_index']]
                                st.markdown(f'''
                                <div class="hint" style="animation: fadeIn 0.5s ease-in-out;">
                                    <strong>Hint #{st.session_state["hint_index"]+1}:</strong> {hint}
                                </div>
                                <style>
                                    @keyframes fadeIn {{
                                        from {{ opacity: 0; transform: translateY(-5px); }}
                                        to {{ opacity: 1; transform: translateY(0); }}
                                    }}
                                </style>
                                ''', unsafe_allow_html=True)
                                
                                # Display confidence for this hint
                                hint_confidence = problem.get('hint_confidence', None)
                                if hint_confidence is not None:
                                    display_confidence_badge(hint_confidence)
                                    
                                st.session_state['hint_index'] += 1
                    else:
                        st.info("You've seen all available hints. If you still need help, try asking a follow-up question in chat mode.")
            else:
                # Display placeholder when no problem has been submitted
                st.info("Submit a math problem to get feedback")
                
                # Add some helpful information
                st.markdown("""
                ### How It Works:
                1. Enter a math problem in the form
                2. Provide your answer
                3. Get instant feedback
                4. Request hints if needed
                5. Track your progress over time
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Chat follow-up section with toggle
            st.markdown('<div class="card" style="margin-top: 20px;">', unsafe_allow_html=True)
            st.subheader("💬 Follow-up Questions")
            
            # Initialize session state for interaction mode if not present
            if 'interaction_mode' not in st.session_state:
                st.session_state['interaction_mode'] = 'structured'
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []
            
            # Add toggle for interaction mode
            interaction_mode_toggle = st.toggle(
                "Enable Chat Mode",
                value=st.session_state['interaction_mode'] == 'chat',
                help="Toggle between structured feedback mode and chat-based follow-up questions"
            )
            
            # Update interaction mode in session state based on toggle
            st.session_state['interaction_mode'] = 'chat' if interaction_mode_toggle else 'structured'
            
            if st.session_state['interaction_mode'] == 'chat':
                st.markdown(f"""
                <div style="padding: 10px; background-color: {accent_color}15; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid {accent_color};">
                    <p style="margin-bottom: 0;"><strong>Chat Mode:</strong> You can ask follow-up questions about the problem.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Only show chat if a problem has been submitted
                if 'current_problem' in st.session_state:
                    # Display chat history if available
                    chat_container = st.container()
                    with chat_container:
                        # Display existing chat messages
                        for message in st.session_state.get('chat_history', []):
                            if message['role'] == 'student':
                                st.markdown(f"""
                                <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                                    <div style="background-color: {correct_color}15; padding: 10px; border-radius: 10px; max-width: 80%; 
                                               border: 1px solid {correct_color}33; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                                        <p style="margin-bottom: 0; color: {text_color};">{message['message']}</p>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:  # tutor
                                st.markdown(f"""
                                <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                                    <div style="background-color: {accent_color}11; padding: 10px; border-radius: 10px; max-width: 80%;
                                               border: 1px solid {accent_color}22; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                                        <p style="margin-bottom: 0; color: {text_color};">{message['message']}</p>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Input for new follow-up question
                    with st.form(key="follow_up_form"):
                        follow_up_question = st.text_input(
                            "Ask a follow-up question:",
                            placeholder="e.g., Can you explain why my answer was wrong?",
                            key="follow_up_input"
                        )
                        follow_up_button = st.form_submit_button(label="Send", type="primary")
                    
                    if follow_up_button and follow_up_question:
                        with st.spinner("Processing your question..."):
                            try:
                                # Get current problem state
                                problem = st.session_state['current_problem']
                                
                                # Initialize state for the math agent
                                state = {
                                    "question": problem['question'],
                                    "student_answer": problem['student_answer'],
                                    "correct_answer": problem.get('correct_answer'),
                                    "analysis": problem.get('analysis', {}),
                                    "feedback": {
                                        "math": {
                                            "assessment": problem['feedback'],
                                            "is_correct": problem['is_correct'],
                                            "proximity_score": problem.get('proximity_score', 5),
                                            "confidence": problem.get('feedback_confidence', 0.7)
                                        }
                                    },
                                    "hint_count": len(st.session_state.get('current_hints', [])),
                                    "hints": st.session_state.get('current_hints', []),
                                    "needs_hint": False,
                                    "interaction_mode": "chat",
                                    "chat_history": st.session_state.get('chat_history', [])
                                }
                                
                                # Process the follow-up question
                                result_state = math_agent.handle_follow_up(state, follow_up_question)
                                
                                # Get the response
                                chat_response = result_state.get("chat_response", "I'm not sure how to answer that.")
                                
                                # Update chat history in session state
                                st.session_state['chat_history'] = result_state.get("chat_history", [])
                                
                                # Rerun to show the updated chat
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error processing follow-up question: {str(e)}")
                else:
                    st.info("Submit a math problem first to enable follow-up questions")
            else:
                st.markdown(f"""
                <div style="padding: 10px; background-color: {warning_color}15; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid {warning_color};">
                    <p style="margin-bottom: 0;"><strong>Structured Mode:</strong> Receive standard feedback and hints based on your answers.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show hint request in structured mode
                if 'current_problem' in st.session_state:
                    if st.button("📝 Request Hint"):
                        with st.spinner("Generating hint..."):
                            try:
                                # Get current problem state
                                problem = st.session_state['current_problem']
                                
                                # Initialize state for the math agent
                                state = {
                                    "question": problem['question'],
                                    "student_answer": problem['student_answer'],
                                    "correct_answer": problem.get('correct_answer'),
                                    "analysis": {},
                                    "feedback": {
                                        "math": {
                                            "assessment": problem['feedback'],
                                            "is_correct": problem['is_correct'],
                                            "proximity_score": problem.get('proximity_score', 5)
                                        }
                                    },
                                    "hint_count": len(st.session_state.get('current_hints', [])) + 1,
                                    "hints": st.session_state.get('current_hints', []),
                                    "needs_hint": True,
                                    "interaction_mode": "structured"
                                }
                                
                                # Process the hint request
                                result_state = math_agent.analyze(state)
                                
                                # Update hints in session state
                                st.session_state['current_hints'] = result_state.get("hints", [])
                                
                                # Rerun to show the updated hints
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error generating hint: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Progress card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📈 Your Progress")
            
            # Display progress metrics
            problems_attempted = st.session_state.get('problems_attempted', 0)
            problems_correct = st.session_state.get('problems_correct', 0)
            
            progress_cols = st.columns(2)
            with progress_cols[0]:
                st.metric("Problems Attempted", problems_attempted)
            with progress_cols[1]:
                st.metric("Problems Correct", problems_correct)
            
            if problems_attempted > 0:
                mastery_percentage = int((problems_correct / problems_attempted) * 100)
                
                # Show progress bar with animation
                st.markdown(f"""
                <div style="margin-top: 15px;">
                    <p style="margin-bottom: 5px;">Mastery Level:</p>
                    <div style="height: 20px; background-color: {accent_color}33; border-radius: 10px; overflow: hidden;">
                        <div style="width: {mastery_percentage}%; height: 100%; background-color: {accent_color}; 
                                 border-radius: 10px; animation: growWidth 1s ease-out;">
                        </div>
                    </div>
                    <p style="text-align: right; margin-top: 5px;">{mastery_percentage}%</p>
                </div>
                <style>
                    @keyframes growWidth {{
                        from {{ width: 0%; }}
                        to {{ width: {mastery_percentage}%; }}
                    }}
                </style>
                """, unsafe_allow_html=True)
                
                # Show mastery level label
                if mastery_percentage < 30:
                    mastery_label = "Beginner"
                elif mastery_percentage < 60:
                    mastery_label = "Intermediate"
                elif mastery_percentage < 80:
                    mastery_label = "Advanced"
                else:
                    mastery_label = "Expert"
                
                st.info(f"Current level: {mastery_label}")
            
            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<footer>
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <p>© 2025 Educational Technology Platform | Made with ❤️ for learning</p>
            <p style="font-size: 0.8rem; margin-top: 10px;">Powered by Streamlit and OpenAI</p>
        </div>
        <div style="text-align: right; max-width: 400px; margin-left: 20px;">
            <p style="color: {accent_color}; font-weight: 600;">About Confidence Indicators</p>
            <p style="font-size: 0.9rem;">This platform uses confidence ratings to help you understand the reliability of feedback, hints, and analysis.</p>
        </div>
    </div>
    <div style="margin-top: 15px; display: flex; justify-content: center; gap: 20px;">
        <span style="color: {correct_color}; font-size: 0.8rem;">■ High Confidence</span>
        <span style="color: {warning_color}; font-size: 0.8rem;">■ Medium Confidence</span>
        <span style="color: {incorrect_color}; font-size: 0.8rem;">■ Low Confidence</span>
    </div>
</footer>
""", unsafe_allow_html=True) 