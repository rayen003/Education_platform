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
import traceback
import re

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

# Add required JavaScript libraries for the knowledge graph and consolidate all CSS styles
st.markdown("""
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js" integrity="sha512-XzMD7bQ/70XsQvjsUxXAOOD1qzLWgW4iAoEbsRGlbIOZXgWxogujXYOaI8xFLpcGDzfZbMc1EvnWN8J9tDXIjg==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" integrity="sha512-8dHWqQTVmYQzEWDgVGUxM4p0QraNZZKQrlDQCWIpjJY2OBtiCh2qk9/jffV+GfZGCk7oLVVxHBJYWR1FFLOIQ==" crossorigin="anonymous" referrerpolicy="no-referrer" />

<style>
/* Additional styles for knowledge graph */
#knowledge-graph {
    height: 100%;
    width: 100%;
    background-color: #f9f9f9;
    border-radius: 10px;
}

/* Chat interface styles */
.message {
    margin-bottom: 15px;
    padding: 12px;
    border-radius: 18px;
    animation: fadeIn 0.3s ease-in-out;
    max-width: 85%;
    line-height: 1.4;
}

.tutor-message {
    background-color: #f0f0f0;
    margin-right: auto;
    border-bottom-left-radius: 5px;
}

.student-message {
    background-color: #1e88e5;
    color: white;
    margin-left: auto;
    text-align: right;
    border-bottom-right-radius: 5px;
}

.message-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 5px;
    font-size: 0.9rem;
}

.message-content {
    line-height: 1.5;
    color: #0A1626 !important;
    font-weight: 500;
}

.student-message .message-content {
    color: white !important;
}

.timestamp {
    color: #666;
    font-size: 0.8rem;
    margin-top: 5px;
    opacity: 0.7;
    text-align: right;
}

.action-buttons {
    display: flex;
    gap: 8px;
    margin-top: 12px;
    flex-wrap: wrap;
}

.hint-button, .reason-button, .potential-question-button {
    background-color: #f4f4f4;
    border: 1px solid #ddd;
    border-radius: 18px;
    padding: 8px 16px;
    margin: 5px;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.2s;
    display: inline-block;
    font-weight: 500;
}

.hint-button {
    background-color: #4caf5022;
    border-color: #4caf50;
    color: #4caf50;
}

.hint-button:hover {
    background-color: #4caf5044;
}

.reason-button {
    background-color: #ff980022;
    border-color: #ff9800;
    color: #ff9800;
}

.reason-button:hover {
    background-color: #ff980044;
}

.potential-question-button {
    background-color: #2196f322;
    border-color: #2196f3;
    color: #2196f3;
}

.potential-question-button:hover {
    background-color: #2196f344;
}

.reasoning-container {
    margin-top: 20px;
    padding: 15px;
    background-color: #f9f9f9;
    border-radius: 10px;
    border-left: 4px solid #ff9800;
}

.reasoning-step {
    margin-bottom: 12px;
    padding: 10px;
    position: relative;
    border-left: 3px solid #ff9800;
    background-color: #fff8e1;
    border-radius: 4px;
}

.reasoning-step:before {
    content: "•";
    position: absolute;
    left: 0;
    color: #ff9800;
    font-weight: bold;
}

/* Expander styling for reasoning */
.streamlit-expanderHeader {
    background-color: #fff8e1 !important;
    border: 1px solid #ff9800 !important;
    border-radius: 8px !important;
    color: #e65100 !important;
    font-weight: 600 !important;
    transition: all 0.2s ease;
    margin-bottom: 10px !important;
}

.streamlit-expanderHeader:hover {
    background-color: #fff4d3 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
}

/* Chat message styling improvements */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 15px;
    padding: 15px;
    max-height: 500px;
    overflow-y: auto;
    margin-bottom: 20px;
    border-radius: 10px;
    background-color: #f9f9f9;
}

.chat-message {
    padding: 12px 18px;
    border-radius: 18px;
    position: relative;
    max-width: 85%;
    font-size: 15px;
    line-height: 1.5;
    margin: 4px 0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    animation: fadeIn 0.3s;
}

.user-message {
    background-color: #1e88e5;
    color: white;
    align-self: flex-end;
    border-bottom-right-radius: 4px;
    margin-left: auto;
}

.tutor-message {
    background-color: #f1f1f1;
    color: #212121;
    align-self: flex-start;
    border-bottom-left-radius: 4px;
    margin-right: auto;
}

.chat-input-container {
    margin-top: 20px;
    background-color: white;
    border-radius: 24px;
    padding: 5px;
    display: flex;
    border: 1px solid #ddd;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

.chat-input {
    flex-grow: 1;
    border: none !important;
    padding: 10px 15px !important;
    border-radius: 20px !important;
    font-size: 1rem !important;
}

.chat-submit {
    background-color: #1e88e5 !important;
    color: white !important;
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    margin: 5px !important;
    border: none !important;
}

/* Make send button more visible */
[data-testid="baseButton-secondary"]:has(div:contains("Send")) {
    background-color: #1e88e5 !important;
    color: white !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Main container for the entire chat UI */
.chat-interface {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 150px);
    background-color: #111111;
    border-radius: 12px;
    overflow: hidden;
    max-width: 1000px;
    margin: 0 auto;
    position: relative;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}

/* Message display area with scrolling */
.message-area {
    flex-grow: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
}

/* Input area - fixed at bottom */
.input-area {
    position: sticky;
    bottom: 0;
    padding: 16px;
    background-color: #1e1e1e;
    border-top: 1px solid #333;
    z-index: 10;
}

/* Chat messages with improved styling */
.chat-message {
    padding: 12px 16px;
    border-radius: 18px;
    margin-bottom: 15px;
    font-size: 16px;
    line-height: 1.5;
    max-width: 85%;
    animation: fadeIn 0.3s ease;
    word-wrap: break-word;
}

/* Student/user messages - enhanced style */
.user-message {
    background-color: #0D8BF0;
    color: white;
    margin-left: auto;
    margin-right: 0;
    border-bottom-right-radius: 4px;
}

/* Tutor/assistant messages - enhanced style */
.tutor-message {
    background-color: #2D2D2D;
    color: #f0f0f0;
    margin-right: auto;
    margin-left: 0;
    border-bottom-left-radius: 4px;
}

/* Timestamp - subtle styling */
.timestamp {
    font-size: 12px;
    opacity: 0.6;
    margin-top: 5px;
    text-align: right;
    color: #aaa;
}

/* Button styles - smaller and more icon-centric */
.action-buttons {
    display: flex;
    gap: 10px;
    margin-top: 10px;
    justify-content: flex-end;
}

/* Input styling */
.input-container {
    display: flex;
    border-radius: 8px;
    background: #333333;
    margin-bottom: 10px;
    overflow: hidden;
    position: relative;
}

/* Custom styling for the textarea */
textarea {
    border: none !important;
    background-color: transparent !important;
    color: white !important;
    resize: none !important;
    padding: 12px 16px !important;
    font-size: 16px !important;
    caret-color: white;
    box-sizing: border-box !important;
    width: 100% !important;
    height: 70px !important;
    min-height: 70px !important;
    line-height: 1.5 !important;
}

textarea:focus {
    border: none !important;
    box-shadow: none !important;
}

/* Fix for Streamlit containers */
.stTextArea {
    background-color: transparent !important;
}

.stTextArea > div {
    background-color: transparent !important;
    border: none !important;
}

/* Hide Streamlit's default elements for textarea */
[data-testid="stTextAreaContainer"] {
    background-color: transparent !important;
    border: none !important;
}

.stTextArea [data-baseweb="base-input"] {
    background-color: transparent !important;
    border: none !important;
}

.css-1eqtdef {
    opacity: 0 !important;
}

/* Mobile responsiveness improvements */
@media screen and (max-width: 768px) {
    .chat-interface {
        height: calc(100vh - 120px);
    }
    
    .input-container {
        flex-direction: column;
    }
    
    textarea {
        min-height: 70px !important;
    }
    
    .action-buttons {
        justify-content: space-around;
    }
}

/* Make scrollbar more subtle */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: #555;
    border-radius: 10px;
}

/* Math formatting */
.math-equation {
    margin: 10px 0;
    padding: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow-x: auto;
}

/* Solution styles */
.key-concepts {
    margin: 10px 0;
    padding: 10px;
    background-color: rgba(13, 139, 240, 0.1);
    border-radius: 6px;
}

.key-concepts ul {
    margin-top: 8px;
    padding-left: 20px;
}

.step-solution {
    margin: 15px 0;
}

.final-answer {
    margin-top: 20px;
    padding: 10px;
    background-color: rgba(40, 167, 69, 0.1);
    border-radius: 6px;
}

/* Fix button styles for Streamlit */
[data-testid="baseButton-secondary"] {
    border-radius: 50% !important;
    min-width: 40px !important;
    width: 40px !important;
    height: 40px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Specific button styling */
.hint-btn [data-testid="baseButton-secondary"] {
    background-color: #1d6f30 !important;
    color: white !important;
}

.feedback-btn [data-testid="baseButton-secondary"] {
    background-color: #0d47a1 !important;
    color: white !important;
}

.solution-btn [data-testid="baseButton-secondary"] {
    background-color: #880e4f !important;
    color: white !important;
}

.cot-btn [data-testid="baseButton-secondary"] {
    background-color: #5d4037 !important;
    color: white !important;
}

.send-btn [data-testid="baseButton-secondary"] {
    background-color: #007bff !important;
    color: white !important;
    width: 48px !important;
    height: 48px !important;
}
</style>
""", unsafe_allow_html=True)

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

# Set color scheme and theme variables
accent_color = "#3174ad"
module_color = "#5a9bd5"
concept_color = "#7cb9e8"
text_color = "#333333"  # Darker text color for better contrast

def generate_graph_html(graph_data, show_labels=True, enable_physics=False, height=700, spacing=200, focus_script=""):
    """Generate HTML for the knowledge graph visualization."""
    
    # Set default colors
    node_color = "#5D8BF4"
    edge_color = "#9DB2BF"
    resource_color = "#2E7D32"  # Define resource_color here for Learning Resources heading
    
    if not graph_data or not graph_data.get('nodes'):
        # Return empty div if no graph data
        return f"""
        <div id="knowledge-graph" style="height: {height}px; width: 100%; background-color: #f5f5f5; display: flex; align-items: center; justify-content: center;">
            <p style="color: #666;">No knowledge graph data available</p>
        </div>
        """
    
    # Ensure nodes and links exist
    nodes = graph_data.get('nodes', [])
    links = graph_data.get('links', [])
    
    # Prepare data for vis.js
    vis_nodes = []
    for node in nodes:
        node_type = node.get('type', 'concept')
        
        # Set node appearance based on type
        shape = "circle"
        if node_type == "module":
            shape = "box"
        
        # Add to nodes array
        vis_nodes.append({
            'id': node.get('id'),
            'label': node.get('label', ''),
            'title': node.get('title', node.get('label', '')),
            'shape': shape,
            'color': "#5B8AF0" if node_type == "concept" else "#FFA500",
            'font': {'size': 14, 'face': 'Arial', 'color': 'white' if node_type == "concept" else "black"},
            'data': {
                'description': node.get('description', ''),
                'module': node.get('module', ''),
                'connections': node.get('connections', 0),
                'centrality': node.get('centrality', 0),
                'resources': node.get('resources', [])
            }
        })
    
    # Prepare edges
    vis_edges = []
    for link in links:
        confidence = link.get('metadata', {}).get('confidence', 0.7)
        relation_type = link.get('type', 'related')
        
        # Format edges based on relationship type and confidence
        vis_edges.append({
            'from': link.get('source'),
            'to': link.get('target'),
            'width': 1 + confidence * 3,  # Width based on confidence
            'arrows': 'to' if not link.get('bidirectional', False) else 'to;from',
            'color': {'color': edge_color, 'opacity': 0.7 + confidence * 0.3},
            'title': relation_type.capitalize(),
            'data': {
                'reasoning': link.get('metadata', {}).get('reasoning', ''),
                'evidence': link.get('metadata', {}).get('evidence', [])
            }
        })
    
    # Convert to JSON for JavaScript
    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)
    
    # Include the focus script if provided
    focus_code = focus_script if focus_script else ""
    
    # Create HTML with vis.js
    return f"""
    <div id="knowledge-graph" style="height: {height}px; width: 100%;"></div>
    
    <!-- Add required libraries -->
    <script src="https://cdn.jsdelivr.net/npm/vis-network/standalone/umd/vis-network.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/vis-network@9.1.2/dist/dist/vis-network.min.css" rel="stylesheet" type="text/css" />
    
    <script>
    (function() {{
        // Create a container
        var container = document.getElementById('knowledge-graph');
        
        // Parse the data
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});
        
        // Create the network data
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        
        // Set options
        var options = {{
            nodes: {{
                shape: 'dot',
                size: 25,
                font: {{
                    size: 14,
                    face: 'Arial'
                }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                width: 2,
                smooth: {{'type': 'continuous'}},
                shadow: true,
                font: {{
                    size: 12,
                    face: 'Arial',
                    align: 'middle'
                }},
                color: {{
                    inherit: false,
                    color: '{edge_color}',
                    opacity: 0.8
                }}
            }},
            physics: {{
                enabled: {str(enable_physics).lower()},
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -100,
                    centralGravity: 0.1,
                    springLength: {spacing},
                    springConstant: 0.08,
                    damping: 0.4
                }},
                stabilization: {{
                    enabled: true,
                    iterations: 1000,
                    fit: true
                }}
            }},
            interaction: {{
                navigationButtons: true,
                keyboard: true,
                hover: true,
                multiselect: false,
                tooltipDelay: 100
            }}
        }};
        
        // Create the network
        setTimeout(function() {{
            // Wait for DOM to be ready
            var network = new vis.Network(container, data, options);
            
            // Set up node click event
            network.on("click", function(params) {{
                if (params.nodes.length > 0) {{
                    var nodeId = params.nodes[0];
                    var node = nodes.get(nodeId);
                    
                    // Update node details in the sidebar
                    var detailsContainer = document.getElementById('node-details-container');
                    var resourcesContainer = document.getElementById('node-resources-container');
                    
                    if (detailsContainer) {{
                        // Update node details
                        var detailsHtml = `
                            <h3>${{node.label}}</h3>
                            <p>${{node.data.description || 'No description available'}}</p>
                            <p><strong>Module:</strong> ${{node.data.module || 'N/A'}}</p>
                            <p><strong>Connections:</strong> ${{node.data.connections}}</p>
                        `;
                        detailsContainer.innerHTML = detailsHtml;
                    }}
                    
                    if (resourcesContainer) {{
                        // Update resources
                        var resources = node.data.resources || [];
                        var resourcesHtml = `<h4 style="margin-top: 0; color: {resource_color}; font-weight: 600;">Learning Resources</h4>`;
                        
                        if (resources.length > 0) {{
                            resourcesHtml += '<ul style="list-style-type: none; padding: 0;">';
                            resources.forEach(function(resource) {{
                                resourcesHtml += `
                                <li style="margin-bottom: 12px; padding: 10px; background-color: #f7f7f7; border-radius: 5px;">
                                    <strong>${{resource.title || 'Untitled Resource'}}</strong>
                                    <p style="margin: 5px 0; font-size: 0.9em;">${{resource.description || 'No description'}}</p>
                                    <a href="${{resource.url || '#'}}" target="_blank" style="color: {resource_color}; text-decoration: none; font-size: 0.9em; font-weight: 500;">Access Resource →</a>
                                </li>`;
                            }});
                            resourcesHtml += '</ul>';
                        }} else {{
                            resourcesHtml += '<p>No resources available for this concept.</p>';
                        }}
                        
                        resourcesContainer.innerHTML = resourcesHtml;
                    }}
                }}
            }});
            
            {focus_code}
        }}, 100);
    }})();
    </script>
    """

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

# Main page content
st.markdown(f"""
<div style="text-align: center; padding: 20px 0 40px;">
    <h1>Educational Technology Platform</h1>
    <p style="font-size: 1.2rem; max-width: 800px; margin: 0 auto;">
        An intelligent learning platform with knowledge graph visualization and math assessment capabilities.
    </p>
</div>
""", unsafe_allow_html=True)

# Replace the grid with just 2 main components
st.markdown("""
<div style="margin-bottom: 30px;">
    <h2 style="margin-bottom: 25px;">Available Services</h2>
</div>
""", unsafe_allow_html=True)

# Initialize app navigation in session state if not already there
if 'current_mode' not in st.session_state:
    st.session_state['current_mode'] = 'home'

# Create 2 columns for our main services
col1, col2 = st.columns(2)

with col1:
    knowledge_graph_card = st.container()
    with knowledge_graph_card:
        st.markdown("""
        <div style="background-color: #e6f0fa; border-radius: 20px; padding: 30px; text-align: center; height: 200px; 
                   display: flex; flex-direction: column; justify-content: center; align-items: center; margin-bottom: 20px;
                   box-shadow: 0 4px 10px rgba(0,0,0,0.1); cursor: pointer;">
            <div style="width: 80px; height: 80px; background-color: #b9d8f6; border-radius: 50%; 
                       display: flex; justify-content: center; align-items: center; margin-bottom: 15px;">
                <span style="font-size: 40px;">📊</span>
            </div>
            <h3 style="font-size: 1.8rem; margin: 0; color: #3174ad;">Knowledge Graph</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Open Knowledge Graph", key="kg_button"):
            st.session_state['current_mode'] = 'knowledge_graph'
            st.rerun()

with col2:
    math_assessment_card = st.container()
    with math_assessment_card:
        st.markdown("""
        <div style="background-color: #faf6e6; border-radius: 20px; padding: 30px; text-align: center; height: 200px; 
                   display: flex; flex-direction: column; justify-content: center; align-items: center; margin-bottom: 20px;
                   box-shadow: 0 4px 10px rgba(0,0,0,0.1); cursor: pointer;">
            <div style="width: 80px; height: 80px; background-color: #f6e9b9; border-radius: 50%; 
                       display: flex; justify-content: center; align-items: center; margin-bottom: 15px;">
                <span style="font-size: 40px;">🔢</span>
            </div>
            <h3 style="font-size: 1.8rem; margin: 0; color: #c49c16;">Math Assessment</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Open Math Assessment", key="math_button"):
            st.session_state['current_mode'] = 'math_assessment'
            st.rerun()

# Show home page instructions when no service is selected
if st.session_state['current_mode'] == 'home':
    st.markdown("""
    <div style="text-align: center; padding: 50px 0; color: #666;">
        <h3>👆 Select a service above to get started</h3>
        <p>Choose Knowledge Graph to visualize educational content or Math Assessment for interactive problem solving.</p>
    </div>
    """, unsafe_allow_html=True)

# Hide tabs by default - we'll show content based on current_mode
tab_knowledge_graph, tab_math_assessment = st.tabs(["📊 Knowledge Graph", "🔢 Math Assessment"])

# Hide tabs completely when using our new navigation
st.markdown("""
<style>
    [data-testid="stTabs"] {display: none !important;}
</style>
""", unsafe_allow_html=True)

# Knowledge Graph mode
if st.session_state['current_mode'] == 'knowledge_graph':
    st.header("Knowledge Graph Visualization")
    
    # Back button
    if st.button("← Back to Home", key="kg_back_btn"):
        st.session_state['current_mode'] = 'home'
        st.rerun()
    
    # Create a three-column layout with a wider middle column for the graph
    # and narrower columns for controls and resources
    control_col, graph_col, resources_col = st.columns([1, 4, 1.5])
    
    # Control panel on the left side
    with control_col:
        st.markdown(f"""
        <div style="background-color: {accent_color}22; padding: 15px; border-radius: 10px; 
                  margin-bottom: 15px; color: {text_color}; border-left: 4px solid {accent_color};">
            <h3 style="margin-top: 0; margin-bottom: 15px; color: {accent_color}; 
                    font-size: 1.3rem; font-weight: 600;">Graph Controls</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualization controls
        enable_physics = st.checkbox("Enable Physics", value=True, 
                                    help="Turn physics simulation on/off. When on, nodes will move dynamically.")
        
        show_labels = st.checkbox("Show Labels", value=True,
                                help="Show or hide node labels")
        
        # Layout adjustments
        st.markdown("#### Layout")
        layout_spacing = st.slider("Node Spacing", 100, 300, 200, 10,
                                 help="Adjust spacing between nodes")
        
        # Search functionality
        st.markdown("#### Search")
        search_term = st.text_input("Search Nodes", 
                                  placeholder="Enter concept or module name")
        
        if search_term and 'current_graph' in st.session_state:
            graph_data = st.session_state['current_graph']
            nodes = graph_data.get('nodes', [])
            matching_nodes = [n for n in nodes if search_term.lower() in n.get('label', '').lower()]
            
            if matching_nodes:
                st.success(f"Found {len(matching_nodes)} matching nodes")
                for node in matching_nodes[:5]:  # Show top 5 matches
                    if st.button(f"🔍 {node.get('label')}", key=f"search_{node.get('id')}"):
                        # Create and store JavaScript code to focus on this node
                        st.session_state['focus_node_id'] = node.get('id')
                        st.rerun()
            else:
                st.warning("No matching nodes found")
        
        # Filter by module
        if 'current_graph' in st.session_state:
            graph_data = st.session_state['current_graph']
            modules = [n.get('label') for n in graph_data.get('nodes', []) 
                     if n.get('type') == 'module']
            
            st.markdown("#### Filter by Module")
            selected_module = st.selectbox("Select Module", 
                                         ["All Modules"] + modules,
                                         index=0)
    
    with graph_col:
        # Input section - make it more compact
        with st.expander("Upload Syllabus", expanded=False):
            syllabus_text = st.text_area(
                "Paste your syllabus text below:",
                height=150,
                placeholder="Course Title: Introduction to Mathematics...",
                value=st.session_state.get('syllabus_text', '')
            )
            
                if st.button("🔍 Generate Knowledge Graph", key="generate_graph_btn"):
                    if syllabus_text:
                        with st.spinner("Generating knowledge graph... This may take a minute."):
                            try:
                            # Store syllabus text in session state
                            st.session_state['syllabus_text'] = syllabus_text
                            
                                # Create a container to show the progress
                                status_container = st.empty()
                                status_container.info("Step 1/4: Processing syllabus text...")
                                
                                # Process the syllabus and generate a graph
                                result = kg_service.process_syllabus(syllabus_text, user_id="streamlit_user")
                                
                            # Check if the graph was generated successfully
                                if result.get('status') == 'success':
                                graph_id = result.get('graph_id')
                                status_container.info(f"Step 2/4: Graph generated with ID: {graph_id}")
                                
                                # Get the graph data
                                status_container.info("Step 3/4: Retrieving graph data...")
                                graph_data = kg_service.get_graph(graph_id)
                                
                                if 'error' not in graph_data:
                                    status_container.info("Step 4/4: Graph data retrieved successfully!")
                                    
                                    # Add metadata if not present
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
                                        
                                else:
                                    st.error(f"Error retrieving graph data: {graph_data.get('error')}")
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
            
        # Display the graph with more vertical space - give it much more height
        if 'current_graph' in st.session_state:
            graph_data = st.session_state['current_graph']
            
            # Apply module filter if selected
            if 'selected_module' in locals() and selected_module != "All Modules":
                # Filter nodes to only show the selected module and its related concepts
                module_node = next((n for n in graph_data.get('nodes', []) 
                                  if n.get('label') == selected_module), None)
                
                if module_node:
                    module_id = module_node.get('id')
                    # Get all links involving this module
                    related_links = [l for l in graph_data.get('links', []) 
                                   if l.get('source') == module_id or l.get('target') == module_id]
                    # Get IDs of all related nodes
                    related_node_ids = set()
                    related_node_ids.add(module_id)
                    for link in related_links:
                        related_node_ids.add(link.get('source'))
                        related_node_ids.add(link.get('target'))
                    
                    # Filter nodes and links
                    filtered_nodes = [n for n in graph_data.get('nodes', []) 
                                    if n.get('id') in related_node_ids]
                    filtered_links = related_links
                    
                    # Create a temporary filtered graph
                    filtered_graph = {
                        'nodes': filtered_nodes,
                        'links': filtered_links,
                        'metadata': graph_data.get('metadata', {})
                    }
                    
                    # Use the filtered graph
                    display_graph = filtered_graph
                else:
                    display_graph = graph_data
            else:
                display_graph = graph_data
            
            # Check if we need to focus on a specific node
            focus_script = ""
            if 'focus_node_id' in st.session_state:
                node_id = st.session_state['focus_node_id']
                focus_script = f"""
                // Focus on the searched node
                setTimeout(function() {{
                    var nodeId = "{node_id}";
                    network.selectNodes([nodeId]);
                    network.focus(nodeId, {{
                        scale: 1.5,
                        animation: true
                    }});
                    // Clear the selection after focusing
                    setTimeout(function() {{ st.session_state.focus_node_id = null; }}, 100);
                }}, 1000);
                """
                # Clear the focus node ID
                st.session_state['focus_node_id'] = None
            
            # Make the graph container much taller (800px)
            st.markdown(f"""
            <div style="height: 800px; border-radius: 15px; overflow: hidden; margin-top: 10px; 
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                {generate_graph_html(display_graph, show_labels, enable_physics, 
                                     height=800, spacing=layout_spacing, focus_script=focus_script)}
            </div>
            
            <!-- Add vis.js directly to ensure it's loaded -->
            <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            """, unsafe_allow_html=True)
            
            # Debug: Let's log some information about the graph
            if display_graph:
                node_count = len(display_graph.get('nodes', []))
                link_count = len(display_graph.get('links', []))
                st.markdown(f"<div style='color: #888; font-size: 0.8rem;'>Debug: Graph has {node_count} nodes and {link_count} links</div>", 
                            unsafe_allow_html=True)
        else:
            st.info("No graph data available. Generate a graph or load sample data.")
            if st.button("Load Sample Data"):
                try:
                    from app.knowledge_graph.api_adapter import KnowledgeGraphService
                    temp_service = KnowledgeGraphService(mock_mode=True)
                    with open('app/static/data/sample_syllabus.txt', 'r') as f:
                            sample_syllabus = f.read()
                    graph_data = temp_service._generate_mock_graph(sample_syllabus)
                    st.session_state['current_graph'] = graph_data
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading sample data: {str(e)}")
        
    # Resources panel on the right side with colored backgrounds instead of white
    with resources_col:
        if 'current_graph' in st.session_state:
            graph_data = st.session_state['current_graph']
            
            # Display course metadata with accent-colored background
                st.markdown(f"""
            <div style="background-color: {accent_color}22; padding: 15px; border-radius: 10px; 
                      margin-bottom: 15px; color: {text_color}; border-left: 4px solid {accent_color};">
                <h3 style="margin-top: 0; margin-bottom: 15px; color: {accent_color}; 
                        font-size: 1.3rem; font-weight: 600;">Course Information</h3>
                <div style="line-height: 1.5;">
                    <p><strong>Title:</strong> {graph_data.get('metadata', {}).get('course_title', 'Untitled Course')}</p>
                    <p><strong>Modules:</strong> {len([n for n in graph_data.get('nodes', []) if n.get('type') == 'module'])}</p>
                    <p><strong>Concepts:</strong> {len([n for n in graph_data.get('nodes', []) if n.get('type') == 'concept'])}</p>
                    <p><strong>Relationships:</strong> {len(graph_data.get('links', []))}</p>
                </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Selected node details with colored background
                st.markdown(f"""
            <div style="background-color: {module_color}22; padding: 15px; border-radius: 10px; 
                      margin-bottom: 15px; color: {text_color}; border-left: 4px solid {module_color};">
                <h3 style="margin-top: 0; margin-bottom: 15px; color: {module_color}; 
                        font-size: 1.3rem; font-weight: 600;">Node Details</h3>
                <div id="node-details-container" style="line-height: 1.5;">
                    <p style="color: #666; font-style: italic;">Click on a node in the graph to see details</p>
                </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Learning resources with colored background
                st.markdown(f"""
            <div style="background-color: {concept_color}22; padding: 15px; border-radius: 10px; 
                      color: {text_color}; border-left: 4px solid {concept_color};">
                <h3 style="margin-top: 0; margin-bottom: 15px; color: {concept_color}; 
                        font-size: 1.3rem; font-weight: 600;">Learning Resources</h3>
                <div id="node-resources-container" style="line-height: 1.5;">
                    <p style="color: #666; font-style: italic;">Select a node to view related resources</p>
                </div>
                </div>
                """, unsafe_allow_html=True)
            
# Math Assessment mode
if st.session_state['current_mode'] == 'math_assessment':
    # Add custom CSS for ChatGPT-like interface with improved aesthetics
    st.markdown("""
    <style>
    /* Main container for the entire chat UI */
    .chat-interface {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 150px);
        background-color: #111111;
        border-radius: 12px;
        overflow: hidden;
        max-width: 1000px;
        margin: 0 auto;
        position: relative;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    
    /* Message display area with scrolling */
    .message-area {
        flex-grow: 1;
        overflow-y: auto;
        padding: 20px;
        display: flex;
        flex-direction: column;
    }
    
    /* Input and actions area - fixed at bottom */
    .input-area {
        position: sticky;
        bottom: 0;
        padding: 16px;
        background-color: #1e1e1e;
        border-top: 1px solid #333;
        z-index: 10;
    }
    
    /* Chat messages with improved styling */
    .chat-message {
        padding: 12px 16px;
        border-radius: 18px;
        margin-bottom: 15px;
        font-size: 16px;
        line-height: 1.5;
        max-width: 85%;
        animation: fadeIn 0.3s ease;
        word-wrap: break-word;
    }
    
    /* Student/user messages - enhanced style */
    .user-message {
        background-color: #0D8BF0;
        color: white;
        margin-left: auto;
        margin-right: 0;
        border-bottom-right-radius: 4px;
    }
    
    /* Tutor/assistant messages - enhanced style */
    .tutor-message {
        background-color: #2D2D2D;
        color: #f0f0f0;
        margin-right: auto;
        margin-left: 0;
        border-bottom-left-radius: 4px;
    }
    
    /* Timestamp - subtle styling */
    .timestamp {
        font-size: 12px;
        opacity: 0.6;
        margin-top: 5px;
        text-align: right;
        color: #aaa;
    }
    
    /* Button styles - smaller and more icon-centric */
    .action-buttons {
        display: flex;
        gap: 10px;
        margin-top: 10px;
        justify-content: flex-end;
    }
    
    /* Input styling */
    .input-container {
        display: flex;
        border-radius: 8px;
        background: #333333;
        margin-bottom: 10px;
        overflow: hidden;
        position: relative;
    }
    
    /* Structured response styling */
    .response-section {
        margin-top: 15px;
    }
    
    .section-title {
        font-weight: 600;
        font-size: 18px;
        margin-bottom: 10px;
    }
    
    .step-item {
        margin-bottom: 10px;
        padding: 8px 12px;
        background-color: rgba(255, 255, 255, 0.08);
        border-radius: 6px;
    }
    
    /* CoT panel styling */
    .cot-panel {
        margin: 10px 0;
        background: #262626;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #444;
        cursor: pointer;
    }
    
    .cot-header {
        padding: 12px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-weight: 600;
    }
    
    .cot-content {
        padding: 16px;
        border-top: 1px solid #444;
    }
    
    .step {
        padding: 10px 12px;
        margin-bottom: 12px;
        background-color: #333;
        border-left: 3px solid #0D8BF0;
        border-radius: 0 6px 6px 0;
        animation: stepIn 0.4s ease forwards;
        opacity: 0;
    }
    
    /* Confidence bar */
    .confidence-bar {
        height: 4px;
        background-color: #444;
        border-radius: 2px;
        overflow: hidden;
        margin: 8px 0 15px 0;
    }
    
    .confidence-level {
        height: 100%;
        border-radius: 2px;
        transition: width 0.5s ease;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes stepIn {
        from { 
            opacity: 0; 
            transform: translateY(10px);
        }
        to { 
            opacity: 1; 
            transform: translateY(0);
        }
    }
    
    /* Hide default Streamlit elements we don't need */
    .stTextArea [data-baseweb=base-input] {
        background-color: transparent !important;
    }
    
    .css-1eqtdef {
        opacity: 0 !important;
    }
    
    /* Custom styling for the textarea */
    textarea {
        border: none !important;
        background-color: transparent !important;
        color: white !important;
        resize: none !important;
        padding: 12px 16px !important;
        font-size: 16px !important;
        caret-color: white;
        box-sizing: border-box !important;
        width: 100% !important;
        height: 70px !important;
        min-height: 70px !important;
        line-height: 1.5 !important;
    }
    
    textarea:focus {
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Fix for Streamlit containers */
    .stTextArea {
        background-color: transparent !important;
    }
    
    .stTextArea > div {
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Hide Streamlit's default elements for textarea */
    [data-testid="stTextAreaContainer"] {
        background-color: transparent !important;
        border: none !important;
    }
    
    .stTextArea [data-baseweb="base-input"] {
        background-color: transparent !important;
        border: none !important;
    }
    
    .css-1eqtdef {
        opacity: 0 !important;
    }
    
    /* Mobile responsiveness improvements */
    @media screen and (max-width: 768px) {
        .chat-interface {
            height: calc(100vh - 120px);
        }
        
        .input-container {
            flex-direction: column;
        }
        
        textarea {
            min-height: 70px !important;
        }
        
        .action-buttons {
            justify-content: space-around;
        }
    }
    
    /* Make scrollbar more subtle */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #555;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("Math Problem Solving")
    
    # Back button
    if st.button("← Back to Home", key="math_back_btn"):
        st.session_state['current_mode'] = 'home'
        st.rerun()
    
    # Initialize session state variables if not already set
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'current_problem' not in st.session_state:
        st.session_state['current_problem'] = ""
    if 'chat_mode' not in st.session_state:
        st.session_state['chat_mode'] = True  # Set chat mode to True by default
    if 'show_reasoning' not in st.session_state:
        st.session_state['show_reasoning'] = False
    if 'reasoning_steps' not in st.session_state:
        st.session_state['reasoning_steps'] = []
    if 'cot_loading' not in st.session_state:
        st.session_state['cot_loading'] = False
    if 'step_indices' not in st.session_state:
        st.session_state['step_indices'] = []
    
    # Create math agent if not already done
    if 'math_agent' not in st.session_state:
        try:
            from app.math_services.agent.math_agent import MathAgent
            st.session_state['math_agent'] = MathAgent()
        except Exception as e:
            st.error(f"Error initializing Math Agent: {str(e)}")
    
    math_agent = st.session_state.get('math_agent')
    
    # Add necessary session state variables for better flow control
    if 'problem_submitted' not in st.session_state:
        st.session_state['problem_submitted'] = False
    if 'answer_submitted' not in st.session_state:
        st.session_state['answer_submitted'] = False  
    if 'solution_revealed' not in st.session_state:
        st.session_state['solution_revealed'] = False
    if 'hint_count' not in st.session_state:
        st.session_state['hint_count'] = 0
    if 'feedback_generated' not in st.session_state:
        st.session_state['feedback_generated'] = False
    if 'answer_is_correct' not in st.session_state:
        st.session_state['answer_is_correct'] = None

    # Main chat container
    st.markdown('<div class="chat-interface">', unsafe_allow_html=True)
    
    # Message area
    st.markdown('<div class="message-area">', unsafe_allow_html=True)
    
    # Display chat history
    if st.session_state['chat_history']:
        for message in st.session_state['chat_history']:
            role = message.get('role', 'system')
            content = message.get('message', '')
            confidence = message.get('confidence', 0.7)
            timestamp = message.get('timestamp', '')
            
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        # Extract hour and minute
                        time_display = timestamp.split()[1][:5]
                    else:
                        time_display = timestamp.strftime("%H:%M")
                except:
                    time_display = ""
            else:
                time_display = ""
            
            if role.lower() == 'student':
                # Student message (blue, right-aligned)
                message_html = f"""
                <div class="chat-message user-message">
                    {content.replace("<", "&lt;").replace(">", "&gt;")}
                    <div class="timestamp">{time_display}</div>
                </div>
                """
                st.markdown(message_html, unsafe_allow_html=True)
            else:
                # Format the tutor message with structured sections if it has certain patterns
                formatted_content = content.replace("<", "&lt;").replace(">", "&gt;")
                
                # Check if the message contains solution explanation with steps
                if "**Explanation:**" in content and "Step" in content:
                    # Format step by step solution with proper structure
                    try:
                        parts = content.split("**Explanation:**", 1)
                        header = parts[0].replace("*", "").replace("**", "")
                        steps_content = parts[1] if len(parts) > 1 else ""
                        
                        # Process the "Step X:" parts
                        if "*Step" in steps_content:
                            steps_list = ""
                            for step in steps_content.split("*Step")[1:]:
                                step_text = step.split("*", 1)[1] if "*" in step else step
                                steps_list += f'<div class="step-item">Step {step_text}</div>'
                        else:
                            steps_list = steps_content
                        
                        formatted_content = f"""
                        {header}
                        <div class="response-section">
                            <div class="section-title">Explanation:</div>
                            <div class="step-solution">
                                {steps_list}
                            </div>
                        </div>
                        """
                    except:
                        # If parsing fails, fall back to the original content
                        pass
                
                # Check for "Key Concepts" section
                elif "Key Concepts" in content:
                    # Try to structure the key concepts section
                    try:
                        if "Final Answer" in content:
                            # Split into three main sections
                            before_key, key_content = content.split("Key Concepts", 1)
                            key_and_steps, final_answer = key_content.split("Final Answer", 1)
                            
                            formatted_content = f"""
                            {before_key.strip()}
                            
                            <div class="section-title">Key Concepts and Formulas</div>
                            <div class="key-concepts">
                                {key_and_steps}
                            </div>
                            
                            <div class="section-title">Final Answer</div>
                            <div class="final-answer">
                                {final_answer}
                            </div>
                            """
                        else:
                            formatted_content = content
                    except:
                        # If parsing fails, fall back to original
                        pass
                
                # Tutor message (dark gray, left-aligned)
                message_html = f"""
                <div class="chat-message tutor-message">
                    {formatted_content}
                    <div class="timestamp">{time_display}</div>
                </div>
                """
                st.markdown(message_html, unsafe_allow_html=True)
                
                # Show confidence indicator for tutor messages
                if confidence:
                    # Determine color based on confidence level
                    color = "#28A745" if confidence > 0.8 else "#FFC107" if confidence > 0.5 else "#DC3545"
                    confidence_html = f"""
                    <div style="display: flex; align-items: center; margin-bottom: 15px; margin-left: 20px; max-width: 170px;">
                        <div class="confidence-bar">
                            <div class="confidence-level" style="width: {int(confidence * 100)}%; background-color: {color};"></div>
                        </div>
                        <span style="font-size: 12px; color: #999; margin-left: 8px;">{int(confidence * 100)}% confidence</span>
                    </div>
                    """
                    st.markdown(confidence_html, unsafe_allow_html=True)
    
    # End message area
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area at the bottom
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    
    # Chain of Thought button
    cot_col1, cot_col2 = st.columns([9, 1])
    with cot_col1:
        # Create an expander-like element that's controlled by our own click handler
        cot_expanded = st.session_state.get('show_reasoning', False)
        
        # Generate Chain of Thought content
        cot_html = f"""
        <div class="cot-panel" id="cot-panel" onclick="toggleCoT()">
            <div class="cot-header">
                <div>🧠 Chain of Thought Reasoning</div>
                <div>{'▼' if cot_expanded else '▶'}</div>
            </div>
        """
        
        # Always display the concise DoT reasoning
        if st.session_state.get('reasoning_steps', []):
            # Get the shorter DoT steps or use the regular steps if DoT isn't available
            dot_steps = []
            if 'math_agent' in globals() and math_agent and math_agent.current_state:
                if hasattr(math_agent.current_state, 'context') and math_agent.current_state.context:
                    dot_steps = math_agent.current_state.context.get('dot_reasoning_steps', [])
                if not dot_steps and hasattr(math_agent.current_state, 'steps'):
                    dot_steps = math_agent.current_state.steps
            
            if not dot_steps:
                dot_steps = st.session_state.get('reasoning_steps', [])
                
                # Limit to just a few steps for conciseness if there are many
                if len(dot_steps) > 4:
                    dot_steps = [dot_steps[0], dot_steps[len(dot_steps)//2], dot_steps[-1]]
            
            # Display the concise steps (these are always visible)
            cot_html += '<div class="dot-content" style="padding: 10px 15px;">'
            for i, step in enumerate(dot_steps):
                cot_html += f"""
                <div class="step">
                    <strong>Step {i+1}:</strong> {step.replace("<", "&lt;").replace(">", "&gt;")}
                </div>
                """
            cot_html += '</div>'
        
        if cot_expanded:
            cot_html += '<div class="cot-content">'
            
            # Get the full CoT reasoning steps
            full_steps = []
            if 'math_agent' in globals() and math_agent and math_agent.current_state:
                if hasattr(math_agent.current_state, 'context') and math_agent.current_state.context:
                    full_steps = math_agent.current_state.context.get('full_reasoning_steps', [])
            
            if not full_steps:
                full_steps = st.session_state.get('reasoning_steps', [])
            
            # Only show the full steps if they're different from what's already shown
            if full_steps and (not dot_steps or len(full_steps) > len(dot_steps)):
                cot_html += '<div style="border-top: 1px solid #eee; margin-top: 10px; padding-top: 10px;">'
                cot_html += '<h4>Full Detailed Reasoning:</h4>'
                
                for i, step in enumerate(full_steps):
                    # Calculate a delay for the animation based on index
                    delay = i * 0.1
                    cot_html += f"""
                    <div class="step" style="animation-delay: {delay}s;"">
                        <strong>Step {i+1}:</strong> {step.replace("<", "&lt;").replace(">", "&gt;")}
                    </div>
                    """
                cot_html += '</div>'
            elif not st.session_state.get('reasoning_steps', []):
                if st.session_state.get('cot_loading', False):
                    cot_html += '<div style="text-align: center; padding: 20px;">Generating detailed reasoning...</div>'
                else:
                    cot_html += '<div style="text-align: center; padding: 20px;">No reasoning steps available yet.</div>'
            
            cot_html += '</div>'
        
        cot_html += '</div>'
        
        # Add JavaScript to handle the click
        cot_html += """
        <script>
        function toggleCoT() {
            // This will be handled by a button click below
            const coTButton = document.querySelector('[data-testid="baseButton-secondary"]:has(div:contains("🧩"))');
            if (coTButton) {
                coTButton.click();
            }
        }
        </script>
        """
        
        st.markdown(cot_html, unsafe_allow_html=True)
    
    with cot_col2:
        # Hidden button that will be clicked by JavaScript
        cot_btn = st.button("🧠", key="cot_toggle_btn")
        st.markdown("""
        <style>
        [data-testid="baseButton-secondary"]:has(div:contains("🧠")) {
            opacity: 0;
            position: absolute;
            pointer-events: none;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if cot_btn:
            # Toggle the reasoning state
            st.session_state['show_reasoning'] = not st.session_state.get('show_reasoning', False)
            
            # If we're showing reasoning and don't have steps yet, generate them
            if st.session_state['show_reasoning'] and not st.session_state.get('reasoning_steps', []) and not st.session_state.get('cot_loading', False):
                st.session_state['cot_loading'] = True
                st.rerun()  # Rerun to show loading state
    
    # Execute CoT generation if in loading state
    if st.session_state.get('cot_loading', False) and math_agent and math_agent.current_state:
        try:
            # Set up context for CoT
            if not math_agent.current_state.context:
                math_agent.current_state.context = {}
            math_agent.current_state.context['reasoning_mode'] = 'cot'
            
            updated_state = math_agent.process_interaction('button', 'reasoning', math_agent.current_state)
            
            # Store reasoning steps if available
            if hasattr(updated_state, 'steps') and updated_state.steps:
                st.session_state['reasoning_steps'] = updated_state.steps
                st.session_state['step_indices'] = list(range(len(updated_state.steps)))
                
            # Update current state
            math_agent.current_state = updated_state
            
            # Clear loading state
            st.session_state['cot_loading'] = False
            st.rerun()
        except Exception as e:
            st.error(f"Error generating reasoning: {str(e)}")
            st.session_state['cot_loading'] = False
    
    # Add JavaScript for keyboard shortcuts and auto-focus
    st.markdown("""
    <script>
    // Add event listener for keyboard shortcuts and Enter key press
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            const textareas = document.querySelectorAll('textarea');
            const submitButtons = document.querySelectorAll('button');
            let chatTextarea;
            let sendButton, hintButton, feedbackButton, solutionButton, reasoningButton;
            
            // Find the chat textarea and buttons
            textareas.forEach(function(textarea) {
                if (textarea.placeholder && textarea.placeholder.includes('Type a math problem or question here')) {
                    chatTextarea = textarea;
                }
            });
            
            submitButtons.forEach(function(button) {
                if (button.innerText.includes('📤')) {
                    sendButton = button;
                } else if (button.innerText.includes('💡')) {
                    hintButton = button;
                } else if (button.innerText.includes('📝')) {
                    feedbackButton = button;
                } else if (button.innerText.includes('🎯')) {
                    solutionButton = button;
                } else if (button.innerText.includes('🧩')) {
                    reasoningButton = button;
                }
            });
            
            // Auto-focus on the textarea
            if (chatTextarea) {
                chatTextarea.focus();
                
                // Add event listener for Enter key
                chatTextarea.addEventListener('keydown', function(e) {
                    // Check if Enter was pressed without Shift (Shift+Enter for new line)
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        if (sendButton) {
                            sendButton.click();
                        }
                    }
                });
            }
            
            // Add global keyboard shortcuts
            document.addEventListener('keydown', function(e) {
                // Only apply shortcuts when not typing in textarea
                if (document.activeElement !== chatTextarea) {
                    if (e.key === 'h' || e.key === 'H') {
                        if (hintButton) {
                            e.preventDefault();
                            hintButton.click();
                        }
                    } else if (e.key === 'f' || e.key === 'F') {
                        if (feedbackButton) {
                            e.preventDefault();
                            feedbackButton.click();
                        }
                    } else if (e.key === 's' || e.key === 'S') {
                        if (solutionButton) {
                            e.preventDefault();
                            solutionButton.click();
                        }
                    } else if (e.key === 'r' || e.key === 'R') {
                        if (reasoningButton) {
                            e.preventDefault();
                            reasoningButton.click();
                        }
                    }
                }
            });
        }, 1000); // Small delay to ensure elements are loaded
    });
                </script>
    """, unsafe_allow_html=True)
    
    # Handle input clearing when needed
    if st.session_state.get('_clear_chat_input', False):
        # Clear the flag
        st.session_state['_clear_chat_input'] = False
        input_value = ""
    else:
        input_value = st.session_state.get('_input_value', "")
    
    # Chat input field with buttons
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    input_placeholder = "Type a math problem or question here..."
    user_input = st.text_area("", value=input_value, placeholder=input_placeholder, height=70, key="chat_input", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Store the current input value
    st.session_state['_input_value'] = user_input
    
    # Action buttons row with conditional display
    st.markdown('<div class="action-buttons">', unsafe_allow_html=True)
    
    # Using columns for button layout with better proportions for alignment
    button_cols = st.columns([1, 1, 1, 1, 1])
    
    with button_cols[0]:
        # Only show hint button if solution hasn't been revealed
        if not st.session_state.get('solution_revealed', False):
            st.markdown('<div class="hint-btn">', unsafe_allow_html=True)
            hint_btn = st.button("💡", key="icon_hint_btn", help="Get a hint for this problem (Shortcut: H)")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Placeholder to maintain spacing
            st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
    
    with button_cols[1]:
        # Only show feedback button if a problem has been submitted
        if st.session_state.get('problem_submitted', False):
            st.markdown('<div class="feedback-btn">', unsafe_allow_html=True)
            feedback_btn = st.button("📝", key="icon_feedback_btn", help="Get feedback on your answer (Shortcut: F)")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
            # Placeholder to maintain spacing
            st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
    
    with button_cols[2]:
        # Only show solution button if a problem has been submitted
        if st.session_state.get('problem_submitted', False):
            st.markdown('<div class="solution-btn">', unsafe_allow_html=True)
            solution_btn = st.button("🎯", key="icon_solution_btn", help="Show the complete solution (Shortcut: S)")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Placeholder to maintain spacing
            st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
    
    with button_cols[3]:
        # Only show reasoning button if a problem has been submitted
        if st.session_state.get('problem_submitted', False):
            st.markdown('<div class="cot-btn">', unsafe_allow_html=True)
            reasoning_btn = st.button("🧩", key="icon_reasoning_btn", help="Show step-by-step reasoning (Shortcut: R)")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Placeholder to maintain spacing
            st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
    
    with button_cols[4]:
        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        submit_chat = st.button("📤", key="icon_send_btn", help="Send message (Shortcut: Enter)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process button actions
    # Hint button
    if hint_btn and math_agent and math_agent.current_state and st.session_state.get('problem_submitted', False):
        with st.spinner("Generating hint..."):
            try:
                # Increment hint count
                st.session_state['hint_count'] += 1
                
                # Use the process_interaction method with button type
                updated_state = math_agent.process_interaction('button', 'hint', math_agent.current_state)
                
                # Extract hint - only get the latest hint
                hint = "I don't have any hints for this problem yet."
                if hasattr(updated_state, 'hints') and updated_state.hints:
                    hint = updated_state.hints[-1]  # Just get the most recent hint
                
                # Add to chat history
                from datetime import datetime
                st.session_state['chat_history'].append({
                    'role': 'tutor',
                    'message': f"💡 **Hint #{st.session_state['hint_count']}:** {hint}",
                    'timestamp': datetime.now()
                })
                
                # Update current state
                math_agent.current_state = updated_state
                            st.rerun()
            except Exception as e:
                st.error(f"Error generating hint: {str(e)}")
    
    # Feedback button
    if feedback_btn and math_agent and math_agent.current_state and st.session_state.get('problem_submitted', False):
        with st.spinner("Generating feedback..."):
            try:
                # Use the process_interaction method with button type
                updated_state = math_agent.process_interaction('button', 'feedback', math_agent.current_state)
                
                # Extract feedback
                feedback_message = "I couldn't generate detailed feedback for this problem."
                if hasattr(updated_state, 'feedback') and updated_state.feedback:
                    if isinstance(updated_state.feedback, dict):
                        feedback_message = updated_state.feedback.get('detail', '')
                        if not feedback_message:
                            feedback_message = updated_state.feedback.get('assessment', '')
                    else:
                        # Use getattr since feedback might be an object, not a dictionary
                        feedback_message = getattr(updated_state.feedback, 'detail', '')
                        if not feedback_message and hasattr(updated_state.feedback, 'assessment'):
                            feedback_message = getattr(updated_state.feedback, 'assessment', '')
                
                # Get the confidence score safely
                confidence_score = 0.7  # Default confidence
                if hasattr(updated_state, 'feedback'):
                    if isinstance(updated_state.feedback, dict):
                        confidence_score = updated_state.feedback.get('confidence', 0.7)
                    else:
                        confidence_score = getattr(updated_state.feedback, 'confidence', 0.7)
                
                # Add to chat history
                from datetime import datetime
                st.session_state['chat_history'].append({
                    'role': 'tutor',
                    'message': f"📝 **Feedback:** {feedback_message}",
                    'confidence': confidence_score,
                    'timestamp': datetime.now()
                })
                
                # Update current state
                math_agent.current_state = updated_state
                
                # Mark feedback as generated
                st.session_state['feedback_generated'] = True
                
                st.rerun()
            except Exception as e:
                st.error(f"Error generating feedback: {str(e)}")
    
    # Solution button
    if solution_btn and math_agent and math_agent.current_state and st.session_state.get('problem_submitted', False):
        with st.spinner("Generating solution..."):
            try:
                # Use the process_interaction method with button type
                updated_state = math_agent.process_interaction('button', 'solution', math_agent.current_state)
                
                # Add to chat history if there's a chat response
                solution_text = "Here's the solution:\n\n"
                if hasattr(updated_state, 'correct_answer') and updated_state.correct_answer:
                    solution_text += f"**Answer:** {updated_state.correct_answer}\n\n"
                
                # Include reasoning steps if available
                if hasattr(updated_state, 'steps') and updated_state.steps:
                    st.session_state['reasoning_steps'] = updated_state.steps
                    solution_text += "**Explanation:**\n\n"
                    for i, step in enumerate(updated_state.steps):
                        solution_text += f"*Step {i+1}:* {step}\n\n"
                
                from datetime import datetime
                st.session_state['chat_history'].append({
                    'role': 'tutor',
                    'message': solution_text,
                    'timestamp': datetime.now()
                })
                
                # Update current state
                math_agent.current_state = updated_state
                
                # Mark solution as revealed (disables hint button)
                st.session_state['solution_revealed'] = True
                
                st.rerun()
                    except Exception as e:
                st.error(f"Error generating solution: {str(e)}")
    
    # Reasoning button
    if reasoning_btn and st.session_state.get('problem_submitted', False):
        # Toggle reasoning visibility and generate if needed
        st.session_state['show_reasoning'] = True
        if not st.session_state.get('reasoning_steps', []) and not st.session_state.get('cot_loading'):
            st.session_state['cot_loading'] = True
        st.rerun()
    
    # Submit button (main chat processing)
    if submit_chat and user_input:
        with st.spinner("Processing..."):
            # Add user message to chat history
            from datetime import datetime
            st.session_state['chat_history'].append({
                'role': 'student',
                'message': user_input,
                'timestamp': datetime.now()
            })
            
            # Process based on whether we have a current problem or not
            if st.session_state.get('current_problem'):
                # This could be either a follow-up question or an answer submission
                
                # Check if we're waiting for an answer (problem submitted but answer not yet submitted)
                if st.session_state.get('problem_submitted', False) and not st.session_state.get('answer_submitted', False):
                    try:
                        # This is likely an answer submission
                        if math_agent and math_agent.current_state:
                            # Set the student answer in the state
                            math_agent.current_state.student_answer = user_input
                            
                            # First, analyze the answer
                            updated_state = math_agent.analyze(math_agent.current_state)
                            
                            # Determine if the answer is correct
                            is_correct = False
                            if hasattr(updated_state, 'analysis') and updated_state.analysis:
                                if isinstance(updated_state.analysis, dict):
                                    is_correct = updated_state.analysis.get('is_correct', False)
                                else:
                                    is_correct = getattr(updated_state.analysis, 'is_correct', False)
                            
                            # Store correctness in session state
                            st.session_state['answer_is_correct'] = is_correct
                            
                            # Then generate feedback automatically
                            updated_state = math_agent.generate_feedback(updated_state)
                            
                            # Extract feedback message
                            feedback_message = "I couldn't generate detailed feedback for this problem."
                            if hasattr(updated_state, 'feedback') and updated_state.feedback:
                                if isinstance(updated_state.feedback, dict):
                                    feedback_message = updated_state.feedback.get('detail', '')
                                    if not feedback_message:
                                        feedback_message = updated_state.feedback.get('assessment', '')
            else:
                                    feedback_message = getattr(updated_state.feedback, 'detail', '')
                                    if not feedback_message and hasattr(updated_state.feedback, 'assessment'):
                                        feedback_message = getattr(updated_state.feedback, 'assessment', '')
                            
                            # Get the confidence score safely
                            confidence_score = 0.7  # Default confidence
                            if hasattr(updated_state, 'feedback'):
                                if isinstance(updated_state.feedback, dict):
                                    confidence_score = updated_state.feedback.get('confidence', 0.7)
                                else:
                                    confidence_score = getattr(updated_state.feedback, 'confidence', 0.7)
                            
                            # Add automatic feedback to chat history
                            st.session_state['chat_history'].append({
                                'role': 'tutor',
                                'message': f"📝 **Feedback:** {feedback_message}",
                                'confidence': confidence_score,
                                'timestamp': datetime.now()
                            })
                            
                            # Mark feedback as generated and answer as submitted
                            st.session_state['feedback_generated'] = True
                            st.session_state['answer_submitted'] = True
                            
                            # Store reasoning steps if available
                            if hasattr(updated_state, 'steps') and updated_state.steps:
                                st.session_state['reasoning_steps'] = updated_state.steps
                            
                            # Update current state
                            math_agent.current_state = updated_state
                            
                    except Exception as e:
                        error_msg = f"Error processing answer: {str(e)}"
                        st.error(error_msg)
                        st.session_state['chat_history'].append({
                            'role': 'tutor',
                            'message': f"I encountered an error: {str(e)}. Please try again or rephrase your answer.",
                            'timestamp': datetime.now()
                        })
                
                else:
                    # This is a follow-up question or comment
                    try:
                        # Use the process_interaction method with text type
                        if math_agent and math_agent.current_state:
                            updated_state = math_agent.process_interaction('text', user_input, math_agent.current_state)
                            
                            # Add response to chat history if available
                            if hasattr(updated_state, 'chat_response') and updated_state.chat_response:
                                st.session_state['chat_history'].append({
                                    'role': 'tutor',
                                    'message': updated_state.chat_response,
                                    'timestamp': datetime.now()
                                })
                            
                            # Store reasoning steps if available
                            if hasattr(updated_state, 'steps') and updated_state.steps:
                                st.session_state['reasoning_steps'] = updated_state.steps
                            
                            # Update current state
                            math_agent.current_state = updated_state
                    except Exception as e:
                        error_msg = f"Error processing follow-up: {str(e)}"
                        st.error(error_msg)
                        st.session_state['chat_history'].append({
                            'role': 'tutor',
                            'message': f"I encountered an error: {str(e)}. Please try again or rephrase your question.",
                            'timestamp': datetime.now()
                        })
            
                else:
                # First problem submission
                st.session_state['current_problem'] = user_input
                st.session_state['problem_submitted'] = True
                
                try:
                    # Create a MathState object
                    from app.math_services.models.state import MathState
                    math_state = MathState(question=user_input, student_answer="")
                    
                    # Solve the problem
                    if math_agent:
                        math_state = math_agent.solve(user_input)
                        
                        # Store reasoning steps if available
                        if hasattr(math_state, 'steps') and math_state.steps:
                            st.session_state['reasoning_steps'] = math_state.steps
                        
                        # Add response to chat history
                        solution_message = f"I've analyzed this problem. "
                        solution_message += "Please provide your answer, and I'll give you feedback. "
                        solution_message += "You can also ask for a hint if you need help."
                        
                        st.session_state['chat_history'].append({
                            'role': 'tutor',
                            'message': solution_message,
                            'timestamp': datetime.now()
                        })
                        
                        # Update current state in math agent
                        math_agent.current_state = math_state
                except Exception as e:
                    error_msg = f"Error processing problem: {str(e)}"
                    st.error(error_msg)
                    st.session_state['chat_history'].append({
                        'role': 'tutor',
                        'message': f"I encountered an error: {str(e)}. Please try again or rephrase your problem.",
                        'timestamp': datetime.now()
                    })
            
            # Clear input
            st.session_state['_clear_chat_input'] = True
            st.rerun()
    
    # Close input area and chat interface
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)