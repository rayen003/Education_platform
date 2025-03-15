from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import os
import re
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from app.math_services.services.llm.openai_service import OpenAILLMService
from app.knowledge_graph.api_adapter import KnowledgeGraphService

# Import the MathAgent for math assessments
from app.math_services.agent.math_agent import MathAgent

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Initialize services with mock mode for development
    llm_service = OpenAILLMService(mock_mode=True)
    kg_service = KnowledgeGraphService(llm_service)
    math_agent = MathAgent(model="gpt-4o-mini")  # It will use the LLM service in mock mode

    @app.route('/')
    def index():
        return render_template('knowledge_graph.html')

    @app.route('/api/knowledge_graph')
    def get_knowledge_graph():
        try:
            # Try to get the graph from the service first
            graph_data = kg_service.get_graph("test_user")
            
            # If no graph exists, generate one from the sample syllabus
            if "error" in graph_data:
                try:
                    # Read the sample graph data if available
                    sample_path = os.path.join(app.static_folder, 'data', 'sample_graph_data.json')
                    if os.path.exists(sample_path):
                        with open(sample_path, 'r') as f:
                            graph_data = json.load(f)
                    else:
                        # Generate a new graph from sample syllabus
                        with open('app/static/data/sample_syllabus.txt', 'r') as f:
                            sample_syllabus = f.read()
                        result = kg_service.process_syllabus(sample_syllabus, user_id="test_user")
                        if result.get('status') == 'success':
                            graph_data = kg_service.get_graph(result.get('graph_id'))
                        else:
                            raise Exception(result.get('message', 'Unknown error processing syllabus'))
                except Exception as e:
                    return jsonify({"error": f"Failed to load sample graph: {str(e)}"}), 500
            
            return jsonify(graph_data)
        except Exception as e:
            print("Error:", str(e))
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/process_syllabus', methods=['POST'])
    def process_syllabus():
        try:
            data = request.json
            syllabus_text = data.get('syllabus', '')
            user_id = data.get('user_id', 'anonymous')
            
            if not syllabus_text:
                return jsonify({"error": "No syllabus text provided"}), 400
            
            result = kg_service.process_syllabus(syllabus_text, user_id=user_id)
            return jsonify(result)
        except Exception as e:
            print("Error processing syllabus:", str(e))
            return jsonify({"error": str(e)}), 500

    return app

def get_node_type(path_name, order):
    if path_name == "Fundamentals":
        return "core"
    elif path_name == "Market Structures":
        return "intermediate"
    else:
        return "advanced"
