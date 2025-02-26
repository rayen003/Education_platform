from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import os
import re

def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route('/')
    def index():
        return render_template('knowledge_graph.html')

    @app.route('/api/knowledge_graph')
    def get_knowledge_graph():
        try:
            with open('app/knowledge_graph/microeconomics_graph.json', 'r') as f:
                data = json.load(f)
            
            # Core concepts we want to focus on
            core_concepts = [
                "Supply and Demand",
                "Market Equilibrium",
                "Price Elasticity",
                "Market Structure",
                "Perfect Competition",
                "Monopolistic Competition",
                "Oligopoly",
                "Monopoly",
                "Consumer Behavior",
                "Production Costs",
                "Economic Efficiency",
                "Market Failure"
            ]
            
            transformed_data = {
                "nodes": [],
                "links": []
            }

            # Create a mapping of original IDs to safe IDs
            id_mapping = {}
            for node in data["nodes"]:
                if node["id"] in core_concepts:  # Only include core concepts
                    safe_id = re.sub(r'[^a-zA-Z0-9]', '_', node["id"]).lower()
                    id_mapping[node["id"]] = safe_id

            # Transform nodes
            learning_paths = {
                "Fundamentals": ["Supply and Demand", "Market Equilibrium", "Price Elasticity"],
                "Market Structures": ["Market Structure", "Perfect Competition", "Monopolistic Competition", "Oligopoly", "Monopoly"],
                "Advanced Concepts": ["Consumer Behavior", "Production Costs", "Economic Efficiency", "Market Failure"]
            }

            # Add nodes with path information
            for path_name, concepts in learning_paths.items():
                for i, concept in enumerate(concepts):
                    if concept in id_mapping:
                        node = next((n for n in data["nodes"] if n["id"] == concept), None)
                        if node:
                            transformed_node = {
                                "id": id_mapping[concept],
                                "name": concept,
                                "type": get_node_type(path_name, i),
                                "description": node["description"],
                                "difficulty": node["difficulty"],
                                "learningState": "not-started",
                                "pathName": path_name,
                                "pathOrder": i,
                                "resources": [
                                    {
                                        "type": "video",
                                        "title": f"Understanding {concept}",
                                        "url": "#"
                                    },
                                    {
                                        "type": "article",
                                        "title": f"Deep Dive into {concept}",
                                        "url": "#"
                                    }
                                ]
                            }
                            transformed_data["nodes"].append(transformed_node)

            # Create directed learning path links
            for path_name, concepts in learning_paths.items():
                # Connect concepts within each path
                for i in range(len(concepts) - 1):
                    if concepts[i] in id_mapping and concepts[i + 1] in id_mapping:
                        transformed_data["links"].append({
                            "source": id_mapping[concepts[i]],
                            "target": id_mapping[concepts[i + 1]],
                            "type": "path",
                            "pathName": path_name
                        })

                # Connect paths to each other at key junction points
                if path_name == "Fundamentals":
                    # Connect Fundamentals to Market Structures
                    transformed_data["links"].append({
                        "source": id_mapping["Price Elasticity"],
                        "target": id_mapping["Market Structure"],
                        "type": "cross_path",
                        "pathName": "cross_path"
                    })
                elif path_name == "Market Structures":
                    # Connect Market Structures to Advanced Concepts
                    transformed_data["links"].append({
                        "source": id_mapping["Monopoly"],
                        "target": id_mapping["Economic Efficiency"],
                        "type": "cross_path",
                        "pathName": "cross_path"
                    })

            return jsonify(transformed_data)
        except Exception as e:
            print("Error:", str(e))
            return jsonify({"error": str(e)}), 500

    return app

def get_node_type(path_name, order):
    if path_name == "Fundamentals":
        return "core"
    elif path_name == "Market Structures":
        return "intermediate"
    else:
        return "advanced"
