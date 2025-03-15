"""
API Adapter for Knowledge Graph Service.

This module provides a simplified interface for interacting with the knowledge graph
generation services. It also includes a mock mode for testing without dependencies.
"""

import os
import json
import uuid
import time
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphService:
    """Interface for interacting with the knowledge graph generation services."""
    
    def __init__(self, llm_service=None, mock_mode=True):
        """
        Initialize the Knowledge Graph service.
        
        Args:
            llm_service: Optional LLM service for processing
            mock_mode: If True, use mock data instead of real services
        """
        self.llm_service = llm_service
        self.mock_mode = mock_mode
        self.graphs = {}  # In-memory storage for generated graphs
        
        # Create directory for JSON files if it doesn't exist
        self.json_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     "app", "static", "data", "json_files")
        os.makedirs(self.json_dir, exist_ok=True)
        
        logger.info(f"Initialized KnowledgeGraphService (mock_mode={mock_mode})")
    
    def process_syllabus(self, syllabus_text: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Process a syllabus to generate a knowledge graph.
        
        Args:
            syllabus_text: Text of the syllabus to process
            user_id: Identifier for the user requesting the graph
            
        Returns:
            Dict containing status and graph ID if successful
        """
        if not syllabus_text:
            return {"status": "error", "message": "No syllabus text provided"}
        
        logger.info(f"Processing syllabus for user {user_id} ({len(syllabus_text)} chars)")
        
        if self.mock_mode:
            # In mock mode, generate a sample graph
            graph_id = str(uuid.uuid4())
            
            # Create a mock graph based on the syllabus text
            graph_data = self._generate_mock_graph(syllabus_text)
            self.graphs[graph_id] = graph_data
            
            # Save to JSON file
            self._save_graph_to_file(graph_id, graph_data)
            
            # Simulate processing time
            time.sleep(1)
            
            return {"status": "success", "graph_id": graph_id}
        else:
            # In real mode, call the actual graph generation service
            if not self.llm_service:
                return {"status": "error", "message": "LLM service not provided"}
            
            try:
                # Generate a real graph using the provided LLM service
                graph_id = str(uuid.uuid4())
                graph_data = self._parse_syllabus_with_llm(syllabus_text)
                self.graphs[graph_id] = graph_data
                
                # Save to JSON file
                self._save_graph_to_file(graph_id, graph_data)
                
                return {"status": "success", "graph_id": graph_id}
            except Exception as e:
                logger.error(f"Error processing syllabus: {str(e)}")
                return {"status": "error", "message": str(e)}
    
    def _parse_syllabus_with_llm(self, syllabus_text: str) -> Dict[str, Any]:
        """Parse the syllabus using the LLM service to extract concepts and relationships."""
        try:
            # Extract course title
            course_title_system_prompt = "You are an expert at analyzing educational content and extracting key information."
            course_title_user_prompt = "Extract the course title from this syllabus: " + syllabus_text[:500]
            course_title_response = self.llm_service.generate_completion(
                system_prompt=course_title_system_prompt,
                user_prompt=course_title_user_prompt
            )
            course_title = course_title_response.get("content", "").strip() or "Untitled Course"
            
            # Extract modules and concepts
            modules_system_prompt = """You are an expert educational content analyzer. 
            Your task is to extract the structure of educational content and represent it as a knowledge graph."""
            
            modules_user_prompt = """Extract the key modules and concepts from this syllabus text. 
            Format your response as JSON with the following structure:
            {
                "modules": [
                    {"name": "Module Name 1", "id": "module_1"},
                    {"name": "Module Name 2", "id": "module_2"}
                ],
                "concepts": [
                    {"name": "Concept Name 1", "id": "concept_1", "module_id": "module_1"},
                    {"name": "Concept Name 2", "id": "concept_2", "module_id": "module_1"},
                    {"name": "Concept Name 3", "id": "concept_3", "module_id": "module_2"}
                ],
                "relationships": [
                    {"source": "concept_1", "target": "concept_2", "type": "prerequisite"},
                    {"source": "concept_2", "target": "concept_3", "type": "related_to"}
                ]
            }
            
            Syllabus: """ + syllabus_text
            
            # Get LLM response
            structure_response = self.llm_service.generate_completion(
                system_prompt=modules_system_prompt,
                user_prompt=modules_user_prompt
            )
            
            try:
                # Find JSON content between triple backticks if present
                response_content = structure_response.get("content", "")
                json_match = re.search(r"```(?:json)?(.*?)```", response_content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1).strip()
                else:
                    json_content = response_content
                
                # Parse the JSON
                parsed_data = json.loads(json_content)
                
                # Create nodes and links
                nodes = []
                links = []
                
                # Add modules as nodes
                for module in parsed_data.get("modules", []):
                    nodes.append({
                        "id": module["id"],
                        "name": module["name"],
                        "type": "module"
                    })
                
                # Add concepts as nodes and link them to modules
                for concept in parsed_data.get("concepts", []):
                    nodes.append({
                        "id": concept["id"],
                        "name": concept["name"],
                        "type": "concept"
                    })
                    
                    # Link concept to its module
                    if "module_id" in concept:
                        links.append({
                            "source": concept["module_id"],
                            "target": concept["id"],
                            "type": "contains",
                            "strength": 1
                        })
                
                # Add concept-to-concept relationships
                for rel in parsed_data.get("relationships", []):
                    links.append({
                        "source": rel["source"],
                        "target": rel["target"],
                        "type": rel["type"],
                        "strength": 0.7
                    })
                
                # Create metadata
                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "course_title": course_title,
                    "node_count": len(nodes),
                    "link_count": len(links),
                    "module_count": len([n for n in nodes if n["type"] == "module"]),
                    "concept_count": len([n for n in nodes if n["type"] == "concept"]),
                    "generated_by": "KnowledgeGraphService (LLM)",
                    "syllabus_length": len(syllabus_text),
                    "graph_id": str(uuid.uuid4())
                }
                
                # Return the complete graph
                return {
                    "nodes": nodes,
                    "links": links,
                    "metadata": metadata
                }
                
            except Exception as e:
                logger.error(f"Failed to parse LLM response: {str(e)}")
                # Fall back to mock graph
                return self._generate_mock_graph(syllabus_text)
                
        except Exception as e:
            logger.error(f"Error in LLM parsing: {str(e)}")
            # Fall back to mock graph
            return self._generate_mock_graph(syllabus_text)
    
    def get_graph(self, graph_id: str) -> Dict[str, Any]:
        """
        Retrieve a generated graph by ID.
        
        Args:
            graph_id: ID of the graph to retrieve
            
        Returns:
            The graph data or an empty dict if not found
        """
        if graph_id in self.graphs:
            return self.graphs[graph_id]
        
        # Try to load from file
        json_path = os.path.join(self.json_dir, f"{graph_id}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                graph_data = json.load(f)
                self.graphs[graph_id] = graph_data
                return graph_data
        
        return {}
    
    def _save_graph_to_file(self, graph_id: str, graph_data: Dict[str, Any]) -> None:
        """Save graph data to a JSON file."""
        json_path = os.path.join(self.json_dir, f"{graph_id}.json")
        with open(json_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        logger.info(f"Saved graph data to {json_path}")
    
    def _generate_mock_graph(self, syllabus_text: str) -> Dict[str, Any]:
        """
        Generate a mock knowledge graph based on the syllabus text.
        
        This is a simplified implementation that creates a basic graph structure
        by parsing the syllabus text for module and concept information.
        
        For short or non-standard syllabus formats, it generates a fallback
        demo graph to ensure something is displayed.
        """
        # Extract course title (simple regex-free approach)
        lines = syllabus_text.split('\n')
        course_title = "Unknown Course"
        for line in lines:
            if line.lower().startswith("course title:"):
                course_title = line.replace("Course Title:", "").replace("course title:", "").strip()
                break
        
        # Extract modules and concepts (simple approach)
        modules = []
        concepts = []
        links = []
        
        # Find the Modules section
        module_section = False
        current_module = None
        module_count = 0
        concept_count = 0
        learning_outcomes_section = False
        
        for line in lines:
            line = line.strip()
            
            # Detect learning outcomes section
            if "learning outcomes:" in line.lower() or "outcomes:" in line.lower():
                learning_outcomes_section = True
                module_section = False
                continue
                
            # Detect start of modules section
            if line.lower() == "modules:" or line.lower() == "modules":
                module_section = True
                learning_outcomes_section = False
                continue
            
            if not module_section and not learning_outcomes_section:
                continue
                
            # Process modules section
            if module_section:
                # Detect module entries (numbered items)
                match = re.match(r'^(\d+)[\.\s]+(.+)$', line)
                if match and match.group(1) and match.group(2):
                    module_count += 1
                    module_number = match.group(1)
                    module_name = match.group(2).strip()
                    module_id = f"module_{module_count}"
                    current_module = {"id": module_id, "name": module_name, "type": "module"}
                    modules.append(current_module)
                
                # Detect concepts (bullet points)
                elif (line.startswith("- ") or line.startswith("* ")) and current_module:
                    concept_count += 1
                    concept_name = line[2:].strip()
                    concept_id = f"concept_{concept_count}"
                    concept = {"id": concept_id, "name": concept_name, "type": "concept"}
                    concepts.append(concept)
                    
                    # Link concept to its module
                    links.append({
                        "source": current_module["id"],
                        "target": concept_id,
                        "type": "contains",
                        "strength": 1
                    })
            
            # Process learning outcomes section
            elif learning_outcomes_section:
                # Detect numbered outcomes
                match = re.match(r'^(\d+)[\.\s]+(.+)$', line)
                if match and match.group(1) and match.group(2):
                    module_count += 1
                    outcome_number = match.group(1)
                    outcome_text = match.group(2).strip()
                    outcome_id = f"outcome_{outcome_number}"
                    outcome = {"id": outcome_id, "name": outcome_text, "type": "outcome"}
                    modules.append(outcome)  # Treat outcomes as modules for visualization
        
        # Generate fallback data if no modules/concepts were found
        if not modules or not concepts:
            # Generate a simple demo graph with common educational modules and concepts
            demo_modules = [
                {"id": "module_1", "name": "Introduction", "type": "module"},
                {"id": "module_2", "name": "Core Concepts", "type": "module"},
                {"id": "module_3", "name": "Advanced Topics", "type": "module"},
                {"id": "module_4", "name": "Applications", "type": "module"}
            ]
            
            demo_concepts = [
                {"id": "concept_1", "name": "Fundamentals", "type": "concept"},
                {"id": "concept_2", "name": "Principles", "type": "concept"},
                {"id": "concept_3", "name": "Key Terminology", "type": "concept"},
                {"id": "concept_4", "name": "Theories", "type": "concept"},
                {"id": "concept_5", "name": "Techniques", "type": "concept"},
                {"id": "concept_6", "name": "Methods", "type": "concept"},
                {"id": "concept_7", "name": "Case Studies", "type": "concept"},
                {"id": "concept_8", "name": "Practical Applications", "type": "concept"}
            ]
            
            demo_links = [
                {"source": "module_1", "target": "concept_1", "type": "contains", "strength": 1},
                {"source": "module_1", "target": "concept_2", "type": "contains", "strength": 1},
                {"source": "module_2", "target": "concept_3", "type": "contains", "strength": 1},
                {"source": "module_2", "target": "concept_4", "type": "contains", "strength": 1},
                {"source": "module_3", "target": "concept_5", "type": "contains", "strength": 1},
                {"source": "module_3", "target": "concept_6", "type": "contains", "strength": 1},
                {"source": "module_4", "target": "concept_7", "type": "contains", "strength": 1},
                {"source": "module_4", "target": "concept_8", "type": "contains", "strength": 1},
                {"source": "concept_1", "target": "concept_3", "type": "prerequisite", "strength": 0.8},
                {"source": "concept_3", "target": "concept_5", "type": "builds_on", "strength": 0.7},
                {"source": "concept_2", "target": "concept_4", "type": "related_to", "strength": 0.6},
                {"source": "concept_5", "target": "concept_7", "type": "applies_to", "strength": 0.9}
            ]
            
            modules = demo_modules
            concepts = demo_concepts
            links = demo_links
            
            # Use the provided text as course title if no title was found
            if course_title == "Unknown Course" and len(syllabus_text) <= 100:
                course_title = f"Sample Course: {syllabus_text[:50]}"
        else:
            # Add some concept-to-concept links for a more interesting graph
            if len(concepts) > 1:
                for i in range(len(concepts) - 1):
                    # Link some concepts together (not all, to keep it realistic)
                    if i % 3 == 0:
                        links.append({
                            "source": concepts[i]["id"],
                            "target": concepts[i + 1]["id"],
                            "type": "related_to",
                            "strength": 0.7
                        })
        
        # Combine everything into the graph structure
        nodes = modules + concepts
        
        # Create metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "course_title": course_title,
            "node_count": len(nodes),
            "link_count": len(links),
            "module_count": len(modules),
            "concept_count": len(concepts),
            "generated_by": "KnowledgeGraphService (mock)",
            "syllabus_length": len(syllabus_text),
            "graph_id": str(uuid.uuid4())
        }
        
        # Assemble the complete graph
        graph_data = {
            "nodes": nodes,
            "links": links,
            "metadata": metadata
        }
        
        return graph_data 