"""
API Adapter Module.

This module provides adapters to connect the knowledge graph components
to the existing Flask API.
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

from app.knowledge_graph.syllabus_parser import SyllabusParser
from app.knowledge_graph.graph_generator import GraphGenerator
from app.knowledge_graph.meta_validator import MetaValidator

logger = logging.getLogger(__name__)

class KnowledgeGraphService:
    """
    Service for managing knowledge graph generation and serving via API.
    This is an adapter to maintain compatibility with the existing Flask app.
    """
    
    def __init__(self, llm_service):
        """
        Initialize the knowledge graph service.
        
        Args:
            llm_service: LLM service to use for validation and enrichment
        """
        self.llm_service = llm_service
        self.parser = SyllabusParser(llm_service)
        self.validator = MetaValidator(llm_service)
        self.graph_generator = GraphGenerator(self.validator)
        self.graph_cache = {}  # Cache for generated graphs
        logger.info("Initialized KnowledgeGraphService")
    
    def process_syllabus(self, syllabus_text: str, user_id: str = None) -> Dict[str, Any]:
        """
        Process a syllabus to generate a knowledge graph.
        
        Args:
            syllabus_text: The syllabus text to process
            user_id: Optional user ID for caching
            
        Returns:
            Dict: Processing result with status and graph ID
        """
        try:
            # Run the async parser in a synchronous context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Parse the syllabus
                parsed_data = loop.run_until_complete(self.parser.parse(syllabus_text))
                
                # Generate the knowledge graph
                graph_data = loop.run_until_complete(self.graph_generator.generate_graph(parsed_data))
                
                # Enrich the data
                enriched_data = self.graph_generator.enrich_graph_data(graph_data)
            finally:
                loop.close()
            
            # Add additional metadata
            enriched_data["metadata"]["source"] = "user_uploaded_syllabus"
            enriched_data["metadata"]["timestamp"] = datetime.now().isoformat()
            enriched_data["metadata"]["user_id"] = user_id
            
            # Cache the graph
            cache_key = f"graph_{user_id}_{hash(syllabus_text)}"
            self.graph_cache[cache_key] = enriched_data
            
            # Get some statistics for the response
            node_count = len(enriched_data.get("nodes", []))
            link_count = len(enriched_data.get("links", []))
            modules = [node["id"] for node in enriched_data.get("nodes", []) 
                      if node.get("type") == "module"]
            
            return {
                "status": "success",
                "graph_id": cache_key,
                "node_count": node_count,
                "link_count": link_count,
                "modules": modules
            }
        except Exception as e:
            logger.error(f"Error processing syllabus: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}
    
    def get_graph(self, graph_id: str) -> Dict[str, Any]:
        """
        Retrieve a generated knowledge graph by ID.
        
        Args:
            graph_id: The graph ID to retrieve
            
        Returns:
            Dict: The graph data or an error
        """
        # If graph_id is a user ID, get the most recent graph for that user
        if graph_id.startswith("user_") and not graph_id.startswith("graph_"):
            user_id = graph_id
            user_graphs = [k for k in self.graph_cache.keys() if k.startswith(f"graph_{user_id}_")]
            
            if user_graphs:
                # Sort by timestamp and get the most recent
                most_recent = sorted(user_graphs)[-1]
                return self.graph_cache[most_recent]
        
        # Otherwise, look for exact graph ID match
        if graph_id in self.graph_cache:
            return self.graph_cache[graph_id]
        
        return {"error": "Graph not found"}
    
    def enrich_graph_with_resources(self, graph_data: Dict) -> Dict[str, Any]:
        """
        Enrich a graph with learning resources.
        
        Args:
            graph_data: The graph data to enrich
            
        Returns:
            Dict: The enriched graph data
        """
        # For now, just return the original graph data
        # In a full implementation, we'd call the LLM to generate resources
        return graph_data 