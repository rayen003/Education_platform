"""
Graph Generator Module.

This module contains classes for generating knowledge graphs from structured data.
"""

import json
import datetime
import logging
import asyncio
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class GraphGenerator:
    """
    Generates knowledge graph data from structured inputs.
    """
    
    def __init__(self, meta_validator=None):
        """
        Initialize the graph generator.
        
        Args:
            meta_validator: Optional validator for relationships (can be None)
        """
        self.meta_validator = meta_validator
        self.node_cache = {}  # Cache for node metadata
        logger.info("Initialized GraphGenerator")
        if meta_validator:
            logger.info("Meta validator provided")
    
    async def generate_graph(self, parsed_data: Dict) -> Dict:
        """
        Generate a complete knowledge graph from parsed syllabus data.
        
        Args:
            parsed_data: The parsed syllabus data
            
        Returns:
            Dict: The complete graph data with nodes, links, and metadata
        """
        logger.info("Generating knowledge graph from parsed data")
        
        # Extract components from parsed data
        content = parsed_data.get("content", {})
        module_metadata = parsed_data.get("module_metadata", {})
        relationships = parsed_data.get("relationships", [])
        course_title = parsed_data.get("course_title", "Unknown Course")
        
        # Generate graph data
        graph_data = await self.generate_graph_data(content, module_metadata, relationships)
        
        # Add course metadata
        graph_data["metadata"]["course_title"] = course_title
        graph_data["metadata"]["generated_at"] = datetime.datetime.now().isoformat()
        
        # Log graph statistics
        logger.info(f"Generated graph with {len(graph_data['nodes'])} nodes and {len(graph_data['links'])} links")
        
        return graph_data
    
    async def generate_with_streaming_validation(self, parsed_data: Dict):
        """
        Generate graph and validate concurrently with streaming updates.
        
        Args:
            parsed_data: The parsed syllabus data
            
        Returns:
            Generator: Yields graph data with progressive validation updates
        """
        logger.info("Starting streaming graph generation with progressive validation")
        
        # Extract components from parsed data
        content = parsed_data.get("content", {})
        module_metadata = parsed_data.get("module_metadata", {})
        relationships = parsed_data.get("relationships", [])
        course_title = parsed_data.get("course_title", "Unknown Course")
        
        # First generate the basic graph without validation
        basic_graph = await self.generate_graph_data(content, module_metadata, relationships)
        
        # Add course metadata
        basic_graph["metadata"]["course_title"] = course_title
        basic_graph["metadata"]["generated_at"] = datetime.datetime.now().isoformat()
        basic_graph["metadata"]["validation_status"] = "in_progress"
        
        # Return basic graph immediately
        logger.info("Generated initial graph without validation")
        yield basic_graph
        
        # If no validator is available, we're done
        if not self.meta_validator:
            logger.info("No validator available, returning basic graph")
            basic_graph["metadata"]["validation_status"] = "skipped"
            yield basic_graph
            return
            
        try:
            # Start validation in the background
            logger.info("Starting background validation of relationships")
            validation_task = asyncio.create_task(
                self.meta_validator.validate_graph_relationships(relationships)
            )
            
            # As validations complete, update the graph
            validated_count = 0
            timeout_seconds = 5
            max_wait_time = 300  # 5 minutes max wait time
            total_wait_time = 0
            
            while not validation_task.done() and total_wait_time < max_wait_time:
                await asyncio.sleep(timeout_seconds)
                total_wait_time += timeout_seconds
                
                logger.info(f"Waiting for validations to complete... ({total_wait_time}s elapsed)")
                
                # We can't check intermediate results easily, so we'll just update the metadata
                basic_graph["metadata"]["validation_status"] = f"in_progress ({total_wait_time}s elapsed)"
                yield basic_graph
            
            # Get final validation results
            if validation_task.done():
                validated_relationships = await validation_task
                logger.info(f"Validation completed for {len(validated_relationships)} relationships")
            else:
                # If we've waited too long, cancel the task and use what we have
                logger.warning(f"Validation timed out after {total_wait_time}s, using partial results")
                validation_task.cancel()
                try:
                    validated_relationships = []
                except asyncio.CancelledError:
                    validated_relationships = []
            
            # Update the graph with validations
            final_graph = self._enrich_graph_with_validations(basic_graph, validated_relationships)
            final_graph["metadata"]["validation_status"] = "completed"
            final_graph["metadata"]["validated_at"] = datetime.datetime.now().isoformat()
            
            logger.info("Completed graph generation with validation")
            yield final_graph
            
        except Exception as e:
            logger.error(f"Error during streaming validation: {str(e)}")
            basic_graph["metadata"]["validation_status"] = f"error: {str(e)}"
            yield basic_graph
    
    def _enrich_graph_with_validations(self, graph_data: Dict, validated_relationships: List[Dict]) -> Dict:
        """Update graph with validation results."""
        # Make a deep copy to avoid modifying the original
        enriched_graph = json.loads(json.dumps(graph_data))
        
        # Create a lookup table for validated relationships
        validation_lookup = {}
        for rel in validated_relationships:
            key = f"{rel.get('source')}:{rel.get('target')}:{rel.get('type')}"
            validation_lookup[key] = rel
        
        # Update links with validation data
        for link in enriched_graph.get("links", []):
            source = link.get("source")
            target = link.get("target")
            link_type = link.get("type")
            
            if isinstance(source, dict):  # If source/target are objects not strings
                source = source.get("id")
            if isinstance(target, dict):
                target = target.get("id")
                
            key = f"{source}:{target}:{link_type}"
            if key in validation_lookup:
                validated_rel = validation_lookup[key]
                
                # Add metadata field if not present
                if "metadata" not in link:
                    link["metadata"] = {}
                
                # Update with validation data
                link["metadata"].update({
                    "validated": True,
                    "confidence": validated_rel.get("confidence", 0.5),
                    "reasoning": validated_rel.get("reasoning", ""),
                    "evidence": validated_rel.get("evidence", []),
                    "bidirectional": validated_rel.get("bidirectional", False),
                    "semantic_type": validated_rel.get("semantic_type", link_type),
                    "common_misconceptions": validated_rel.get("common_misconceptions", []),
                    "historical_context": validated_rel.get("historical_context", "")
                })
                
                # Update link strength based on confidence
                link["strength"] = validated_rel.get("confidence", 0.5) * 5
        
        # Add validation summary to metadata
        if "metadata" not in enriched_graph:
            enriched_graph["metadata"] = {}
            
        validated_count = sum(1 for link in enriched_graph.get("links", [])
                            if link.get("metadata", {}).get("validated", False))
                            
        enriched_graph["metadata"]["validated_link_count"] = validated_count
        enriched_graph["metadata"]["validation_percentage"] = (
            validated_count / len(enriched_graph.get("links", [])) * 100
            if enriched_graph.get("links") else 0
        )
        
        return enriched_graph
    
    async def generate_graph_data(self, content: Dict, module_metadata: Dict, relationships: List[Dict] = None) -> Dict:
        """
        Generate graph data with nodes and links.
        
        Args:
            content: Content information (concepts by module)
            module_metadata: Metadata for modules
            relationships: Optional list of relationships between concepts
            
        Returns:
            Dict: Graph data with nodes, links, and metadata
        """
        logger.info("Generating graph data structure")
        nodes = []
        links = []
        
        # First pass: Create nodes
        if content and module_metadata:
            # Create module nodes
            for module_id, module_data in module_metadata.items():
                module_node = {
                    "id": module_id,
                    "type": "module",
                    "difficulty": module_data.get("difficulty", 3),
                    "description": module_data.get("description", ""),
                }
                nodes.append(module_node)
                
                # Create concept nodes for this module
                module_concepts = content.get(module_id, [])
                for concept in module_concepts:
                    concept_node = {
                        "id": concept,
                        "type": "concept",
                        "module": module_id,
                        "difficulty": module_data.get("difficulty", 3),
                        "description": f"Concept: {concept}"
                    }
                    nodes.append(concept_node)
                    
                    # Create module-concept link
                    links.append({
                        "source": module_id,
                        "target": concept,
                        "type": "module_concept",
                        "strength": 1
                    })
            
            # Second pass: Create links between concepts based on relationships
            if relationships:
                for rel in relationships:
                    link = {
                        "source": rel.get("source"),
                        "target": rel.get("target"),
                        "type": rel.get("type", "related"),
                        "strength": rel.get("strength", 1),
                        "description": rel.get("description", "")
                    }
                    links.append(link)
        
        # Third pass: Validate relationships if validator is available
        validated_links = await self._validate_relationships(relationships) if self.meta_validator else []
        if validated_links:
            self._enrich_links_with_validation(links, validated_links)
        
        # Create the complete graph data
        graph_data = {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "node_count": len(nodes),
                "link_count": len(links),
                "modules": list(module_metadata.keys()) if module_metadata else []
            }
        }
        
        return graph_data
    
    async def _validate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Validate relationships using the meta-validator."""
        if not relationships or not self.meta_validator:
            return []
            
        try:
            logger.info(f"Validating {len(relationships)} relationships")
            validated = await self.meta_validator.validate_graph_relationships(relationships)
            logger.info(f"Completed validation of {len(validated)} relationships")
            return validated
        except Exception as e:
            logger.error(f"Error validating relationships: {str(e)}")
            return []
    
    def _enrich_links_with_validation(self, links: List[Dict], validated_rels: List[Dict]) -> None:
        """Enrich links with validation data."""
        # Create a lookup table for validated relationships
        validation_lookup = {}
        for rel in validated_rels:
            key = f"{rel.get('source')}:{rel.get('target')}:{rel.get('type')}"
            validation_lookup[key] = rel
        
        # Update existing links with validation data
        for link in links:
            key = f"{link.get('source')}:{link.get('target')}:{link.get('type')}"
            if key in validation_lookup:
                validated_rel = validation_lookup[key]
                
                # Skip relationship links between modules and concepts
                if link.get("type") == "module_concept":
                    continue
                    
                # Add metadata field if not present
                if "metadata" not in link:
                    link["metadata"] = {}
                
                # Update with validation data
                link["metadata"].update({
                    "validated": True,
                    "confidence": validated_rel.get("confidence", 0.5),
                    "reasoning": validated_rel.get("reasoning", ""),
                    "evidence": validated_rel.get("evidence", []),
                    "bidirectional": validated_rel.get("bidirectional", False),
                    "semantic_type": validated_rel.get("semantic_type", link.get("type", "related")),
                    "common_misconceptions": validated_rel.get("common_misconceptions", []),
                    "historical_context": validated_rel.get("historical_context", "")
                })
                
                # Update link strength based on confidence
                link["strength"] = validated_rel.get("confidence", 0.5) * 5
    
    def enrich_graph_data(self, graph_data: Dict) -> Dict:
        """
        Enrich graph data with additional metadata and statistics.
        
        Args:
            graph_data: The raw graph data
            
        Returns:
            Dict: The enriched graph data
        """
        logger.info("Enriching graph data with additional metadata")
        
        # Make a deep copy to avoid modifying the original
        enriched_data = json.loads(json.dumps(graph_data))
        
        # Add timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        # Add statistics
        nodes = enriched_data.get("nodes", [])
        links = enriched_data.get("links", [])
        
        # Count node types
        node_types = {}
        for node in nodes:
            node_type = node.get("type", "unknown")
            if node_type in node_types:
                node_types[node_type] += 1
            else:
                node_types[node_type] = 1
        
        # Count link types
        link_types = {}
        for link in links:
            link_type = link.get("type", "unknown")
            if link_type in link_types:
                link_types[link_type] += 1
            else:
                link_types[link_type] = 1
        
        # Calculate connectivity metrics
        connectivity = self._calculate_connectivity(nodes, links)
        
        # Add enhanced metadata
        if "metadata" not in enriched_data:
            enriched_data["metadata"] = {}
        
        enriched_data["metadata"].update({
            "timestamp": timestamp,
            "generator": "GraphGenerator",
            "node_types": node_types,
            "link_types": link_types,
            "connectivity": connectivity,
            "stats": {
                "total_nodes": len(nodes),
                "total_links": len(links),
                "average_degree": 2 * len(links) / len(nodes) if nodes else 0,
                "validated_links": sum(1 for link in links if link.get("metadata", {}).get("validated", False)),
                "high_confidence_links": sum(1 for link in links if link.get("metadata", {}).get("confidence", 0) >= 0.8)
            }
        })
        
        # Enrich nodes with additional metadata
        for node in enriched_data.get("nodes", []):
            # Calculate node degree (number of connections)
            node["degree"] = sum(1 for link in links if link["source"] == node["id"] or link["target"] == node["id"])
            
            # Add metadata field if not present
            if "metadata" not in node:
                node["metadata"] = {}
            
            # Add empty arrays for certain fields if not present
            for field in ["learning_resources", "key_terms", "applications"]:
                if field not in node["metadata"]:
                    node["metadata"][field] = []
        
        # Enrich links with additional metadata
        for link in enriched_data.get("links", []):
            # Add metadata field if not present
            if "metadata" not in link:
                link["metadata"] = {}
            
            # Flag relationship as validated if it has confidence score
            if "confidence" in link["metadata"]:
                link["metadata"]["validated"] = True
            
            # Extract semantic type if available
            if "type" in link and "semantic_type" not in link["metadata"]:
                link["metadata"]["semantic_type"] = link["type"]
        
        logger.info(f"Enriched graph data with {len(enriched_data['metadata'])} metadata fields")
        return enriched_data
    
    def _calculate_connectivity(self, nodes: List[Dict], links: List[Dict]) -> Dict:
        """Calculate connectivity metrics for the graph."""
        if not nodes or not links:
            return {
                "average_degree": 0,
                "density": 0,
                "max_degree": 0,
                "isolated_nodes": 0
            }
        
        # Calculate node degrees
        degrees = {}
        for node in nodes:
            node_id = node["id"]
            degrees[node_id] = 0
        
        for link in links:
            source = link["source"]
            target = link["target"]
            if source in degrees:
                degrees[source] += 1
            if target in degrees:
                degrees[target] += 1
        
        # Calculate metrics
        max_degree = max(degrees.values()) if degrees else 0
        isolated_nodes = sum(1 for degree in degrees.values() if degree == 0)
        average_degree = sum(degrees.values()) / len(nodes) if nodes else 0
        
        # Calculate density (ratio of actual to possible connections)
        n = len(nodes)
        possible_connections = n * (n - 1) / 2 if n > 1 else 1
        density = len(links) / possible_connections if possible_connections > 0 else 0
        
        return {
            "average_degree": average_degree,
            "density": density,
            "max_degree": max_degree,
            "isolated_nodes": isolated_nodes
        } 