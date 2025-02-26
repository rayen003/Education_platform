import json
from typing import Dict, List, Any
import numpy as np

class KnowledgeGraphGenerator:
    def __init__(self):
        self.node_cache = {}
        
    def generate_concept_metadata(self, concept: str, module: str, module_data: Dict) -> Dict:
        """Generate rich metadata for a concept."""
        return {
            "id": concept,
            "type": "concept",
            "module": module,
            "difficulty": module_data["difficulty"],
            "mastery_level": 0,  # Track user's mastery of this concept
            "last_visited": None,  # Track when user last interacted with this concept
            "visit_count": 0,  # Track how many times user has visited this concept
            "recommended_order": module_data.get("concept_order", {}).get(concept, 0),
            "estimated_time": module_data.get("estimated_times", {}).get(concept, 30),  # minutes
            "learning_style_tags": module_data.get("learning_styles", {}).get(concept, []),
            "prerequisites": self._get_concept_prerequisites(concept, module_data),
            "next_concepts": self._get_next_concepts(concept, module_data),
            "resources": {
                "explore": f"/resources/{concept.lower().replace(' ', '_')}",
                "related": f"/related/{concept.lower().replace(' ', '_')}"
            }
        }
        
    def _get_concept_prerequisites(self, concept: str, module_data: Dict) -> List[str]:
        """Get direct prerequisites for a concept."""
        concept_order = module_data.get("concept_order", {})
        current_order = concept_order.get(concept, 0)
        
        # Only consider concepts that come directly before this one
        prerequisites = []
        for prereq, order in concept_order.items():
            if order == current_order - 1:
                prerequisites.append(prereq)
                
        # Add cross-module prerequisites if needed
        if module_data["prerequisites"] and current_order == 0:
            for prereq_module in module_data["prerequisites"]:
                if prereq_module in self.node_cache:
                    last_concepts = self._get_last_concepts_in_module(prereq_module)
                    prerequisites.extend(last_concepts)
        
        return prerequisites
    
    def _get_next_concepts(self, concept: str, module_data: Dict) -> List[str]:
        """Get concepts that should be learned after this one."""
        concept_order = module_data.get("concept_order", {})
        current_order = concept_order.get(concept, 0)
        
        next_concepts = []
        for next_concept, order in concept_order.items():
            if order == current_order + 1:
                next_concepts.append(next_concept)
        return next_concepts
    
    def _get_last_concepts_in_module(self, module: str) -> List[str]:
        """Get the final concepts in a module's learning sequence."""
        if module not in self.node_cache:
            return []
            
        module_concepts = self.node_cache[module]
        if not module_concepts:
            return []
            
        max_order = max(c.get("recommended_order", 0) for c in module_concepts)
        return [c["id"] for c in module_concepts if c.get("recommended_order", 0) == max_order]

    def generate_graph_data(self, content: Dict, module_metadata: Dict) -> Dict:
        """Generate knowledge graph data with directed edges and simplified relationships."""
        nodes = []
        links = []
        
        # First pass: Create module nodes
        for module, module_data in module_metadata.items():
            module_node = {
                "id": module,
                "type": "module",
                "difficulty": module_data["difficulty"],
                "description": module_data["description"],
                "prerequisites": module_data["prerequisites"],
                "estimated_time": sum(module_data.get("estimated_times", {}).values()),
                "completion_rate": 0  # Track module completion
            }
            nodes.append(module_node)
            self.node_cache[module] = []
            
            # Add directed prerequisite links between modules
            for prereq in module_data["prerequisites"]:
                links.append({
                    "source": prereq,
                    "target": module,
                    "type": "module_prerequisite",
                    "strength": 2
                })
        
        # Second pass: Create concept nodes with linear progression
        for module, concepts in content.items():
            module_data = module_metadata[module]
            module_concepts = []
            
            for concept in concepts:
                concept_node = self.generate_concept_metadata(concept, module, module_data)
                nodes.append(concept_node)
                module_concepts.append(concept_node)
                self.node_cache[module].append(concept_node)
                
                # Add directed edge from module to first concepts
                if concept_node["recommended_order"] == 0:
                    links.append({
                        "source": module,
                        "target": concept,
                        "type": "module_concept",
                        "strength": 1
                    })
                
                # Add prerequisite links
                for prereq in concept_node["prerequisites"]:
                    links.append({
                        "source": prereq,
                        "target": concept,
                        "type": "prerequisite",
                        "strength": 1.5
                    })
                
                # Add next concept links
                for next_concept in concept_node["next_concepts"]:
                    links.append({
                        "source": concept,
                        "target": next_concept,
                        "type": "next_concept",
                        "strength": 1
                    })
        
        return {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "moduleCount": len(module_metadata),
                "conceptCount": len([n for n in nodes if n["type"] == "concept"]),
                "maxDifficulty": max(n["difficulty"] for n in nodes),
                "modules": list(module_metadata.keys()),
                "totalEstimatedTime": sum(n.get("estimated_time", 0) for n in nodes),
                "recommendedPath": self._generate_recommended_path(nodes, links)
            }
        }
        
    def _generate_recommended_path(self, nodes, links) -> List[str]:
        """Generate a recommended learning path through the concepts."""
        # Implementation can be enhanced with more sophisticated path finding
        concept_nodes = [n for n in nodes if n["type"] == "concept"]
        return sorted(
            [n["id"] for n in concept_nodes],
            key=lambda x: next((n["recommended_order"] for n in concept_nodes if n["id"] == x), 0)
        )