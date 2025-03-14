"""
Syllabus Parser Module.

This module contains classes for parsing and structuring syllabus text into
a format suitable for knowledge graph generation.
"""

import re
import json
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class SyllabusParser:
    """
    Parses syllabus text into structured data with concepts, modules, and relationships.
    """
    
    def __init__(self, llm_service=None):
        """
        Initialize the parser.
        
        Args:
            llm_service: Optional LLM service for advanced parsing (can be None for basic parsing)
        """
        self.llm_service = llm_service
        logger.info("Initialized SyllabusParser")
    
    async def parse(self, syllabus_text: str) -> Dict:
        """
        Parse the syllabus text into structured content, modules, and relationships.
        
        Args:
            syllabus_text: The raw syllabus text to parse
            
        Returns:
            Dict: The parsed syllabus data
        """
        logger.info("Parsing syllabus text")
        
        # Extract course title
        course_title_match = re.search(r"Course Title:(.+?)(?:\n\s*\n|\n\s*[A-Za-z]+:)", syllabus_text, re.DOTALL)
        course_title = course_title_match.group(1).strip() if course_title_match else "Unknown Course"
        logger.info(f"Extracted course title: {course_title}")
        
        # Parse modules and their concepts
        modules = {}
        module_metadata = {}
        
        # Look for numbered modules with indented concepts
        module_pattern = re.compile(r'\s*(\d+)\.\s+([^\n]+)(?:\n((?:\s*-\s+[^\n]+\n?)+))?', re.MULTILINE)
        
        module_matches = list(module_pattern.finditer(syllabus_text))
        logger.info(f"Found {len(module_matches)} potential module matches")
        
        for match in module_matches:
            module_num = match.group(1).strip()
            module_name = match.group(2).strip()
            
            # Create module ID
            module_id = f"Module {module_num}: {module_name}"
            logger.info(f"Processing module: {module_id}")
            
            # Extract concepts for this module
            concepts = []
            if match.group(3):
                concept_lines = match.group(3).strip().split('\n')
                for line in concept_lines:
                    line = line.strip()
                    if line.startswith('-'):
                        concept = line[1:].strip()
                        concepts.append(concept)
            
            if concepts:
                modules[module_id] = concepts
                
                # Create module metadata
                module_metadata[module_id] = {
                    "difficulty": int(module_num),
                    "description": f"Module {module_num}: {module_name}",
                    "prerequisites": [],  # We'll fill this later
                    "concept_order": {concept: i for i, concept in enumerate(concepts)}
                }
        
        # If no modules found, try alternative approach
        if not modules:
            modules, module_metadata = self._try_alternative_parsing(syllabus_text)
        
        # Add prerequisites between modules
        module_ids = sorted(modules.keys(), key=lambda m: int(re.search(r'Module (\d+):', m).group(1)))
        for i, module_id in enumerate(module_ids):
            if i > 0:
                module_metadata[module_id]["prerequisites"] = [module_ids[i-1]]
        
        # Generate initial relationships
        relationships = self._generate_initial_relationships(modules, module_ids)
        
        # Create the result
        result = {
            "content": modules,
            "module_metadata": module_metadata,
            "relationships": relationships,
            "course_title": course_title
        }
        
        total_concepts = sum(len(concepts) for concepts in modules.values())
        logger.info(f"Extracted {len(modules)} modules with {total_concepts} concepts")
        logger.info(f"Generated {len(relationships)} initial relationships")
        
        return result
    
    def _try_alternative_parsing(self, syllabus_text: str) -> tuple:
        """Try alternative parsing approaches when the main approach fails."""
        logger.warning("No modules found with first pattern, trying alternative approach...")
        
        modules = {}
        module_metadata = {}
        
        # Try extracting modules by looking for numbered sections and their contents
        sections = re.findall(r'(\d+)\.\s+([^\n]+)\s+((?:[ \t]*-[^\n]+\n?)+)', syllabus_text)
        
        for section_num, section_name, section_content in sections:
            module_id = f"Module {section_num}: {section_name}"
            
            # Extract concepts
            concepts = []
            for line in section_content.split('\n'):
                line = line.strip()
                if line.startswith('-'):
                    concept = line[1:].strip()
                    if concept:  # Only add non-empty concepts
                        concepts.append(concept)
            
            if concepts:
                modules[module_id] = concepts
                module_metadata[module_id] = {
                    "difficulty": int(section_num),
                    "description": f"Module {section_num}: {section_name}",
                    "prerequisites": [],
                    "concept_order": {concept: i for i, concept in enumerate(concepts)}
                }
        
        return modules, module_metadata
    
    def _generate_initial_relationships(self, modules: Dict, module_ids: List) -> List[Dict]:
        """Generate initial relationships between concepts."""
        relationships = []
        
        # Add sequential relationships within modules
        for module_id, concepts in modules.items():
            for i in range(1, len(concepts)):
                source = concepts[i-1]
                target = concepts[i]
                relationships.append({
                    "source": source,
                    "target": target,
                    "type": "prerequisite",
                    "strength": 3,
                    "description": f"{source} is a prerequisite for {target}"
                })
        
        # Add relationships between modules (last concept of one module to first concept of next)
        for i in range(1, len(module_ids)):
            prev_module = module_ids[i-1]
            curr_module = module_ids[i]
            
            if modules[prev_module] and modules[curr_module]:
                source = modules[prev_module][-1]  # Last concept of previous module
                target = modules[curr_module][0]   # First concept of current module
                
                relationships.append({
                    "source": source,
                    "target": target,
                    "type": "builds_on",
                    "strength": 2,
                    "description": f"{target} builds on knowledge from {source}"
                })
        
        return relationships
    
    def enrich_with_llm(self, parsed_data: Dict) -> Dict:
        """
        Enrich the parsed data with LLM-generated content (if LLM service is available).
        
        Args:
            parsed_data: The basic parsed data from the syllabus
            
        Returns:
            Dict: The enriched parsed data
        """
        if not self.llm_service:
            logger.info("No LLM service provided for enrichment, skipping")
            return parsed_data
        
        # This is where we'd call the LLM to enrich the data with more advanced insights
        # For now, we'll just return the original data
        logger.info("LLM enrichment not implemented yet, returning basic parsed data")
        return parsed_data 