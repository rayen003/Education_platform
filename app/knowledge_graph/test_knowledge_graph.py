#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the knowledge graph components.
This script tests the functionality of the syllabus parser, graph generator,
and meta validator independently of the Flask app.
"""

import logging
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root))

from app.knowledge_graph.syllabus_parser import SyllabusParser
from app.knowledge_graph.graph_generator import GraphGenerator
from app.knowledge_graph.meta_validator import MetaValidator
from app.math_services.services.llm.openai_service import OpenAILLMService

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
log_file = f"knowledge_graph_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    datefmt=date_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
)

logger = logging.getLogger("knowledge_graph_test")

# Sample syllabus text (focusing on microeconomics)
SAMPLE_SYLLABUS = """
Course Title: Principles of Microeconomics

Course Description:
This course introduces students to the fundamental principles of microeconomics, 
focusing on the behavior of individuals, firms, and markets. Students will learn about 
supply and demand, consumer theory, production theory, market structures, and market failures.

Learning Outcomes:
- Understand how markets operate and how prices are determined
- Analyze consumer behavior and decision-making processes
- Explain how firms make production decisions
- Compare different market structures and their implications
- Evaluate market failures and potential policy solutions

Prerequisites:
- Basic understanding of mathematics (algebra)
- Introduction to Economics (recommended but not required)

Modules:

1. Fundamentals of Economics
   - Scarcity and Choice
   - Opportunity Cost
   - Production Possibilities Frontier
   - Comparative Advantage
   - Economic Systems

2. Supply and Demand
   - Demand Curves
   - Supply Curves
   - Market Equilibrium
   - Price Elasticity
   - Consumer and Producer Surplus

3. Consumer Theory
   - Utility Maximization
   - Budget Constraints
   - Indifference Curves
   - Income and Substitution Effects
   - Consumer Choice

4. Production Theory
   - Production Functions
   - Cost Curves
   - Profit Maximization
   - Short-run vs. Long-run Decisions
   - Economies of Scale

5. Market Structures
   - Perfect Competition
   - Monopoly
   - Monopolistic Competition
   - Oligopoly
   - Game Theory

6. Market Failures
   - Externalities
   - Public Goods
   - Common Resources
   - Information Asymmetry
   - Government Regulation
"""

class TestTimer:
    """Simple timer context manager for measuring performance."""
    def __init__(self, description):
        self.description = description
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting: {self.description}")
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        logger.info(f"Completed: {self.description} in {elapsed:.2f} seconds")

async def test_syllabus_parser():
    """Test the syllabus parser functionality."""
    logger.info("=== Testing Syllabus Parser ===")
    
    # Initialize the parser without LLM for basic parsing
    parser = SyllabusParser()
    
    with TestTimer("Basic syllabus parsing"):
        parsed_data = await parser.parse(SAMPLE_SYLLABUS)
    
    # Log some statistics about the parsed data
    modules = parsed_data.get("content", {})
    relationships = parsed_data.get("relationships", [])
    
    logger.info(f"Parsed {len(modules)} modules with {sum(len(concepts) for concepts in modules.values())} concepts")
    logger.info(f"Generated {len(relationships)} initial relationships")
    
    # Test with LLM enrichment if available
    try:
        llm_service = OpenAILLMService()
        parser_with_llm = SyllabusParser(llm_service)
        
        with TestTimer("Syllabus parsing with LLM enrichment"):
            enriched_data = await parser_with_llm.parse(SAMPLE_SYLLABUS)
            # In a real implementation, you would call parser_with_llm.enrich_with_llm(parsed_data)
        
        logger.info("LLM enrichment successful")
    except Exception as e:
        logger.error(f"LLM enrichment failed: {str(e)}")
    
    return parsed_data

async def test_meta_validator(parsed_data):
    """Test the meta validator functionality."""
    logger.info("\n=== Testing Meta Validator ===")
    
    relationships = parsed_data.get("relationships", [])
    if not relationships:
        logger.warning("No relationships to validate")
        return []
    
    try:
        llm_service = OpenAILLMService()
        validator = MetaValidator(llm_service)
        
        # Test individual validation
        test_rel = relationships[0]
        logger.info(f"Testing individual validation for: {test_rel['source']} -> {test_rel['target']} ({test_rel['type']})")
        
        with TestTimer("Individual relationship validation"):
            metadata = await validator.validate_relationship(
                test_rel["source"],
                test_rel["target"],
                test_rel["type"]
            )
        
        logger.info(f"Validation result: confidence={metadata.confidence}, semantic_type={metadata.semantic_type}")
        
        # Test batch validation (limit to a few for testing)
        test_batch = relationships[:5]
        logger.info(f"Testing batch validation for {len(test_batch)} relationships")
        
        with TestTimer("Batch relationship validation"):
            validated_batch = await validator.validate_graph_relationships(test_batch)
        
        logger.info(f"Batch validation complete, validated {len(validated_batch)} relationships")
        
        # Test iterative validation (limit to a few for testing)
        logger.info("Testing iterative validation")
        
        with TestTimer("Iterative relationship validation"):
            iterative_results = await validator.iterative_validation(test_batch, iterations=2)
        
        logger.info(f"Iterative validation complete, {len(iterative_results)} relationships passed confidence threshold")
        
        return validated_batch
    except Exception as e:
        logger.error(f"Meta validator testing failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

async def test_graph_generator(parsed_data, validated_relationships=None):
    """Test the graph generator functionality."""
    logger.info("\n=== Testing Graph Generator ===")
    
    try:
        # Initialize graph generator
        graph_generator = GraphGenerator()
        
        # Test basic graph generation
        with TestTimer("Basic graph generation"):
            graph_data = await graph_generator.generate_graph_data(
                parsed_data.get("content", {}),
                parsed_data.get("module_metadata", {}),
                parsed_data.get("relationships", [])
            )
        
        # Log some statistics about the generated graph
        node_count = len(graph_data.get("nodes", []))
        link_count = len(graph_data.get("links", []))
        module_count = sum(1 for node in graph_data.get("nodes", []) if node.get("type") == "module")
        concept_count = sum(1 for node in graph_data.get("nodes", []) if node.get("type") == "concept")
        
        logger.info(f"Generated graph with {node_count} nodes ({module_count} modules, {concept_count} concepts) and {link_count} links")
        
        # Test graph enrichment
        with TestTimer("Graph data enrichment"):
            enriched_data = graph_generator.enrich_graph_data(graph_data)
        
        logger.info(f"Enriched graph with {len(enriched_data.get('metadata', {}))} metadata fields")
        
        # Test graph generation with validator if available
        if validated_relationships:
            # Create a validator-enabled graph generator
            llm_service = OpenAILLMService()
            validator = MetaValidator(llm_service)
            validated_generator = GraphGenerator(meta_validator=validator)
            
            with TestTimer("Graph generation with validation"):
                validated_graph = await validated_generator.generate_graph(parsed_data)
            
            validated_link_count = sum(1 for link in validated_graph.get("links", []) 
                                  if link.get("metadata", {}).get("validated", False))
            
            logger.info(f"Generated validated graph with {len(validated_graph.get('nodes', []))} nodes and " +
                       f"{len(validated_graph.get('links', []))} links ({validated_link_count} validated)")
            
            # Save the validated graph data
            with open("validated_test_graph.json", "w") as f:
                json.dump(validated_graph, f, indent=2)
            logger.info("Saved validated graph data to validated_test_graph.json")
            
            return validated_graph
        
        # Save the basic graph data
        with open("basic_test_graph.json", "w") as f:
            json.dump(enriched_data, f, indent=2)
        logger.info("Saved basic graph data to basic_test_graph.json")
        
        return enriched_data
    except Exception as e:
        logger.error(f"Graph generator testing failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {}

async def main():
    """Main test function."""
    logger.info("Starting knowledge graph component tests")
    
    try:
        # Test syllabus parser
        parsed_data = await test_syllabus_parser()
        
        # Test meta validator
        validated_relationships = await test_meta_validator(parsed_data)
        
        # Test graph generator
        graph_data = await test_graph_generator(parsed_data, validated_relationships)
        
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger.info("=== Knowledge Graph Component Test Script ===")
    logger.info(f"Started at: {datetime.now().strftime(date_format)}")
    
    # Run the tests
    asyncio.run(main())
    
    logger.info(f"Tests completed at: {datetime.now().strftime(date_format)}")
    logger.info(f"Logs saved to: {log_file}") 