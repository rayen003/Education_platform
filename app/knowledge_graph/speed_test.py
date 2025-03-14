#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Speed test for validating relationships.
This script tests the optimized validation with a small syllabus.
"""

import logging
import asyncio
import time
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from app.knowledge_graph.syllabus_parser import SyllabusParser
from app.knowledge_graph.graph_generator import GraphGenerator
from app.knowledge_graph.meta_validator import MetaValidator
from app.math_services.services.llm.openai_service import OpenAILLMService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("speed_test")

# Small sample syllabus for testing
SMALL_SYLLABUS = """
Course Title: Introduction to Python

Course Description:
A brief introduction to Python programming language basics.

Modules:

1. Python Basics
   - Variables
   - Data Types
   - Control Flow

2. Functions
   - Function Definition
   - Parameters
   - Return Values

3. Data Structures
   - Lists
   - Dictionaries
   - Tuples
"""

async def run_test():
    """Run the speed test with both old and new validation methods."""
    logger.info("=== Speed Test ===")
    logger.info("Testing the optimized validation with a small syllabus")
    
    # Initialize services
    llm_service = OpenAILLMService()
    validator = MetaValidator(llm_service)
    parser = SyllabusParser()
    
    # Parse the syllabus
    start_time = time.time()
    logger.info("Parsing syllabus...")
    parsed_data = await parser.parse(SMALL_SYLLABUS)
    
    # Log parsing results
    modules = parsed_data.get("content", {})
    relationships = parsed_data.get("relationships", [])
    
    logger.info(f"Parsed {len(modules)} modules with {sum(len(concepts) for concepts in modules.values())} concepts")
    logger.info(f"Generated {len(relationships)} initial relationships")
    logger.info(f"Parsing took {time.time() - start_time:.2f} seconds")
    
    # First test: Basic graph generation without validation
    logger.info("\nTest 1: Basic graph generation without validation")
    start_time = time.time()
    
    graph_generator = GraphGenerator()
    basic_graph = await graph_generator.generate_graph(parsed_data)
    
    logger.info(f"Basic graph generation took {time.time() - start_time:.2f} seconds")
    logger.info(f"Generated graph with {len(basic_graph.get('nodes', []))} nodes and {len(basic_graph.get('links', []))} links")
    
    # Second test: Graph generation with optimized validation
    logger.info("\nTest 2: Graph generation with optimized validation")
    start_time = time.time()
    
    graph_generator = GraphGenerator(validator)
    graph_data_generator = graph_generator.generate_with_streaming_validation(parsed_data)
    
    # Get the initial graph immediately (should be fast)
    initial_time = time.time()
    initial_graph = await anext(graph_data_generator)
    logger.info(f"Initial graph generation took {time.time() - initial_time:.2f} seconds")
    
    # Process all updates (should take longer due to validation)
    validation_start = time.time()
    last_graph = initial_graph
    try:
        async for graph_update in graph_data_generator:
            validation_progress = graph_update.get('metadata', {}).get('validation_status', 'unknown')
            logger.info(f"Validation progress: {validation_progress}")
            last_graph = graph_update
    except StopAsyncIteration:
        pass
        
    logger.info(f"Full validation took {time.time() - validation_start:.2f} seconds")
    logger.info(f"Total graph generation with validation took {time.time() - start_time:.2f} seconds")
    
    # Check how many relationships were validated
    validated_count = sum(1 for link in last_graph.get('links', [])
                        if link.get('metadata', {}).get('validated', False))
    logger.info(f"Validated {validated_count} of {len(last_graph.get('links', []))} relationships")
    
    # Third test: Just validate a batch directly (to see raw batch validation speed)
    if relationships:
        logger.info("\nTest 3: Direct batch validation")
        test_batch = relationships[:min(5, len(relationships))]
        
        start_time = time.time()
        validated_batch = await validator.validate_batch(test_batch)
        
        logger.info(f"Batch validation of {len(test_batch)} relationships took {time.time() - start_time:.2f} seconds")
        logger.info(f"Average time per relationship: {(time.time() - start_time) / len(test_batch):.2f} seconds")
    
    logger.info("\nSpeed test completed!")

if __name__ == "__main__":
    logger.info("Starting speed test...")
    asyncio.run(run_test()) 