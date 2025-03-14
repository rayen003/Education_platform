#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a knowledge graph from a syllabus input.

This script provides a command-line interface to generate knowledge graphs
from syllabus text using the modular knowledge graph components.
"""

import os
import sys
import json
import logging
import asyncio
import argparse
import datetime
from pathlib import Path
from typing import List, Dict

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from app.knowledge_graph.syllabus_parser import SyllabusParser
from app.knowledge_graph.graph_generator import GraphGenerator
from app.knowledge_graph.meta_validator import MetaValidator
from app.math_services.services.llm.openai_service import OpenAILLMService

# Configure logging
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"graph_generation_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("knowledge_graph")

# Define the default sample syllabus text
DEFAULT_SYLLABUS = """
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

async def generate_knowledge_graph(
    syllabus_text: str, 
    validate: bool = True, 
    model_name: str = "gpt-4o-mini",
    output_file: str = None
) -> dict:
    """
    Generate a knowledge graph from a syllabus text.
    
    Args:
        syllabus_text: The raw syllabus text to process
        validate: Whether to validate relationships using LLM
        model_name: The LLM model to use for validation
        output_file: Optional output file path for the graph data
        
    Returns:
        dict: The generated graph data
    """
    logger.info("Starting knowledge graph generation process")
    
    try:
        # Initialize services
        llm_service = None
        meta_validator = None
        
        if validate:
            logger.info(f"Initializing OpenAI LLM service with model {model_name}")
            llm_service = OpenAILLMService(model=model_name)
            logger.info("Initializing Meta Validator")
            meta_validator = MetaValidator(llm_service)
        
        # Initialize parser and graph generator
        logger.info("Initializing Syllabus Parser")
        parser = SyllabusParser(llm_service)
        
        logger.info("Initializing Graph Generator")
        graph_generator = GraphGenerator(meta_validator)
        
        # Parse syllabus text
        logger.info("Parsing syllabus text")
        parsed_data = await parser.parse(syllabus_text)
        
        # Generate graph with progressive validation
        logger.info("Generating knowledge graph with progressive validation")
        graph_data_generator = graph_generator.generate_with_streaming_validation(parsed_data)
        
        # Get the initial graph data
        initial_graph = await anext(graph_data_generator)
        current_graph = initial_graph
        
        # Process streaming updates if validation is enabled
        if validate:
            logger.info("Processing streaming validation updates")
            try:
                # Process each update as it becomes available
                async for graph_update in graph_data_generator:
                    logger.info(f"Received graph update with validation status: {graph_update.get('metadata', {}).get('validation_status', 'unknown')}")
                    current_graph = graph_update
            except StopAsyncIteration:
                logger.info("Streaming validation completed")
        
        # Enrich the final graph data
        logger.info("Enriching graph data with additional metadata")
        enriched_data = graph_generator.enrich_graph_data(current_graph)
        
        # Save the graph data if requested
        if output_file:
            out_path = output_file
        else:
            prefix = "validated_" if validate else ""
            json_dir = os.path.join(os.path.dirname(__file__), "json_files")
            os.makedirs(json_dir, exist_ok=True)
            out_path = os.path.join(json_dir, f"{prefix}graph_data_{timestamp}.json")
        
        logger.info(f"Saving graph data to {out_path}")
        with open(out_path, "w") as f:
            json.dump(enriched_data, f, indent=2)
        
        # Log statistics
        node_count = len(enriched_data.get("nodes", []))
        link_count = len(enriched_data.get("links", []))
        module_count = sum(1 for node in enriched_data.get("nodes", []) 
                          if node.get("type") == "module")
        
        logger.info(f"Generated graph contains {node_count} nodes and {link_count} links")
        logger.info(f"Graph includes {module_count} modules")
        
        if validate:
            validated_links = sum(1 for link in enriched_data.get("links", [])
                              if link.get("metadata", {}).get("validated", False))
            logger.info(f"Validated {validated_links} of {link_count} relationships")
        
        logger.info("Knowledge graph generation completed successfully")
        return enriched_data
    
    except Exception as e:
        logger.error(f"Error generating knowledge graph: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate a knowledge graph from a syllabus text")
    
    parser.add_argument(
        "--input",
        "-i",
        help="Input syllabus file path (if not provided, uses the default example)",
        type=str
    )
    
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path for the graph data",
        type=str
    )
    
    parser.add_argument(
        "--model",
        "-m",
        help="LLM model to use (default: gpt-4o-mini)",
        default="gpt-4o-mini",
        type=str
    )
    
    parser.add_argument(
        "--no-validate",
        help="Skip relationship validation",
        action="store_true"
    )
    
    return parser.parse_args()

async def main():
    """Main entry point for the script."""
    logger.info("=== Knowledge Graph Generation ===")
    logger.info(f"Started at: {datetime.datetime.now()}")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Get syllabus text
    syllabus_text = DEFAULT_SYLLABUS
    if args.input:
        logger.info(f"Reading syllabus from {args.input}")
        try:
            with open(args.input, "r") as f:
                syllabus_text = f.read()
        except Exception as e:
            logger.error(f"Failed to read input file: {e}")
            sys.exit(1)
    else:
        logger.info("Using default sample syllabus")
    
    # Generate the knowledge graph
    graph_data = await generate_knowledge_graph(
        syllabus_text=syllabus_text,
        validate=not args.no_validate,
        model_name=args.model,
        output_file=args.output
    )
    
    if "error" in graph_data:
        logger.error("Failed to generate knowledge graph")
        sys.exit(1)
    
    logger.info(f"Finished at: {datetime.datetime.now()}")
    logger.info(f"Logs saved to {log_file}")

if __name__ == "__main__":
    # Check if running on Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the main function
    asyncio.run(main()) 