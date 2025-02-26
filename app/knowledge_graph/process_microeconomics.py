from graph_generator import KnowledgeGraphGenerator
import json

"""Process microeconomics content into a structured knowledge graph."""

microeconomics_content = {
    "Production Theory": [
        "Production Function",
        "Law of Diminishing Returns",
        "Variable Costs",
        "Fixed Costs",
        "Total Costs",
        "Marginal Costs",
        "Average Costs",
        "Economies of Scale",
        "Diseconomies of Scale",
        "Short Run Production",
        "Long Run Production"
    ]
}

microeconomics_metadata = {
    "Production Theory": {
        "difficulty": 3,
        "description": "Study of how firms convert inputs into outputs and the associated costs",
        "prerequisites": [],
        "concept_order": {
            "Production Function": 0,
            "Law of Diminishing Returns": 1,
            "Variable Costs": 2,
            "Fixed Costs": 2,
            "Total Costs": 3,
            "Marginal Costs": 4,
            "Average Costs": 4,
            "Short Run Production": 5,
            "Long Run Production": 6,
            "Economies of Scale": 7,
            "Diseconomies of Scale": 7
        },
        "estimated_times": {
            "Production Function": 45,
            "Law of Diminishing Returns": 30,
            "Variable Costs": 20,
            "Fixed Costs": 20,
            "Total Costs": 25,
            "Marginal Costs": 35,
            "Average Costs": 35,
            "Short Run Production": 40,
            "Long Run Production": 40,
            "Economies of Scale": 30,
            "Diseconomies of Scale": 30
        },
        "learning_styles": {
            "Production Function": ["visual", "mathematical"],
            "Law of Diminishing Returns": ["visual", "real-world"],
            "Variable Costs": ["mathematical", "practical"],
            "Fixed Costs": ["mathematical", "practical"],
            "Total Costs": ["mathematical", "analytical"],
            "Marginal Costs": ["mathematical", "analytical"],
            "Average Costs": ["mathematical", "analytical"],
            "Short Run Production": ["theoretical", "analytical"],
            "Long Run Production": ["theoretical", "analytical"],
            "Economies of Scale": ["theoretical", "real-world"],
            "Diseconomies of Scale": ["theoretical", "real-world"]
        }
    }
}

def main():
    # Initialize the graph generator
    generator = KnowledgeGraphGenerator()
    
    # Generate graph data with enhanced metadata
    graph_data = generator.generate_graph_data(microeconomics_content, microeconomics_metadata)
    
    # Save to file
    with open('microeconomics_graph.json', 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print("Generated knowledge graph data with:")
    print(f"- {len(graph_data['nodes'])} nodes")
    print(f"- {len(graph_data['links'])} links")

if __name__ == "__main__":
    main()
