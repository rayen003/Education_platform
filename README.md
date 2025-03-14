# Educational Technology Knowledge Graph Generator

This project generates interactive knowledge graphs from educational content, particularly syllabus text. It identifies concepts, their relationships, and validates these relationships using Large Language Models.

## Project Structure

```
Edtech_project/
├── app/
│   ├── knowledge_graph/       # Core knowledge graph generation components
│   │   ├── syllabus_parser.py # Parses syllabus text into structured data
│   │   ├── graph_generator.py # Generates graph data from parsed content
│   │   ├── meta_validator.py  # Validates relationships with LLM
│   │   ├── api_adapter.py     # Connects components to Flask API
│   │   └── test_knowledge_graph.py # Tests for knowledge graph components
│   │
│   ├── math_services/         # Services for mathematical operations
│   │   └── services/
│   │       └── llm/
│   │           ├── base_service.py    # Base LLM service interface
│   │           └── openai_service.py  # OpenAI implementation
│   │
│   ├── static/                # Static assets for the web app
│   │   └── data/
│   │       └── sample_syllabus.txt # Sample syllabus for testing
│   │
│   ├── templates/             # HTML templates for the web app
│   │   └── knowledge_graph.html # Main UI template
│   │
│   └── __init__.py            # Flask app initialization
│
├── generate_knowledge_graph.py  # CLI script for generating graphs
└── README.md                    # This file
```

## Core Components

### Syllabus Parser

The syllabus parser extracts structured information from raw syllabus text, including:
- Course title and description
- Module information
- Concepts within each module
- Initial relationships between concepts

### Graph Generator

The graph generator builds a knowledge graph from parsed syllabus data:
- Creates nodes for modules and concepts
- Establishes links between related concepts
- Enriches the graph with metadata and statistics

### Meta Validator

The meta validator uses Large Language Models to validate and enrich relationships:
- Validates relationships between concepts
- Assigns confidence scores
- Determines if relationships are bidirectional
- Provides reasoning and evidence
- Identifies semantic relationship types

## Usage

### Command-line Interface

Generate a knowledge graph from a syllabus file:

```bash
python generate_knowledge_graph.py --input path/to/syllabus.txt --output output_graph.json
```

Options:
- `--input`, `-i`: Input syllabus file path
- `--output`, `-o`: Output JSON file path
- `--model`, `-m`: LLM model to use (default: gpt-4o-mini)
- `--no-validate`: Skip relationship validation

### API

Start the Flask web app:

```bash
export FLASK_APP=app
export FLASK_ENV=development
flask run
```

API endpoints:
- `GET /api/knowledge_graph`: Get the current knowledge graph
- `POST /api/process_syllabus`: Process a new syllabus

## Development

### Requirements

- Python 3.8+
- OpenAI API key
- Flask

### Testing

Run the knowledge graph component tests:

```bash
python app/knowledge_graph/test_knowledge_graph.py
```

## License

MIT License
