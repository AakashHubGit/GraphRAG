# Simplified Graph RAG Implementation

A clean, simplified implementation of Graph RAG (Retrieval-Augmented Generation) that combines traditional RAG with knowledge graphs for enhanced reasoning and retrieval.

## What is Graph RAG?

Graph RAG extends traditional RAG by:

1. **Building knowledge graphs** from documents to capture entity relationships
2. **Using graph structure** for more intelligent retrieval
3. **Enabling multi-hop reasoning** through connected entities
4. **Providing better context** by understanding entity relationships

## Core Components

- **DocumentProcessor**: Handles text chunking and entity/relation extraction using Gemini LLM
- **KnowledgeGraph**: Manages graph structure, entities, relations, and embeddings
- **GraphRAGRetriever**: Combines semantic similarity with graph-based retrieval
- **GraphRAGSystem**: Main system integrating all components

## Key Features

- **Hybrid Retrieval**: Combines semantic similarity (traditional RAG) with graph structure
- **Multi-hop Reasoning**: Can follow entity relationships across multiple steps
- **Context-aware Embeddings**: Entity embeddings include relationship context
- **LLM-based Extraction**: Uses Gemini for more accurate entity/relation extraction
- **Simple Architecture**: Single file with clear separation of concerns

## Quick Start

### 1. Setup Environment

```bash
# Clone or download the files
# Ensure you have Python 3.8+

# Run setup script
python setup.py
```

### 2. Configure API Key

Edit the `.env` file created by setup:

```bash
# Get your API key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

### 3. Test Installation

```bash
python test_graph_rag.py
```

### 4. Run Example

```bash
python graph_rag.py
```

## Example Usage

```python
from graph_rag import GraphRAGSystem

# Initialize system
system = GraphRAGSystem()

# Your documents
documents = [
    "FutureTech Foundation is requesting $100,000 for youth training programs. John Smith leads the organization.",
    "ABC Institute offers robotics workshops for $2,000. Dr. Sarah Johnson manages the robotics program."
]

# Build knowledge base
system.build_knowledge_base(documents)

# Query the system
result = system.query("What is the budget for FutureTech Foundation?")
print(result['answer'])

# Multi-hop reasoning
reasoning_result = system.multi_hop_reasoning("How much funding does John Smith's organization need?")
print(reasoning_result['answer'])
```

## How It Works

### 1. Knowledge Graph Construction

- Documents are processed to extract entities (PERSON, ORG, MONEY, etc.)
- Relationships between entities are identified using LLM
- A directed graph is built connecting related entities
- Context-aware embeddings are generated for all entities

### 2. Enhanced Retrieval

- **Semantic retrieval**: Find text chunks similar to query (traditional RAG)
- **Entity retrieval**: Find entities semantically related to query
- **Graph expansion**: Include neighboring entities from graph structure
- **Subgraph extraction**: Get relevant portion of knowledge graph

### 3. Response Generation

- Combines document sources with graph relationship facts
- LLM generates response using both textual evidence and structural knowledge
- Multi-hop reasoning follows entity chains for complex queries

## Key Improvements Over Traditional RAG

| Aspect        | Traditional RAG             | Graph RAG                                     |
| ------------- | --------------------------- | --------------------------------------------- |
| Context       | Just similar text chunks    | Text chunks + entity relationships            |
| Reasoning     | Single-step lookup          | Multi-hop reasoning through entities          |
| Connections   | Limited by chunk boundaries | Follows entity relationships across documents |
| Understanding | Semantic similarity only    | Semantic + structural knowledge               |

## Architecture Overview

```
Documents → DocumentProcessor → KnowledgeGraph
                ↓                    ↓
        Text Chunks +             Entities &
        Embeddings               Relations
                ↓                    ↓
            GraphRAGRetriever ← Subgraph
                    ↓
            Enhanced Context
                    ↓
            GraphRAGSystem → LLM Response
```

## File Structure

```
graph-rag/
├── graph_rag.py          # Main implementation (all components)
├── requirements.txt      # Python dependencies
├── setup.py             # Setup and verification script
├── test_graph_rag.py    # Basic functionality test (created by setup)
├── .env                 # Environment variables (created by setup)
└── README.md           # This file
```

## Configuration

The system uses several configurable parameters in the main classes:

```python
# Document processing
chunk_size = 500          # Words per chunk
chunk_overlap = 50        # Overlap between chunks

# Retrieval
top_k_chunks = 3         # Number of chunks to retrieve
top_k_entities = 3       # Number of entities to retrieve
similarity_threshold = 0.1  # Minimum similarity for relevance

# Multi-hop reasoning
max_hops = 3             # Maximum reasoning chain length
```

## Troubleshooting

### Common Issues

1. **spaCy model not found**

   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **API key not working**

   - Verify your Gemini API key is correct
   - Check the .env file format
   - Ensure no extra spaces in API key

3. **Memory issues with large documents**

   - Reduce chunk_size
   - Process documents in smaller batches
   - Use lighter embedding models

4. **Poor entity extraction**
   - Check document quality and formatting
   - Consider preprocessing to clean text
   - Adjust LLM prompt in extract_entities_relations_llm()

### Performance Tips

- **For large document sets**: Consider using FAISS indexing (commented out in simplified version)
- **For production**: Add caching for embeddings and query results
- **For accuracy**: Fine-tune similarity thresholds based on your use case
- **For speed**: Use smaller embedding models or reduce expansion distance

## Limitations

1. **API Dependency**: Requires Google Gemini API for entity extraction and response generation
2. **English Only**: Currently configured for English text (spaCy model)
3. **Memory Usage**: Stores all embeddings in memory (not suitable for very large corpora)
4. **Simple Relation Types**: Uses basic relationship extraction (could be enhanced)

## Extensions and Improvements

Potential enhancements you could add:

- **Persistent Storage**: Save/load knowledge graphs to disk
- **Batch Processing**: Handle large document collections efficiently
- **Advanced Indexing**: Use FAISS or similar for faster similarity search
- **Multi-language**: Support for other languages via different spaCy models
- **Relationship Scoring**: Weight relationships by confidence/importance
- **Query Expansion**: Automatically expand queries with related terms
- **Visualization**: Generate graph visualizations for debugging
- **Metrics**: Add evaluation metrics for retrieval quality

## Contributing

The code is designed to be easily extensible. Key extension points:

- **Entity Extraction**: Modify `extract_entities_relations_llm()` for better extraction
- **Retrieval Strategy**: Enhance `graph_enhanced_retrieval()` with new methods
- **Response Generation**: Customize `_generate_response()` for domain-specific formatting
- **Graph Operations**: Add new graph analysis methods to `KnowledgeGraph`

## License

This implementation is provided as educational material. Adapt and use as needed for your projects.

---

For questions or issues, refer to the troubleshooting section or check that all setup steps were completed successfully.
