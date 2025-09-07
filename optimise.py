import spacy
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Set
import json
import re
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from process import DocumentProcessor
from kg import KnowledgeGraph
from retrieval import GraphRAGRetriever
from response import GraphRAGSystem
import openai
from typing import Optional
import pickle
import json
from datetime import datetime
import os

class GraphPersistence:
    """
    Handle saving and loading knowledge graphs
    Essential for production systems - you don't want to rebuild graphs every time
    """
    
    @staticmethod
    def save_knowledge_graph(kg: KnowledgeGraph, filepath: str, include_embeddings: bool = True):
        """
        Save knowledge graph to file
        
        We save as JSON for human readability, but you could use pickle for better performance
        """
        print(f"Saving knowledge graph to {filepath}")
        
        # Prepare graph data for serialization
        graph_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_nodes': len(kg.graph.nodes),
                'total_edges': len(kg.graph.edges),
                'total_chunks': len(kg.text_chunks)
            },
            'nodes': dict(kg.graph.nodes(data=True)),
            'edges': [(u, v, d) for u, v, d in kg.graph.edges(data=True)],
            'text_chunks': kg.text_chunks
        }
        
        # Optionally include embeddings (they're large!)
        if include_embeddings:
            graph_data['entity_embeddings'] = {
                k: v.tolist() for k, v in kg.entity_embeddings.items()
            }
            graph_data['chunk_embeddings'] = {
                str(k): v.tolist() for k, v in kg.chunk_embeddings.items()
            }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Knowledge graph saved successfully")
        print(f"File size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
    
    @staticmethod
    def load_knowledge_graph(filepath: str) -> KnowledgeGraph:
        """Load knowledge graph from file"""
        print(f"Loading knowledge graph from {filepath}")
        
        with open(filepath, 'r') as f:
            graph_data = json.load(f)
        
        # Create new knowledge graph instance
        kg = KnowledgeGraph()
        
        # Restore graph structure
        for node_id, node_data in graph_data['nodes'].items():
            kg.graph.add_node(node_id, **node_data)
        
        for u, v, edge_data in graph_data['edges']:
            kg.graph.add_edge(u, v, **edge_data)
        
        # Restore text chunks
        kg.text_chunks = graph_data['text_chunks']
        
        # Restore embeddings if present
        if 'entity_embeddings' in graph_data:
            kg.entity_embeddings = {
                k: np.array(v) for k, v in graph_data['entity_embeddings'].items()
            }
        
        if 'chunk_embeddings' in graph_data:
            kg.chunk_embeddings = {
                int(k): np.array(v) for k, v in graph_data['chunk_embeddings'].items()
            }
        
        print(f"Loaded graph with {len(kg.graph.nodes)} nodes and {len(kg.graph.edges)} edges")
        return kg

class GraphRAGOptimizer:
    """
    Performance optimization utilities for Graph RAG
    """
    
    def __init__(self, graph_rag: GraphRAGSystem):
        self.system = graph_rag
        self.query_cache = {}  # Cache for frequent queries
        self.subgraph_cache = {}  # Cache for computed subgraphs
    
    def optimize_graph_structure(self):
        """
        Optimize graph structure for better performance
        
        Optimizations:
        1. Remove low-confidence relationships
        2. Merge similar entities
        3. Prune disconnected components
        """
        kg = self.system.kg
        original_edges = len(kg.graph.edges)
        
        # Remove low-confidence edges
        edges_to_remove = []
        for u, v, data in kg.graph.edges(data=True):
            weight = data.get('weight', 1.0)
            if weight < 0.3:  # Threshold for minimum confidence
                edges_to_remove.append((u, v))
        
        kg.graph.remove_edges_from(edges_to_remove)
        
        # Remove isolated nodes (nodes with no connections)
        isolated_nodes = list(nx.isolates(kg.graph))
        kg.graph.remove_nodes_from(isolated_nodes)
        
        print(f"Graph optimization complete:")
        print(f"- Removed {len(edges_to_remove)} low-confidence edges")
        print(f"- Removed {len(isolated_nodes)} isolated nodes")
        print(f"- Edges: {original_edges} → {len(kg.graph.edges)}")
    
    def cached_query(self, question: str, cache_ttl: int = 3600) -> Dict:
        """
        Implement query caching for better performance
        
        Useful for:
        - Repeated queries
        - Similar queries
        - Demo/testing scenarios
        """
        # Simple cache key based on question
        cache_key = question.lower().strip()
        
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            cached_result['from_cache'] = True
            return cached_result
        
        # Not in cache, perform actual query
        result = self.system.query(question)
        
        # Cache the result
        self.query_cache[cache_key] = result
        result['from_cache'] = False
        
        return result