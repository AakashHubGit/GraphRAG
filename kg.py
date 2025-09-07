import spacy
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Set
import json
import re
from process import DocumentProcessor
from llm import extract_with_gemini


class KnowledgeGraph:
    def __init__(self):
        # NetworkX Directed Graph to store entities and relationships
        self.graph = nx.DiGraph()
        
        # Storage for embeddings and text
        self.entity_embeddings = {}    # Entity name → embedding vector
        self.text_chunks = {}         # Chunk ID → chunk info
        self.chunk_embeddings = {}    # Chunk ID → embedding vector

   
    def add_entity(self, entity: str, entity_type: str, metadata: Dict = None):
        """
        Add entity to knowledge graph
        
        Each entity becomes a node in the graph with:
        - Unique identifier (the entity text)
        - Type (PERSON, ORG, etc.)
        - Metadata (source document, confidence, etc.)
        """
        if not self.graph.has_node(entity):
            node_data = {
                'type': entity_type,
                'metadata': metadata or {}
            }
            self.graph.add_node(entity, **node_data)
    
    def add_relation(self, subject: str, predicate: str, obj: str, weight: float = 1.0):
        """
        Add relationship between entities
        
        Creates directed edges in the graph:
        subject --[predicate]--> object
        
        Weight represents relationship strength/confidence
        """
        self.graph.add_edge(subject, obj, relation=predicate, weight=weight)
    
    def build_from_documents(self, documents: List[str],processor: DocumentProcessor, use_llm: bool = True):
        """
        Build knowledge graph using either spaCy (default) or Gemini LLM.
        """
        chunk_id = 0
        for doc_id, document in enumerate(documents):
            clean_text = document.strip()
            print("Testing from documents: ",clean_text)
            # === Use LLM or spaCy ===
            if use_llm:
                doc_analysis = extract_with_gemini(clean_text)
            else:
                doc_analysis = processor.extract_entities_and_relations(clean_text)
            
            # Add entities
            for entity in doc_analysis["entities"]:
                self.add_entity(
                    entity["text"], 
                    entity.get("label", "unknown"), 
                    {"document_id": doc_id, "source": "document"}
                )
            
            # Add relations
            for relation in doc_analysis["relations"]:
                self.add_relation(
                    relation["subject"], 
                    relation["predicate"], 
                    relation["object"], 
                    relation.get("confidence", 0.8)
                )
            
            # Chunk + embed as before
            chunks = processor.chunk_document(clean_text)
            for chunk in chunks:
                self.text_chunks[chunk_id] = {
                    "text": chunk,
                    "document_id": doc_id,
                    "entities": []
                }
                chunk_entities = extract_with_gemini(chunk)["entities"] if use_llm else []
                self.text_chunks[chunk_id]["entities"] = [e["text"] for e in chunk_entities]
                
                chunk_embedding = processor.encoder.encode(chunk)
                self.chunk_embeddings[chunk_id] = chunk_embedding
                chunk_id += 1
        
        # Generate entity embeddings
        self._generate_entity_embeddings(processor)

    def _generate_entity_embeddings(self, processor: DocumentProcessor):
        """
        Generate embeddings for entities using their graph context
        
        Why context-aware embeddings?
        Instead of just embedding the entity name, we include its relationships
        This makes the embedding more informative about the entity's role/context
        """
        for entity in self.graph.nodes():
            # Get entity's neighbors (connected entities)
            neighbors = list(self.graph.neighbors(entity))
            predecessors = list(self.graph.predecessors(entity))
            
            # Create rich context string
            context_parts = [entity]  # Start with entity itself
            
            # Add relationships context
            if neighbors:
                context_parts.append("connected to: " + ", ".join(neighbors[:5]))
            if predecessors:
                context_parts.append("mentioned with: " + ", ".join(predecessors[:5]))
            
            # Add entity type context
            entity_type = self.graph.nodes[entity].get('type', '')
            if entity_type:
                context_parts.append(f"type: {entity_type}")
            
            context = " | ".join(context_parts)
            
            # Generate embedding for this rich context
            embedding = processor.encoder.encode(context)
            self.entity_embeddings[entity] = embedding
    
    def format_subgraph_for_llm(self, subgraph: nx.DiGraph) -> str:
        facts = []
        for u, v, data in subgraph.edges(data=True):
            relation = data.get("relation", "related_to")
            facts.append(f"{u} --[{relation}]--> {v}")
        return "\n".join(facts)


    def find_related_entities(self, entity: str, max_distance: int = 2) -> List[str]:
        """
        Find entities related to given entity within graph distance
        
        Graph distance = minimum number of edges to traverse
        Distance 1 = direct neighbors
        Distance 2 = neighbors of neighbors
        """
        if entity not in self.graph:
            return []
        
        related = set()
        
        # Use BFS to find entities at each distance level
        for distance in range(1, max_distance + 1):
            for node in self.graph.nodes():
                try:
                    # Calculate shortest path length between entities
                    path_length = nx.shortest_path_length(self.graph, entity, node)
                    if path_length == distance:
                        related.add(node)
                except nx.NetworkXNoPath:
                    # No path exists between entities
                    continue
        
        return list(related)
    
    def get_subgraph(self, entities: List[str], expand_distance: int = 1) -> nx.DiGraph:
        """
        Extract subgraph around given entities
        
        This is crucial for providing focused context to the LLM
        Instead of the entire graph, we extract a relevant subgraph
        """
        all_entities = set(entities)
        
        # Expand to include related entities
        for entity in entities:
            if entity in self.graph:
                related = self.find_related_entities(entity, expand_distance)
                all_entities.update(related)
        
        # Filter entities that actually exist in our graph
        valid_entities = [e for e in all_entities if e in self.graph]
        
        # Return induced subgraph
        return self.graph.subgraph(valid_entities)
    
    def get_entity_paths(self, start_entity: str, end_entity: str, max_paths: int = 3) -> List[List[str]]:
        """Find paths between two entities (for reasoning chains)"""
        if start_entity not in self.graph or end_entity not in self.graph:
            return []
        
        try:
            # Find all simple paths (no cycles) between entities
            paths = list(nx.all_simple_paths(self.graph, start_entity, end_entity, cutoff=4))
            return paths[:max_paths]  # Return top paths
        except nx.NetworkXNoPath:
            return []