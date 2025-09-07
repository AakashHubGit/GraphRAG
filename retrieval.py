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

class GraphRAGRetriever:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.chunk_index = None      # FAISS index for chunks
        self.entity_index = None     # FAISS index for entities
        self._build_indices()
    
    def _build_indices(self):
        """
        Build FAISS indices for efficient similarity search
        
        FAISS (Facebook AI Similarity Search) enables:
        - Fast approximate nearest neighbor search
        - Scalable to millions of vectors
        - GPU acceleration support
        """
        
        # BUILD CHUNK INDEX
        if self.kg.chunk_embeddings:
            # Convert embeddings dict to numpy array
            chunk_embeddings = np.array(list(self.kg.chunk_embeddings.values()))
            dimension = chunk_embeddings.shape[1]  # Usually 384 for MiniLM
            
            # IndexFlatIP = Inner Product (cosine similarity after normalization)
            self.chunk_index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            # After normalization: inner product = cosine similarity
            faiss.normalize_L2(chunk_embeddings)
            self.chunk_index.add(chunk_embeddings)
            
            print(f"Built chunk index with {len(chunk_embeddings)} vectors")
        
        # BUILD ENTITY INDEX
        if self.kg.entity_embeddings:
            entity_embeddings = np.array(list(self.kg.entity_embeddings.values()))
            dimension = entity_embeddings.shape[1]
            
            self.entity_index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(entity_embeddings)
            self.entity_index.add(entity_embeddings)
            
            print(f"Built entity index with {len(entity_embeddings)} vectors")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve text chunks most similar to query
        
        Process:
        1. Convert query to embedding
        2. Search chunk index for similar chunks
        3. Return top-k results with metadata
        """
        if self.chunk_index is None:
            return []
        
        # Encode query using same model as chunks
        processor = DocumentProcessor()
        query_embedding = processor.encoder.encode(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search for similar chunks
        scores, indices = self.chunk_index.search(query_embedding, top_k)
        
        results = []
        chunk_ids = list(self.kg.chunk_embeddings.keys())
        
        # Process search results
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(chunk_ids) and score > 0.1:  # Minimum similarity threshold
                chunk_id = chunk_ids[idx]
                results.append({
                    'chunk_id': chunk_id,
                    'text': self.kg.text_chunks[chunk_id]['text'],
                    'score': float(score),
                    'entities': self.kg.text_chunks[chunk_id]['entities'],
                    'document_id': self.kg.text_chunks[chunk_id]['document_id']
                })
        
        return results
    
    def retrieve_relevant_entities(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve entities most similar to query
        
        This is unique to Graph RAG - we can find relevant entities
        even if they're not directly mentioned in the query
        """
        if not self.kg.entity_embeddings:
            return []
        
        processor = DocumentProcessor()
        query_embedding = processor.encoder.encode(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Get all entity embeddings
        entity_names = list(self.kg.entity_embeddings.keys())
        entity_embeddings = np.array(list(self.kg.entity_embeddings.values()))
        
        # Normalize for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        entity_norms = entity_embeddings / np.linalg.norm(entity_embeddings, axis=1, keepdims=True)
        
        # Calculate similarities
        similarities = np.dot(query_norm, entity_norms.T)[0]
        
        # Get top-k entities
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                entity_name = entity_names[idx]
                results.append({
                    'entity': entity_name,
                    'score': float(similarities[idx]),
                    'type': self.kg.graph.nodes[entity_name].get('type', 'unknown'),
                    'metadata': self.kg.graph.nodes[entity_name].get('metadata', {})
                })
        
        return results
    
    def graph_enhanced_retrieval(self, query: str, top_k_chunks: int = 3, top_k_entities: int = 3) -> Dict:
        """
        THE CORE OF GRAPH RAG
        
        Combines multiple retrieval strategies:
        1. Semantic similarity (like traditional RAG)
        2. Entity-based retrieval
        3. Graph structure expansion
        4. Relationship context
        """
        
        # STEP 1: Get relevant chunks (traditional RAG component)
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k_chunks)
        
        # STEP 2: Get relevant entities (graph component)
        relevant_entities = self.retrieve_relevant_entities(query, top_k_entities)
        
        # STEP 3: Expand entities using graph structure
        expanded_entities = set()
        
        # Add directly relevant entities
        for entity_info in relevant_entities:
            entity = entity_info['entity']
            expanded_entities.add(entity)
            
            # Add directly connected entities (1-hop neighbors)
            if entity in self.kg.graph:
                neighbors = list(self.kg.graph.neighbors(entity))
                predecessors = list(self.kg.graph.predecessors(entity))
                
                # Add top neighbors by connection strength
                all_connected = neighbors + predecessors
                expanded_entities.update(all_connected[:3])  # Limit expansion
        
        # STEP 4: Also add entities mentioned in relevant chunks
        for chunk_info in relevant_chunks:
            chunk_entities = chunk_info.get('entities', [])
            expanded_entities.update(chunk_entities)
        
        # STEP 5: Get subgraph for the expanded entity set
        subgraph = self.kg.get_subgraph(list(expanded_entities), expand_distance=1)
        
        # STEP 6: Format graph information as text context
        graph_context = self._format_graph_context(subgraph)
        
        return {
            'relevant_chunks': relevant_chunks,
            'relevant_entities': relevant_entities,
            'expanded_entities': list(expanded_entities),
            'subgraph': subgraph,
            'graph_context': graph_context
        }
    
    def _format_graph_context(self, subgraph: nx.DiGraph) -> str:
        """
        Convert graph structure to text that LLM can understand
        
        This is crucial - we need to represent graph information
        in a way that's useful for the LLM
        """
        if not subgraph.nodes():
            return ""
        
        context_parts = []
        
        # ENTITIES BY TYPE
        entities_by_type = {}
        for node, data in subgraph.nodes(data=True):
            entity_type = data.get('type', 'unknown')
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(node)
        
        for entity_type, entities in entities_by_type.items():
            if entities:
                context_parts.append(f"{entity_type.title()}s: {', '.join(entities)}")
        
        # RELATIONSHIPS
        relations = []
        relation_counts = {}
        
        for source, target, data in subgraph.edges(data=True):
            relation = data.get('relation', 'related_to')
            relation_text = f"{source} {relation} {target}"
            relations.append(relation_text)
            
            # Count relation types
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
        
        if relations:
            context_parts.append(f"Key relationships: {'; '.join(relations[:10])}")
        
        # GRAPH STATISTICS
        if len(subgraph.nodes()) > 2:
            context_parts.append(f"Graph contains {len(subgraph.nodes())} entities with {len(subgraph.edges())} relationships")
        
        return "\n".join(context_parts)