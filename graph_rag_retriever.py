from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Set
from knowledge_graph import KnowledgeGraph
from document_processor import DocumentProcessor

class GraphRAGRetriever:
    """Handles retrieval using both semantic similarity and graph structure"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
    
    def retrieve_relevant_chunks(self, query: str, processor: DocumentProcessor, top_k: int = 3) -> List[Dict]:
        """Retrieve chunks most similar to query"""
        if not self.kg.chunk_embeddings:
            return []
        
        query_embedding = processor.encoder.encode(query)
        
        results = []
        for chunk_id, chunk_embedding in self.kg.chunk_embeddings.items():
            similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            
            if similarity > 0.1:  # Threshold
                results.append({
                    'chunk_id': chunk_id,
                    'text': self.kg.text_chunks[chunk_id]['text'],
                    'score': float(similarity),
                    'entities': self.kg.text_chunks[chunk_id]['entities']
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
    
    def retrieve_relevant_entities(self, query: str, processor: DocumentProcessor, top_k: int = 3) -> List[Dict]:
        """Retrieve entities most similar to query"""
        if not self.kg.entity_embeddings:
            return []
        
        query_embedding = processor.encoder.encode(query)
        
        results = []
        for entity, entity_embedding in self.kg.entity_embeddings.items():
            similarity = cosine_similarity([query_embedding], [entity_embedding])[0][0]
            
            if similarity > 0.1:
                results.append({
                    'entity': entity,
                    'score': float(similarity),
                    'type': self.kg.graph.nodes[entity].get('type', 'unknown')
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
    
    def graph_enhanced_retrieval(self, query: str, processor: DocumentProcessor) -> Dict:
        """Core Graph RAG retrieval combining chunks, entities, and graph structure"""
        
        # Get relevant chunks (traditional RAG)
        relevant_chunks = self.retrieve_relevant_chunks(query, processor)
        
        # Get relevant entities (Graph RAG enhancement)
        relevant_entities = self.retrieve_relevant_entities(query, processor)
        
        # Expand entities using graph structure
        expanded_entities = set()
        for entity_info in relevant_entities:
            entity = entity_info['entity']
            expanded_entities.add(entity)
            
            # Add neighbors (1-hop expansion)
            if entity in self.kg.graph:
                neighbors = list(self.kg.graph.neighbors(entity))
                predecessors = list(self.kg.graph.predecessors(entity))
                expanded_entities.update(neighbors[:2])
                expanded_entities.update(predecessors[:2])
        
        # Add entities from relevant chunks
        for chunk_info in relevant_chunks:
            expanded_entities.update(chunk_info.get('entities', []))
        
        # Get subgraph
        subgraph = self.kg.get_subgraph(list(expanded_entities))
        
        return {
            'relevant_chunks': relevant_chunks,
            'relevant_entities': relevant_entities,
            'expanded_entities': list(expanded_entities),
            'subgraph': subgraph
        }
