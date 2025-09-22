import networkx as nx
from typing import List, Dict
from document_processor import DocumentProcessor

class KnowledgeGraph:
    """Manages the knowledge graph structure and operations"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_embeddings = {}
        self.text_chunks = {}
        self.chunk_embeddings = {}

    def add_entity(self, entity: str, entity_type: str, metadata: Dict = None):
        """Add entity to knowledge graph"""
        if not self.graph.has_node(entity):
            self.graph.add_node(entity, type=entity_type, metadata=metadata or {})
    
    def add_relation(self, subject: str, predicate: str, obj: str, weight: float = 1.0):
        """Add relationship between entities"""
        self.graph.add_edge(subject, obj, relation=predicate, weight=weight)
    
    def build_from_documents(self, documents: List[str], processor: DocumentProcessor):
        """Build knowledge graph from documents"""
        chunk_id = 0
        
        for doc_id, document in enumerate(documents):
            # Extract entities and relations
            doc_analysis = processor.extract_entities_relations_llm(document)
            
            # Add entities
            for entity in doc_analysis["entities"]:
                self.add_entity(
                    entity["text"], 
                    entity.get("label", "OTHER"), 
                    {"document_id": doc_id}
                )
            
            # Add relations
            for relation in doc_analysis["relations"]:
                self.add_relation(
                    relation["subject"], 
                    relation["predicate"], 
                    relation["object"], 
                    relation.get("confidence", 0.8)
                )
            
            # Process chunks
            chunks = processor.chunk_document(document)
            for chunk in chunks:
                # Store chunk with entities
                chunk_entities = processor.extract_entities_relations_llm(chunk)["entities"]
                self.text_chunks[chunk_id] = {
                    "text": chunk,
                    "document_id": doc_id,
                    "entities": [e["text"] for e in chunk_entities]
                }
                
                # Generate chunk embedding
                self.chunk_embeddings[chunk_id] = processor.encoder.encode(chunk)
                chunk_id += 1
        
        # Generate entity embeddings
        self._generate_entity_embeddings(processor)
    
    def _generate_entity_embeddings(self, processor: DocumentProcessor):
        """Generate context-aware embeddings for entities"""
        for entity in self.graph.nodes():
            # Create context from entity and its connections
            neighbors = list(self.graph.neighbors(entity))
            predecessors = list(self.graph.predecessors(entity))
            entity_type = self.graph.nodes[entity].get('type', '')
            
            context_parts = [entity]
            if neighbors:
                context_parts.append("connected to: " + ", ".join(neighbors[:3]))
            if predecessors:
                context_parts.append("mentioned with: " + ", ".join(predecessors[:3]))
            if entity_type:
                context_parts.append(f"type: {entity_type}")
            
            context = " | ".join(context_parts)
            self.entity_embeddings[entity] = processor.encoder.encode(context)
    
    def get_subgraph(self, entities: List[str], expand_distance: int = 1) -> nx.DiGraph:
        """Extract subgraph around given entities"""
        all_entities = set(entities)
        
        # Expand to include neighboring entities
        for entity in entities:
            if entity in self.graph:
                for distance in range(1, expand_distance + 1):
                    for node in self.graph.nodes():
                        try:
                            path_length = nx.shortest_path_length(self.graph, entity, node)
                            if path_length == distance:
                                all_entities.add(node)
                        except nx.NetworkXNoPath:
                            continue
        
        # Return subgraph with valid entities
        valid_entities = [e for e in all_entities if e in self.graph]
        return self.graph.subgraph(valid_entities)