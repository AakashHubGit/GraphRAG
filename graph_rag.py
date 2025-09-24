# graph_rag.py - Fixed version with single initialization

import spacy
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Set
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class DocumentProcessor:
    """Handles text processing, chunking, and embedding generation"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback if spaCy model not available
            self.nlp = None
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def chunk_document(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split document into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def extract_entities_relations_llm(self, text: str) -> Dict:
        """Extract entities and relations using Gemini LLM"""
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""
        Extract entities and relations from this text for a knowledge graph.

        Return JSON with:
        1. "entities": [{{"text": "entity_name", "label": "PERSON|ORG|GPE|MONEY|DATE|OTHER"}}]
        2. "relations": [{{"subject": "entity1", "predicate": "relation_type", "object": "entity2", "confidence": 0.0-1.0}}]

        Rules:
        - Extract all meaningful entities
        - Connect every entity to at least one other
        - Use descriptive predicates like "leads", "requests", "costs", "located_in"
        - Ensure high connectivity

        Text: "{text[:2000]}"  # Limit text length
        """

        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean response text
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            result = json.loads(response_text.strip())
            return result
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return {"entities": [], "relations": []}

class KnowledgeGraph:
    """Manages the knowledge graph structure and operations"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_embeddings = {}
        self.text_chunks = {}
        self.chunk_embeddings = {}
        self.is_built = False  # Track if graph is already built

    def add_entity(self, entity: str, entity_type: str, metadata: Dict = None):
        """Add entity to knowledge graph"""
        if not self.graph.has_node(entity):
            self.graph.add_node(entity, type=entity_type, metadata=metadata or {})
    
    def add_relation(self, subject: str, predicate: str, obj: str, weight: float = 1.0):
        """Add relationship between entities"""
        self.graph.add_edge(subject, obj, relation=predicate, weight=weight)
    
    def build_from_documents(self, documents: List[str], processor: DocumentProcessor):
        """Build knowledge graph from documents - only if not already built"""
        if self.is_built:
            print("Knowledge graph already built. Skipping...")
            return
            
        chunk_id = 0
        
        for doc_id, document in enumerate(documents):
            print(f"Processing document {doc_id + 1}/{len(documents)}...")
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
        self.is_built = True
        print(f"Knowledge base built: {len(self.graph.nodes)} entities, {len(self.graph.edges)} relations")
    
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


class GraphRAGSystem:
    """Main Graph RAG system integrating all components"""
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.processor = DocumentProcessor()
        self.kg = KnowledgeGraph()
        self.retriever = None
        self.is_initialized = False  # Track initialization
        
        # Initialize Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel(model)
    
    def build_knowledge_base(self, documents: List[str]):
        """Build the complete knowledge base - only once"""
        if self.is_initialized:
            print("Graph RAG system already initialized. Skipping...")
            return
            
        print("Building knowledge graph...")
        self.kg.build_from_documents(documents, self.processor)
        self.retriever = GraphRAGRetriever(self.kg)
        self.is_initialized = True
    
    def query(self, question: str) -> Dict:
        """Main query function implementing Graph RAG"""
        if not self.retriever:
            raise ValueError("Knowledge base not built. Call build_knowledge_base() first.")
        
        # Retrieve relevant information
        retrieval_result = self.retriever.graph_enhanced_retrieval(question, self.processor)
        
        # Build context
        context_parts = []
        
        # Add document chunks
        for i, chunk_info in enumerate(retrieval_result['relevant_chunks']):
            context_parts.append(f"Source {i+1}: {chunk_info['text']}")
        
        # Add graph relationships
        graph_facts = []
        for u, v, data in retrieval_result['subgraph'].edges(data=True):
            relation = data.get('relation', 'related_to')
            graph_facts.append(f"{u} --[{relation}]--> {v}")
        
        if graph_facts:
            context_parts.append(f"Knowledge Graph Facts:\n" + "\n".join(graph_facts[:10]))
        
        context = "\n\n".join(context_parts)
        
        # Generate response
        response = self._generate_response(question, context)
        
        return {
            'answer': response,
            'context_used': context,
            'retrieval_info': {
                'chunks_found': len(retrieval_result['relevant_chunks']),
                'entities_found': len(retrieval_result['relevant_entities']),
                'graph_nodes': len(retrieval_result['subgraph'].nodes())
            }
        }
    
    def _generate_response(self, question: str, context: str) -> str:
        """Generate response using Gemini"""
        prompt = f"""You are an AI assistant that answers questions using both document sources and knowledge graph relationships.

Context:
{context}

Question: {question}

Instructions:
1. Use document sources as primary evidence
2. Use graph relationships to understand connections between entities
3. Be specific and reference your sources
4. If information is incomplete, say so

Answer:"""

        try:
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=500
                )
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

def get_pdf_text(pdf_docs):
    """Extract text from PDF files"""
    from PyPDF2 import PdfReader
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into chunks"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_text(text)