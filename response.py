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
import openai
from typing import Optional
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class GraphRAGSystem:
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.processor = DocumentProcessor()
        self.kg = KnowledgeGraph()
        self.retriever = None
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = model
        self.client = genai.GenerativeModel(model)

    
    def build_knowledge_base(self, documents: List[str]):
        """Initialize the entire system with documents"""
        print("Building knowledge graph from documents...")
        self.kg.build_from_documents(documents, self.processor)
        self.retriever = GraphRAGRetriever(self.kg)
        print("Knowledge base construction complete!")
    
    def query(self, question: str, use_graph_context: bool = True, max_context_length: int = 2000) -> Dict:
        """
        Main query function - this is where Graph RAG magic happens
        
        Process:
        1. Retrieve relevant information using graph-enhanced retrieval
        2. Combine text chunks + graph context
        3. Generate response using LLM
        4. Return comprehensive result
        """
        if self.retriever is None:
            raise ValueError("Knowledge base not built. Call build_knowledge_base() first.")
        
        print(f"Processing query: {question}")
        
        # STEP 1: Graph-enhanced retrieval
        retrieval_result = self.retriever.graph_enhanced_retrieval(question)
        context_parts = []
        if retrieval_result['relevant_entities']:
            paths = []
            for e1 in retrieval_result['relevant_entities'][:3]:
                for e2 in retrieval_result['relevant_entities'][:3]:
                    if e1 != e2:
                        paths.extend(self.kg.get_entity_paths(e1['entity'], e2['entity']))
            if paths:
                path_strings = [" -> ".join(p) for p in paths]
                context_parts.append("Reasoning Paths:\n" + "\n".join(path_strings[:5]))

        # STEP 2: Build context for LLM
        
        # Add text chunk context (traditional RAG component)
        print(f"Found {len(retrieval_result['relevant_chunks'])} relevant chunks")
        for i, chunk_info in enumerate(retrieval_result['relevant_chunks']):
            context_parts.append(f"Source {i+1}: {chunk_info['text']}")
        
        # Add graph context (Graph RAG enhancement)
        if use_graph_context and retrieval_result['graph_context']:
            # print(f"Adding graph context with {len(retrieval_result['expanded_entities'])} entities")
            # context_parts.append(f"Knowledge Graph Context:\n{retrieval_result['graph_context']}")
            graph_facts = self.kg.format_subgraph_for_llm(retrieval_result['subgraph'])
            context_parts.append(f"Knowledge Graph Facts:\n{graph_facts}")
        
        # Combine and truncate context if too long
        full_context = "\n\n".join(context_parts)
        if len(full_context) > max_context_length:
            full_context = full_context[:max_context_length] + "...[truncated]"
        
        # STEP 3: Generate response using LLM
        response = self._generate_response(question, full_context)
        
        # STEP 4: Return comprehensive result
        return {
            'answer': response,
            'context_used': full_context,
            'retrieval_info': {
                'relevant_chunks': len(retrieval_result['relevant_chunks']),
                'relevant_entities': len(retrieval_result['relevant_entities']),
                'expanded_entities': len(retrieval_result['expanded_entities']),
                'graph_nodes_used': len(retrieval_result['subgraph'].nodes()) if retrieval_result['subgraph'] else 0
            },
            'detailed_sources': retrieval_result
        }
    
    def _generate_response(self, question: str, context: str) -> str:
        """
        Generate response using Google's Gemini LLM

        Prompt design:
        - Use both text + graph context
        - Handle incomplete information
        - Give structured, evidence-based answers
        """

        system_prompt = """
You are an AI assistant that answers questions using both textual content and structured knowledge graphs.
Instructions:
1. Use text sources for factual evidence.
2. Use graph facts and reasoning paths to connect entities (multi-hop reasoning).
3. Explicitly explain how solutions, resources, and goals are connected.
4. If graph facts are incomplete, infer logical dependencies based on context.
"""


        user_prompt = f"""Context Information:
{context}

Question: {question}

Based on the above context (both text sources and knowledge graph relationships), please provide a comprehensive answer to the question. Make sure to:
- Reference specific information from the sources
- Explain any relevant relationships or connections
- Be clear about the confidence level of your answer

Answer:"""

        try:
            # Gemini expects just a single input string or a list of "parts"
            response = self.client.generate_content(
                [system_prompt, user_prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    top_p=0.9,
                    max_output_tokens=600,
                )
            )
            print(response.text)
            return response.text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
        
    def explain_retrieval(self, question: str) -> Dict:
        """
        Explain how the system retrieved information for a query
        Useful for debugging and understanding system behavior
        """
        retrieval_result = self.retriever.graph_enhanced_retrieval(question)
        
        explanation = {
            'query_analysis': self.processor.extract_entities_and_relations(question),
            'chunk_retrieval': {
                'method': 'semantic_similarity',
                'chunks_found': len(retrieval_result['relevant_chunks']),
                'top_chunk_scores': [c['score'] for c in retrieval_result['relevant_chunks'][:3]]
            },
            'entity_retrieval': {
                'method': 'entity_embedding_similarity',
                'entities_found': len(retrieval_result['relevant_entities']),
                'top_entities': [(e['entity'], e['score']) for e in retrieval_result['relevant_entities'][:3]]
            },
            'graph_expansion': {
                'original_entities': len(retrieval_result['relevant_entities']),
                'expanded_entities': len(retrieval_result['expanded_entities']),
                'expansion_method': 'graph_neighbors_and_chunk_entities'
            }
        }
        
        return explanation