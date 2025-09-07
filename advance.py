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
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()


class AdvancedGraphRAG(GraphRAGSystem):
    def __init__(self, model: str = "gemini-2.0-flash"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel(model)
    
    def multi_hop_reasoning(self, query: str, max_hops: int = 3) -> Dict:
        """
        Perform multi-hop reasoning through the knowledge graph
        
        Multi-hop reasoning allows the system to answer complex questions
        that require connecting multiple pieces of information:
        
        Example: "How much funding does the organization led by John Smith need?"
        Hop 1: Find "John Smith" 
        Hop 2: Find "John Smith" → "leads" → "FutureTech Foundation"
        Hop 3: Find "FutureTech Foundation" → "requests" → "$100,000"
        """
        
        print(f"Performing multi-hop reasoning for: {query}")
        
        # STEP 1: Extract entities from query
        query_analysis = self.processor.extract_entities_and_relations(query)
        query_entities = [e['text'] for e in query_analysis['entities']]
        
        if not query_entities:
            print("No entities found in query, falling back to standard retrieval")
            return self.query(query)
        
        print(f"Starting entities: {query_entities}")
        
        # STEP 2: Find reasoning paths from each query entity
        all_reasoning_paths = []
        for entity in query_entities:
            if entity in self.kg.graph:
                paths = self._find_reasoning_paths(entity, max_hops)
                all_reasoning_paths.extend(paths)
        
        print(f"Found {len(all_reasoning_paths)} reasoning paths")
        
        # STEP 3: Score and rank reasoning paths
        ranked_paths = self._rank_reasoning_paths(all_reasoning_paths, query)
        
        # STEP 4: Build enhanced context using reasoning paths
        enhanced_context = self._build_reasoning_context(query, ranked_paths)
        
        # STEP 5: Generate response with reasoning explanation
        response = self._generate_reasoning_response(query, enhanced_context, ranked_paths)
        
        return {
            'answer': response,
            'reasoning_paths': ranked_paths[:5],  # Top 5 paths
            'context': enhanced_context,
            'path_count': len(all_reasoning_paths)
        }
    
    def _find_reasoning_paths(self, start_entity: str, max_hops: int) -> List[Dict]:
        """
        Find all possible reasoning paths from a starting entity
        
        A reasoning path is a sequence of connected entities that might
        lead to relevant information for answering the query
        """
        paths = []
        
        def dfs_with_context(current_entity, path, depth, visited_relations):
            """
            Depth-first search to find reasoning paths
            We track relations to understand WHY entities are connected
            """
            if depth >= max_hops:
                return
            
            # Explore outgoing edges (entities this entity relates to)
            for neighbor in self.kg.graph.neighbors(current_entity):
                if neighbor not in [p['entity'] for p in path]:  # Avoid cycles
                    # Get relationship information
                    edge_data = self.kg.graph.get_edge_data(current_entity, neighbor)
                    relation = edge_data.get('relation', 'related_to') if edge_data else 'related_to'
                    
                    new_path_step = {
                        'entity': neighbor,
                        'relation_from_previous': relation,
                        'hop_number': depth + 1
                    }
                    
                    new_path = path + [new_path_step]
                    
                    # Calculate path relevance score
                    path_score = self._calculate_path_relevance(new_path)
                    
                    paths.append({
                        'path': new_path,
                        'start_entity': start_entity,
                        'end_entity': neighbor,
                        'length': len(new_path),
                        'score': path_score
                    })
                    
                    # Continue searching if we haven't reached max depth
                    if depth < max_hops - 1:
                        dfs_with_context(neighbor, new_path, depth + 1, visited_relations + [relation])
        
        # Start DFS from the given entity
        initial_path = [{'entity': start_entity, 'relation_from_previous': None, 'hop_number': 0}]
        dfs_with_context(start_entity, initial_path, 0, [])
        
        return paths
    
    def _calculate_path_relevance(self, path: List[Dict]) -> float:
        """
        Score reasoning paths based on multiple factors
        
        Factors that make a path more relevant:
        1. Shorter paths (more direct connections)
        2. Strong relationship types (like 'leads', 'requests', 'owns')
        3. Important entity types (PERSON, ORG, MONEY)
        """
        if not path:
            return 0.0
        
        base_score = 1.0
        
        # FACTOR 1: Path length penalty (shorter is better)
        length_penalty = 0.8 ** (len(path) - 1)
        
        # FACTOR 2: Relationship strength
        strong_relations = ['leads', 'requests', 'owns', 'manages', 'funds', 'creates', 'develops']
        relation_bonus = 0.0
        
        for step in path[1:]:  # Skip first step (no relation)
            relation = step.get('relation_from_previous', '').lower()
            if any(strong_rel in relation for strong_rel in strong_relations):
                relation_bonus += 0.3
        
        # FACTOR 3: Entity type importance
        important_types = ['PERSON', 'ORG', 'MONEY', 'PRODUCT']
        type_bonus = 0.0
        
        for step in path:
            entity = step['entity']
            if entity in self.kg.graph:
                entity_type = self.kg.graph.nodes[entity].get('type', '')
                if entity_type in important_types:
                    type_bonus += 0.2
        
        final_score = base_score * length_penalty + relation_bonus + type_bonus
        return min(final_score, 2.0)  # Cap at 2.0
    
    def _rank_reasoning_paths(self, paths: List[Dict], query: str) -> List[Dict]:
        """Rank reasoning paths by relevance to query"""
        # Sort paths by score (descending)
        sorted_paths = sorted(paths, key=lambda p: p['score'], reverse=True)
        
        # Add query relevance scoring
        query_embedding = self.processor.encoder.encode(query)
        
        for path_info in sorted_paths:
            # Create path text representation
            path_text = " -> ".join([step['entity'] for step in path_info['path']])
            path_embedding = self.processor.encoder.encode(path_text)
            
            # Calculate similarity to query
            similarity = cosine_similarity([query_embedding], [path_embedding])[0][0]
            path_info['query_similarity'] = float(similarity)
            
            # Combine scores
            path_info['final_score'] = (path_info['score'] + path_info['query_similarity']) / 2
        
        # Re-sort by final score
        return sorted(sorted_paths, key=lambda p: p['final_score'], reverse=True)
    
    def _build_reasoning_context(self, query: str, ranked_paths: List[Dict]) -> str:
        """
        Build context that includes reasoning chains
        
        This context helps the LLM understand not just WHAT information
        is relevant, but HOW different pieces connect together
        """
        context_parts = []
        
        # Add standard retrieval context
        standard_retrieval = self.retriever.graph_enhanced_retrieval(query)
        
        # Text chunks
        for i, chunk_info in enumerate(standard_retrieval['relevant_chunks']):
            context_parts.append(f"Document Source {i+1}:\n{chunk_info['text']}")
        
        # Reasoning paths
        if ranked_paths:
            context_parts.append("Reasoning Chains:")
            for i, path_info in enumerate(ranked_paths[:3]):  # Top 3 paths
                path_description = []
                for j, step in enumerate(path_info['path']):
                    if j == 0:
                        path_description.append(step['entity'])
                    else:
                        relation = step['relation_from_previous']
                        path_description.append(f"--{relation}--> {step['entity']}")
                
                path_text = " ".join(path_description)
                context_parts.append(f"Chain {i+1}: {path_text}")
        
        # Entity context
        if standard_retrieval['graph_context']:
            context_parts.append(f"Entity Overview:\n{standard_retrieval['graph_context']}")
        
        return "\n\n".join(context_parts)
    
    def _generate_reasoning_response(self, question: str, context: str, reasoning_paths: List[Dict]) -> str:
        """Generate response that explains the reasoning process using Gemini"""

        system_prompt = """You are an advanced AI assistant that uses knowledge graphs for reasoning.

When answering questions:
1. Use the document sources as your primary evidence
2. Use the reasoning chains to understand how information connects
3. Explain your reasoning process when connections are important
4. Be specific about relationships between entities
5. Acknowledge when reasoning chains support or contradict each other"""

        # Build reasoning explanation text from reasoning paths
        reasoning_explanation = ""
        if reasoning_paths:
            reasoning_explanation = "\n\nReasoning Process:\n"
            for i, path_info in enumerate(reasoning_paths[:2]):  # limit to first 2 paths
                path_text = " → ".join([step['entity'] for step in path_info['path']])
                reasoning_explanation += f"Connection {i+1}: {path_text}\n"

        user_prompt = f"""Context and Evidence:
{context}
{reasoning_explanation}

Question: {question}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. References specific evidence from the sources
3. Explains any important connections or relationships
4. Indicates confidence level in your answer

Answer:"""

        try:
            # Gemini takes prompts as a single string or list of "parts"
            response = self.client.generate_content(
                [system_prompt, user_prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,   # Low temperature for reasoning
                    max_output_tokens=700,
                )
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"