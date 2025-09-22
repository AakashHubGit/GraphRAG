import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Set
import os

# Import other classes
from document_processor import DocumentProcessor
from knowledge_graph import KnowledgeGraph
from graph_rag_retriever import GraphRAGRetriever

class GraphRAGSystem:
    """Main Graph RAG system integrating all components"""
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.processor = DocumentProcessor()
        self.kg = KnowledgeGraph()
        self.retriever = None
        
        # Initialize Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel(model)
    
    def build_knowledge_base(self, documents: List[str]):
        """Build the complete knowledge base"""
        print("Building knowledge graph...")
        self.kg.build_from_documents(documents, self.processor)
        self.retriever = GraphRAGRetriever(self.kg)
        print(f"Knowledge base built: {len(self.kg.graph.nodes)} entities, {len(self.kg.graph.edges)} relations")
    
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
    
    def multi_hop_reasoning(self, query: str, max_hops: int = 3) -> Dict:
        """Perform multi-hop reasoning through the knowledge graph"""
        # Extract entities from query
        query_analysis = self.processor.extract_entities_relations_llm(query)
        query_entities = [e['text'] for e in query_analysis['entities']]
        
        if not query_entities:
            return self.query(query)
        
        # Find reasoning paths
        all_paths = []
        for entity in query_entities:
            if entity in self.kg.graph:
                paths = self._find_reasoning_paths(entity, max_hops)
                all_paths.extend(paths)
        
        # Build enhanced context
        standard_result = self.query(query)
        
        if all_paths:
            reasoning_context = "\nReasoning Chains:\n"
            for i, path in enumerate(all_paths[:3]):
                path_text = " → ".join([step['entity'] for step in path['path']])
                reasoning_context += f"Chain {i+1}: {path_text}\n"
            
            enhanced_context = standard_result['context_used'] + reasoning_context
            response = self._generate_response(query, enhanced_context)
        else:
            response = standard_result['answer']
        
        return {
            'answer': response,
            'reasoning_paths': all_paths[:5],
            'path_count': len(all_paths)
        }
    
    def _find_reasoning_paths(self, start_entity: str, max_hops: int) -> List[Dict]:
        """Find reasoning paths from starting entity"""
        paths = []
        
        def dfs_paths(current_entity, path, depth):
            if depth >= max_hops:
                return
            
            for neighbor in self.kg.graph.neighbors(current_entity):
                if neighbor not in [p['entity'] for p in path]:
                    edge_data = self.kg.graph.get_edge_data(current_entity, neighbor)
                    relation = edge_data.get('relation', 'related_to') if edge_data else 'related_to'
                    
                    new_path = path + [{
                        'entity': neighbor,
                        'relation_from_previous': relation,
                        'hop_number': depth + 1
                    }]
                    
                    paths.append({
                        'path': new_path,
                        'start_entity': start_entity,
                        'end_entity': neighbor,
                        'length': len(new_path)
                    })
                    
                    if depth < max_hops - 1:
                        dfs_paths(neighbor, new_path, depth + 1)
        
        initial_path = [{'entity': start_entity, 'relation_from_previous': None, 'hop_number': 0}]
        dfs_paths(start_entity, initial_path, 0)
        
        return paths