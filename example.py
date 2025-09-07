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
from advance import AdvancedGraphRAG

def create_comprehensive_example():
    """
    Complete example showing all Graph RAG capabilities
    """
    
    # Mock system for demonstration (replace with real OpenAI key)
    class DemoGraphRAGSystem(AdvancedGraphRAG):
        def __init__(self):
            self.processor = DocumentProcessor()
            self.kg = KnowledgeGraph()
            self.retriever = None
        
        def get_graph_stats(self) -> Dict:
            """Return statistics about the knowledge graph"""
            entity_types = {}
            for node, data in self.kg.graph.nodes(data=True):
                etype = data.get("type", "unknown")
                entity_types[etype] = entity_types.get(etype, 0) + 1

            stats = {
                "total_entities": len(self.kg.graph.nodes),
                "total_relations": len(self.kg.graph.edges),
                "total_chunks": len(self.kg.text_chunks),
                "entity_types": entity_types,
                "avg_degree": (sum(dict(self.kg.graph.degree()).values()) / len(self.kg.graph.nodes))
                            if self.kg.graph.nodes else 0
            }
            return stats

        def _generate_response(self, question: str, context: str) -> str:
            """Mock LLM response for demo"""
            # Analyze context to provide realistic response
            if "$100,000" in context and "FutureTech" in context:
                return "Based on the grant proposal, FutureTech Foundation is requesting $100,000 for their Youth Empowerment Through Digital Skills Training program. This funding will support training 500 underserved youth in coding, graphic design, and digital marketing."
            elif "robotics" in context.lower() and "$2,000" in context:
                return "According to the proposal, ABC Technical Institute's Introduction to Robotics workshop costs $2,000 total, which includes $1,200 for robotics kits, $600 for instructor fees, and $200 for materials."
            else:
                return f"Based on the provided context, I can help answer questions about the information contained in the knowledge graph and documents."
        
        def _generate_reasoning_response(self, question: str, context: str, reasoning_paths: List[Dict]) -> str:
            """Mock reasoning response"""
            reasoning_info = ""
            if reasoning_paths:
                reasoning_info = f" I found {len(reasoning_paths)} reasoning chains connecting the relevant entities."
            
            return self._generate_response(question, context) + reasoning_info
    
    # Initialize system
    system = DemoGraphRAGSystem()
    
    # Sample documents (using the uploaded documents as reference)
    documents = [
        """
        Grant Proposal: Youth Empowerment Through Digital Skills Training
        Submitted by: FutureTech Foundation
        Date: March 3, 2025
        Submitted to: National Education Grant Program
        
        FutureTech Foundation is requesting a $100,000 grant to launch the Youth Empowerment
        Through Digital Skills Training program. This initiative aims to provide free digital skills
        training to underserved youth in urban and rural communities, equipping them with essential
        knowledge in coding, graphic design, and digital marketing.
        
        Project Objectives:
        - Train 500 underserved youth in digital skills within one year
        - Provide certifications in coding, graphic design, and digital marketing
        - Partner with local businesses and tech firms to create job placement opportunities
        - Establish mentorship programs with industry professionals
        
        Budget Breakdown:
        - Training Materials & Laptops: $40,000
        - Trainer & Staff Salaries: $30,000
        - Marketing & Outreach: $10,000
        - Student Stipends & Certification Fees: $15,000
        - Program Evaluation & Reporting: $5,000
        """,
        """
        Proposal for Educational Program: Introduction to Robotics Workshop
        Submitted by: ABC Technical Institute
        Date: March 3, 2025
        Submitted to: Students and Faculty of XYZ High School
        
        ABC Technical Institute proposes to conduct an Introduction to Robotics workshop at XYZ
        High School. This week-long workshop is designed to introduce students to the basics of
        robotics engineering, programming, and real-world applications.
        
        Workshop Details:
        Duration: 1 week (5 sessions, 3 hours each)
        Target Audience: Students in grades 9-12 interested in STEM
        Location: XYZ High School, Computer Lab Room 101
        
        Budget:
        Total Cost: $2,000
        - Robotics kits: $1,200
        - Instructor fees: $600
        - Materials and handouts: $200
        
        The workshop will cover robotics history, engineering components, Python programming,
        hands-on robot building, and final presentations by student teams.
        """
    ]
    
    print("=== BUILDING KNOWLEDGE BASE ===")
    system.build_knowledge_base(documents)
    
    # Display graph statistics
    stats = system.get_graph_stats()
    print("\n=== KNOWLEDGE GRAPH STATISTICS ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== TESTING DIFFERENT QUERY TYPES ===")
    
    # Test different types of queries
    test_cases = [
        {
            'query': "What is the total budget requested by FutureTech Foundation?",
            'type': 'direct_fact_retrieval',
            'explanation': "Tests basic entity-fact retrieval"
        },
        {
            'query': "Which organizations are submitting proposals and what are their costs?",
            'type': 'multi_entity_comparison',
            'explanation': "Tests ability to compare multiple entities"
        },
        {
            'query': "What types of training programs are mentioned in the documents?",
            'type': 'concept_aggregation', 
            'explanation': "Tests ability to aggregate related concepts"
        },
        {
            'query': "How do the budgets of both programs compare?",
            'type': 'relationship_analysis',
            'explanation': "Tests ability to analyze relationships between entities"
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n--- {test_case['type'].upper()} ---")
        print(f"Query: {test_case['query']}")
        print(f"Purpose: {test_case['explanation']}")
        
        # Standard query
        standard_result = system.query(test_case['query'])
        
        # Multi-hop reasoning query
        reasoning_result = system.multi_hop_reasoning(test_case['query'])
        
        print(f"\nStandard Answer: {standard_result['answer']}")
        print(f"Reasoning Answer: {reasoning_result['answer']}")
        
        # Show retrieval information
        print(f"\nRetrieval Info:")
        print(f"- Chunks retrieved: {standard_result['retrieval_info']['relevant_chunks']}")
        print(f"- Entities involved: {standard_result['retrieval_info']['relevant_entities']}")
        print(f"- Reasoning paths: {reasoning_result.get('path_count', 0)}")
        
        results.append({
            'test_case': test_case,
            'standard_result': standard_result,
            'reasoning_result': reasoning_result
        })
    
    return system, results

def demonstrate_graph_analysis():
    """
    Demonstrate graph analysis capabilities
    """
    system, results = create_comprehensive_example()
    
    print("\n" + "="*60)
    print("DETAILED GRAPH ANALYSIS")
    print("="*60)
    
    # Analyze the knowledge graph structure
    kg = system.kg
    
    print(f"\nGraph Structure Analysis:")
    print(f"- Total entities: {len(kg.graph.nodes)}")
    print(f"- Total relationships: {len(kg.graph.edges)}")
    print(f"- Graph density: {nx.density(kg.graph):.3f}")
    print(f"- Connected components: {nx.number_weakly_connected_components(kg.graph)}")
    
    # Show entity types distribution
    entity_types = {}
    for node, data in kg.graph.nodes(data=True):
        entity_type = data.get('type', 'unknown')
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    print(f"\nEntity Type Distribution:")
    for etype, count in sorted(entity_types.items()):
        print(f"- {etype}: {count}")
    
    # Show relationship types
    relation_types = {}
    for u, v, data in kg.graph.edges(data=True):
        relation = data.get('relation', 'unknown')
        relation_types[relation] = relation_types.get(relation, 0) + 1
    
    print(f"\nRelationship Type Distribution:")
    for rtype, count in sorted(relation_types.items()):
        print(f"- {rtype}: {count}")
    
    # Find central entities (most connected)
    degree_centrality = nx.degree_centrality(kg.graph)
    top_central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print(f"\nMost Central Entities:")
    for entity, centrality in top_central:
        print(f"- {entity}: {centrality:.3f}")
    
    return system

# Run the comprehensive example
if __name__ == "__main__":
    system = demonstrate_graph_analysis()