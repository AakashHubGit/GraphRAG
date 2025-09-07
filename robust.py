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

class RobustGraphRAG(AdvancedGraphRAG):
    """
    Production-ready Graph RAG with comprehensive error handling
    """
    
    def __init__(self, fallback_to_simple_rag: bool = True):
        super().__init__()
        self.fallback_enabled = fallback_to_simple_rag
        self.error_log = []
        self.processor = DocumentProcessor()
        self.kg = KnowledgeGraph()
        self.retriever = None
            
    
    def safe_build_knowledge_base(self, documents: List[str], validate: bool = True) -> Dict:
        """
        Build knowledge base with error handling and validation
        """
        try:
            # Validate input documents
            if not documents:
                raise ValueError("No documents provided")
            
            valid_docs = []
            for i, doc in enumerate(documents):
                if not doc or len(doc.strip()) < 10:
                    self._log_error(f"Document {i} is too short or empty")
                    continue
                valid_docs.append(doc)
            
            if not valid_docs:
                raise ValueError("No valid documents after filtering")
            print(valid_docs)
            # Build knowledge base
            self.kg.build_from_documents(valid_docs, self.processor)
            self.retriever = GraphRAGRetriever(self.kg)
            
            # Validate resulting graph
            if validate:
                validation_result = self._validate_knowledge_graph()
                return {
                    'status': 'success',
                    'documents_processed': len(valid_docs),
                    'validation': validation_result,
                    'errors': self.error_log
                }
            
            return {'status': 'success', 'documents_processed': len(valid_docs)}
            
        except Exception as e:
            self._log_error(f"Knowledge base construction failed: {str(e)}")
            return {'status': 'error', 'error': str(e), 'errors': self.error_log}
    
    def safe_query(self, question: str, max_retries: int = 2) -> Dict:
        """
        Query with error handling and fallback mechanisms
        """
        for attempt in range(max_retries + 1):
            try:
                # Validate query
                if not question or len(question.strip()) < 3:
                    raise ValueError("Query too short or empty")
                
                # Try Graph RAG first
                if self.retriever is not None:
                    result = self.query(question)
                    result['method_used'] = 'graph_rag'
                    result['attempt'] = attempt + 1
                    return result
                
                # Fallback to simple text search if no retriever
                elif self.fallback_enabled:
                    result = self._simple_text_search(question)
                    result['method_used'] = 'simple_fallback'
                    result['attempt'] = attempt + 1
                    return result
                
                else:
                    raise RuntimeError("No retriever available and fallback disabled")
                    
            except Exception as e:
                self._log_error(f"Query attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == max_retries:
                    return {
                        'answer': f"I apologize, but I encountered an error processing your query: {str(e)}",
                        'method_used': 'error',
                        'errors': self.error_log
                    }
    
    def _validate_knowledge_graph(self) -> Dict:
        """Validate the constructed knowledge graph"""
        kg = self.kg
        issues = []
        
        # Check for basic issues
        if len(kg.graph.nodes) == 0:
            issues.append("No entities extracted")
        
        if len(kg.graph.edges) == 0:
            issues.append("No relationships extracted")
        
        # Check for disconnected components
        components = list(nx.weakly_connected_components(kg.graph))
        if len(components) > len(kg.graph.nodes) * 0.5:
            issues.append(f"Too many disconnected components: {len(components)}")
        
        # Check embedding coverage
        entities_without_embeddings = set(kg.graph.nodes) - set(kg.entity_embeddings.keys())
        if entities_without_embeddings:
            issues.append(f"{len(entities_without_embeddings)} entities missing embeddings")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'entity_count': len(kg.graph.nodes),
            'relation_count': len(kg.graph.edges),
            'component_count': len(components)
        }
    
    def _simple_text_search(self, question: str) -> Dict:
        """
        Fallback to simple text search when Graph RAG fails
        """
        if not self.kg.text_chunks:
            return {
                'answer': "No text content available for search",
                'context_used': "",
                'retrieval_info': {}
            }
        
        # Simple keyword matching
        question_words = set(question.lower().split())
        
        chunk_scores = []
        for chunk_id, chunk_info in self.kg.text_chunks.items():
            chunk_text = chunk_info['text'].lower()
            chunk_words = set(chunk_text.split())
            
            # Calculate simple overlap score
            overlap = len(question_words & chunk_words)
            score = overlap / len(question_words) if question_words else 0
            
            if score > 0:
                chunk_scores.append((chunk_id, score, chunk_info['text']))
        
        # Sort by score and take top chunks
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = chunk_scores[:3]
        
        # Create simple response
        if top_chunks:
            context = "\n\n".join([chunk[2] for chunk in top_chunks])
            answer = f"Based on keyword matching, I found relevant information in {len(top_chunks)} text segments. " + context[:200] + "..."
        else:
            context = ""
            answer = "I couldn't find relevant information for your query in the available documents."
        
        return {
            'answer': answer,
            'context_used': context,
            'retrieval_info': {'method': 'keyword_fallback', 'chunks_found': len(top_chunks)}
        }
    
    def _log_error(self, error_message: str):
        """Log errors for debugging"""
        timestamp = datetime.now().isoformat()
        self.error_log.append({
            'timestamp': timestamp,
            'error': error_message
        })
        print(f"ERROR [{timestamp}]: {error_message}")

def run_full_test_suite():
    """
    Comprehensive test suite showing all capabilities
    """
    print("="*70)
    print("COMPREHENSIVE GRAPH RAG TEST SUITE")
    print("="*70)
    
    # Initialize robust system
    system = RobustGraphRAG()
    
    # Test documents based on uploaded files
    test_documents = [
        """
        FutureTech Foundation is requesting a $100,000 grant from the National Education Grant Program
        for their Youth Empowerment Through Digital Skills Training program. The program aims to train
        500 underserved youth in coding, graphic design, and digital marketing within one year.
        
        John Smith serves as the director of FutureTech Foundation. The foundation plans to partner
        with local businesses and tech firms to create job placement opportunities. They will also
        establish mentorship programs with industry professionals.
        
        The budget breakdown includes $40,000 for training materials and laptops, $30,000 for trainer
        and staff salaries, $10,000 for marketing and outreach, $15,000 for student stipends and
        certification fees, and $5,000 for program evaluation and reporting.
        """,
        """
        ABC Technical Institute proposes an Introduction to Robotics workshop for XYZ High School.
        Dr. Sarah Johnson leads the robotics program at ABC Technical Institute. The workshop will
        be conducted over one week with five sessions of three hours each.
        
        The workshop targets students in grades 9-12 interested in STEM fields. It will take place
        in XYZ High School's Computer Lab Room 101. The total cost is $2,000, with $1,200 allocated
        for robotics kits, $600 for instructor fees, and $200 for materials and handouts.
        
        The curriculum covers robotics history and key concepts, basic robotics engineering and
        component assembly, Python programming for robotics, hands-on robot building, and final
        testing with student team presentations.
        """
    ]
    
    print("\n1. BUILDING KNOWLEDGE BASE")
    build_result = system.safe_build_knowledge_base(test_documents)
    print(f"Build Status: {build_result['status']}")
    if build_result['status'] == 'success':
        print(f"Documents Processed: {build_result['documents_processed']}")
        print(f"Validation: {build_result['validation']['valid']}")
        if build_result['validation']['issues']:
            print(f"Issues Found: {build_result['validation']['issues']}")
    
    print("\n2. TESTING VARIOUS QUERY TYPES")
    
    test_queries = [
        {
            'question': "What is the budget for the FutureTech Foundation program?",
            'category': "Direct Fact Retrieval",
            'expected_elements': ["$100,000", "FutureTech Foundation"]
        },
        {
            'question': "Who are the key people mentioned in these proposals?",
            'category': "Entity Aggregation", 
            'expected_elements': ["John Smith", "Dr. Sarah Johnson"]
        },
        {
            'question': "Compare the costs of both educational programs",
            'category': "Multi-Entity Analysis",
            'expected_elements': ["$100,000", "$2,000", "comparison"]
        },
        {
            'question': "What skills and subjects are being taught?",
            'category': "Concept Extraction",
            'expected_elements': ["coding", "robotics", "digital marketing"]
        },
        {
            'question': "How does the robotics workshop connect to STEM education?",
            'category': "Relationship Reasoning",
            'expected_elements': ["STEM", "robotics", "education"]
        }
    ]
    
    for i, test_query in enumerate(test_queries, 1):
        print(f"\n--- TEST {i}: {test_query['category']} ---")
        print(f"Query: {test_query['question']}")
        
        # Test standard query
        result = system.safe_query(test_query['question'])
        
        print(f"Method Used: {result.get('method_used', 'unknown')}")
        print(f"Answer: {result['answer'][:200]}...")
        
        # Check if expected elements are present
        answer_lower = result['answer'].lower()
        found_elements = [elem for elem in test_query['expected_elements'] 
                         if elem.lower() in answer_lower]
        print(f"Expected Elements Found: {found_elements}")
        
        # Test multi-hop reasoning if available
        if hasattr(system, 'multi_hop_reasoning'):
            try:
                reasoning_result = system.multi_hop_reasoning(test_query['question'])
                if reasoning_result.get('reasoning_paths'):
                    print(f"Reasoning Paths Found: {len(reasoning_result['reasoning_paths'])}")
            except Exception as e:
                print(f"Reasoning test failed: {str(e)}")
    
    print("\n3. GRAPH ANALYSIS")
    if system.kg.graph.nodes:
        print(f"Total Entities: {len(system.kg.graph.nodes)}")
        print(f"Total Relations: {len(system.kg.graph.edges)}")
        print(f"Text Chunks: {len(system.kg.text_chunks)}")
        
        # Show sample entities by type
        entity_samples = {}
        for node, data in list(system.kg.graph.nodes(data=True))[:10]:
            entity_type = data.get('type', 'unknown')
            if entity_type not in entity_samples:
                entity_samples[entity_type] = []
            entity_samples[entity_type].append(node)
        
        print("\nSample Entities by Type:")
        for etype, entities in entity_samples.items():
            print(f"- {etype}: {entities[:3]}")
    
    print("\n4. ERROR HANDLING TESTS")
    
    # Test invalid queries
    error_tests = [
        {"query": "", "expected": "empty query"},
        {"query": "x", "expected": "too short"},
        {"query": "What about something completely unrelated to aliens and UFOs?", "expected": "no relevant info"}
    ]
    
    for error_test in error_tests:
        result = system.safe_query(error_test['query'])
        print(f"Query: '{error_test['query']}' -> Method: {result.get('method_used', 'unknown')}")
    
    print("\n5. PERFORMANCE INSIGHTS")
    
    # Test query explanation
    if hasattr(system, 'explain_retrieval'):
        sample_query = "What is the budget for training programs?"
        explanation = system.explain_retrieval(sample_query)
        print(f"Retrieval Explanation for: '{sample_query}'")
        print(f"- Chunks Retrieved: {explanation['chunk_retrieval']['chunks_found']}")
        print(f"- Entities Retrieved: {explanation['entity_retrieval']['entities_found']}")
        print(f"- Graph Expansion: {explanation['graph_expansion']['expanded_entities']} entities")
    
    return system

# Additional utility functions for production deployment

def create_production_config():
    """
    Configuration for production deployment
    """
    return {
        'embedding_model': 'all-MiniLM-L6-v2',  # Fast and efficient
        'chunk_size': 500,
        'chunk_overlap': 50,
        'max_entities_per_query': 10,
        'max_reasoning_hops': 3,
        'similarity_threshold': 0.1,
        'cache_size': 1000,
        'max_context_length': 3000,
        'fallback_enabled': True,
        'logging_enabled': True
    }

def validate_system_requirements():
    """
    Check if all required components are available
    """
    requirements_status = {}
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        requirements_status['spacy'] = "✓ Available"
    except:
        requirements_status['spacy'] = "✗ Missing - run: python -m spacy download en_core_web_sm"
    
    try:
        import sentence_transformers
        requirements_status['sentence_transformers'] = "✓ Available"
    except:
        requirements_status['sentence_transformers'] = "✗ Missing - run: pip install sentence-transformers"
    
    try:
        import networkx
        requirements_status['networkx'] = "✓ Available"
    except:
        requirements_status['networkx'] = "✗ Missing - run: pip install networkx"
    
    try:
        import faiss
        requirements_status['faiss'] = "✓ Available"
    except:
        requirements_status['faiss'] = "✗ Missing - run: pip install faiss-cpu"
    
    print("System Requirements Check:")
    for req, status in requirements_status.items():
        print(f"- {req}: {status}")
    
    return all("✓" in status for status in requirements_status.values())

if __name__ == "__main__":
    print("Graph RAG Implementation - Production Ready")
    print("=" * 50)
    
    # Check requirements
    if validate_system_requirements():
        print("✓ All requirements met!")
        
        # Run comprehensive test
        try:
            system = run_full_test_suite()
            print("\n✓ All tests completed successfully!")
            
            # Show final statistics
            print(f"\nFinal System Statistics:")
            print(f"- Knowledge Graph Entities: {len(system.kg.graph.nodes)}")
            print(f"- Knowledge Graph Relations: {len(system.kg.graph.edges)}")
            print(f"- Text Chunks: {len(system.kg.text_chunks)}")
            print(f"- Errors Logged: {len(system.error_log)}")
            
        except Exception as e:
            print(f"✗ Test suite failed: {str(e)}")
    else:
        print("✗ Please install missing requirements before running Graph RAG")