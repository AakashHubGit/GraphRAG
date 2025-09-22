# demo.py - Interactive demonstration of Graph RAG capabilities

from graph_rag import GraphRAGSystem
import json
import os

def print_separator(title=""):
    """Print a nice separator"""
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print("-" * 60)

def demonstrate_system():
    """Run a comprehensive demo of Graph RAG capabilities"""
    
    print("Graph RAG Interactive Demo")
    print("This demo will show you how Graph RAG works step by step")
    
    # Check API key
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_gemini_api_key_here":
        print("\nWarning: No valid API key found in .env file")
        print("Demo will show structure but may not generate responses")
        print("Please set up your Gemini API key for full functionality")
    
    # Sample documents
    documents = [
        """
        TechStart Inc. is a software development company founded in 2020 by Maria Rodriguez. 
        The company is based in Austin, Texas and specializes in mobile app development. 
        TechStart recently received $2 million in Series A funding from Venture Capital Partners. 
        The company currently employs 25 developers and designers.
        """,
        """
        Venture Capital Partners is an investment firm led by CEO David Chen. The firm focuses 
        on early-stage technology companies and has $100 million under management. They recently 
        invested $2 million in TechStart Inc. and $5 million in DataFlow Systems. The firm is 
        headquartered in Silicon Valley, California.
        """,
        """
        DataFlow Systems develops data analytics software for enterprise clients. The company 
        was founded by Dr. Jennifer Park and is located in Seattle, Washington. DataFlow recently 
        raised $5 million from Venture Capital Partners to expand their machine learning capabilities. 
        The company serves over 100 enterprise customers.
        """
    ]
    
    # Initialize system
    print_separator("INITIALIZING SYSTEM")
    system = GraphRAGSystem()
    
    print("Building knowledge base from sample documents...")
    system.build_knowledge_base(documents)
    
    # Show graph statistics
    kg = system.kg
    print(f"Knowledge Graph Built:")
    print(f"  - Entities: {len(kg.graph.nodes)}")
    print(f"  - Relationships: {len(kg.graph.edges)}")  
    print(f"  - Text Chunks: {len(kg.text_chunks)}")
    
    # Show sample entities by type
    print("\nSample Entities by Type:")
    entity_types = {}
    for node, data in kg.graph.nodes(data=True):
        etype = data.get('type', 'unknown')
        if etype not in entity_types:
            entity_types[etype] = []
        entity_types[etype].append(node)
    
    for etype, entities in entity_types.items():
        print(f"  - {etype}: {entities[:3]}")
    
    # Show sample relationships  
    print("\nSample Relationships:")
    relationships = []
    for u, v, data in kg.graph.edges(data=True):
        relation = data.get('relation', 'related_to')
        relationships.append(f"{u} --[{relation}]--> {v}")
    
    for rel in relationships[:5]:
        print(f"  - {rel}")
    
    # Interactive query session
    print_separator("INTERACTIVE QUERY SESSION")
    
    demo_queries = [
        {
            "query": "Who founded TechStart Inc?",
            "explanation": "Direct entity-fact retrieval"
        },
        {
            "query": "Which companies received funding from Venture Capital Partners?",
            "explanation": "Multi-entity relationship query"
        },
        {
            "query": "How much total funding did Venture Capital Partners invest?",
            "explanation": "Aggregation across relationships"
        },
        {
            "query": "What is the connection between Maria Rodriguez and David Chen?",
            "explanation": "Multi-hop reasoning through entities"
        }
    ]
    
    for i, demo_query in enumerate(demo_queries, 1):
        print(f"\n--- Demo Query {i} ---")
        print(f"Query: {demo_query['query']}")
        print(f"Type: {demo_query['explanation']}")
        
        # Standard Graph RAG query
        try:
            result = system.query(demo_query['query'])
            print(f"\nAnswer: {result['answer']}")
            print(f"Context sources: {result['retrieval_info']['chunks_found']} chunks, {result['retrieval_info']['entities_found']} entities")
            
            # Show reasoning paths if available
            try:
                reasoning_result = system.multi_hop_reasoning(demo_query['query'])
                if reasoning_result['path_count'] > 0:
                    print(f"Reasoning paths found: {reasoning_result['path_count']}")
                    if reasoning_result.get('reasoning_paths'):
                        print("Sample reasoning path:")
                        first_path = reasoning_result['reasoning_paths'][0]
                        path_text = " → ".join([step['entity'] for step in first_path['path']])
                        print(f"  {path_text}")
            except:
                pass
                
        except Exception as e:
            print(f"Query failed: {e}")
        
        input("\nPress Enter to continue to next demo query...")
    
    # Interactive session
    print_separator("FREE-FORM QUERIES")
    print("Now you can ask your own questions!")
    print("Enter 'quit' to exit, 'help' for tips, 'graph' to see graph info")
    
    while True:
        try:
            user_query = input("\nYour question: ").strip()
            
            if user_query.lower() == 'quit':
                break
            elif user_query.lower() == 'help':
                print("\nQuery Tips:")
                print("- Ask about specific entities: 'What does TechStart do?'")
                print("- Ask about relationships: 'Who invested in DataFlow?'")
                print("- Ask for comparisons: 'Compare TechStart and DataFlow funding'")
                print("- Ask about connections: 'How are Maria and David connected?'")
                continue
            elif user_query.lower() == 'graph':
                show_graph_analysis(system)
                continue
            elif not user_query:
                continue
            
            # Process query
            result = system.query(user_query)
            print(f"\nAnswer: {result['answer']}")
            
            # Show retrieval info
            info = result['retrieval_info']
            print(f"Retrieved: {info['chunks_found']} text chunks, {info['entities_found']} entities, {info['graph_nodes']} graph nodes")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error processing query: {e}")

def show_graph_analysis(system):
    """Show detailed graph analysis"""
    print_separator("GRAPH ANALYSIS")
    
    kg = system.kg
    
    # Basic stats
    print(f"Graph Statistics:")
    print(f"  - Nodes: {len(kg.graph.nodes)}")
    print(f"  - Edges: {len(kg.graph.edges)}")
    
    if len(kg.graph.nodes) > 0:
        import networkx as nx
        print(f"  - Density: {nx.density(kg.graph):.3f}")
        print(f"  - Weakly connected components: {nx.number_weakly_connected_components(kg.graph)}")
        
        # Most connected entities
        degree_centrality = nx.degree_centrality(kg.graph)
        top_central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"\nMost Connected Entities:")
        for entity, centrality in top_central:
            print(f"  - {entity}: {centrality:.3f}")
        
        # Relationship types
        relation_counts = {}
        for u, v, data in kg.graph.edges(data=True):
            rel = data.get('relation', 'unknown')
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
        
        print(f"\nRelationship Types:")
        for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {rel}: {count}")

def create_custom_documents_demo():
    """Allow users to input their own documents"""
    print_separator("CUSTOM DOCUMENTS DEMO")
    print("Enter your own documents to see Graph RAG in action!")
    print("Enter each document on a separate line. Type 'DONE' when finished.")
    
    documents = []
    doc_num = 1
    
    while True:
        doc = input(f"\nDocument {doc_num}: ").strip()
        if doc.upper() == 'DONE':
            break
        elif doc:
            documents.append(doc)
            doc_num += 1
        
        if len(documents) >= 5:  # Limit for demo
            print("Maximum 5 documents for demo")
            break
    
    if not documents:
        print("No documents provided")
        return
    
    print(f"\nProcessing {len(documents)} documents...")
    
    # Build system with custom documents
    custom_system = GraphRAGSystem()
    custom_system.build_knowledge_base(documents)
    
    # Show what was built
    kg = custom_system.kg
    print(f"Built knowledge graph with {len(kg.graph.nodes)} entities and {len(kg.graph.edges)} relationships")
    
    # Interactive queries
    print("\nNow ask questions about your documents!")
    while True:
        query = input("\nQuery (or 'back' to return): ").strip()
        if query.lower() == 'back':
            break
        elif query:
            try:
                result = custom_system.query(query)
                print(f"Answer: {result['answer']}")
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main demo function"""
    print("=" * 70)
    print("          Welcome to Graph RAG Interactive Demo")
    print("=" * 70)
    
    while True:
        print("\nDemo Options:")
        print("1. Run full demo with sample documents")
        print("2. Try with your own documents")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            demonstrate_system()
        elif choice == '2':
            create_custom_documents_demo()
        elif choice == '3':
            print("Thanks for trying Graph RAG!")
            break
        else:
            print("Invalid option. Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main()