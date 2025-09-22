# app.py - Fixed version with proper error handling

import streamlit as st
import PyPDF2
import io
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from graph_rag import GraphRAGSystem, DocumentProcessor
import tempfile
import os
from datetime import datetime

class PDFProcessor:
    """Handles PDF file processing and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(uploaded_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""

class GraphVisualizer:
    """Handles knowledge graph visualization"""
    
    @staticmethod
    def plot_networkx_graph(graph: nx.DiGraph, figsize=(12, 8)):
        """Create a matplotlib visualization of the graph"""
        plt.figure(figsize=figsize)
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # Draw nodes with different colors based on type
        node_colors = []
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'OTHER')
            if node_type == 'PERSON':
                node_colors.append('lightblue')
            elif node_type == 'ORG':
                node_colors.append('lightgreen')
            elif node_type == 'GPE':
                node_colors.append('lightcoral')
            elif node_type == 'MONEY':
                node_colors.append('gold')
            elif node_type == 'DATE':
                node_colors.append('violet')
            else:
                node_colors.append('lightgray')
        
        # Draw the graph
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=500, alpha=0.9)
        nx.draw_networkx_edges(graph, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.7)
        nx.draw_networkx_labels(graph, pos, font_size=8)
        
        # Add edge labels
        edge_labels = {(u, v): d.get('relation', '') 
                      for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, 
                                   font_size=6)
        
        plt.title("Knowledge Graph", fontsize=16)
        plt.axis('off')
        return plt

    @staticmethod
    def plot_interactive_graph(graph: nx.DiGraph):
        """Create an interactive Plotly graph visualization"""
        if len(graph.nodes()) == 0:
            # Return empty graph if no nodes
            fig = go.Figure()
            fig.update_layout(title="Empty Graph - No entities extracted")
            return fig
            
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # Extract node positions
        x_nodes = [pos[node][0] for node in graph.nodes()]
        y_nodes = [pos[node][1] for node in graph.nodes()]
        
        # Create node trace
        node_trace = go.Scatter(
            x=x_nodes, y=y_nodes,
            mode='markers+text',
            hoverinfo='text',
            text=[str(node) for node in graph.nodes()],
            textposition="middle center",
            marker=dict(
                size=20,
                color=['lightblue' if graph.nodes[node].get('type') == 'PERSON' 
                      else 'lightgreen' if graph.nodes[node].get('type') == 'ORG'
                      else 'lightcoral' for node in graph.nodes()],
                line=dict(width=2)
            )
        )
        
        # Create edge traces
        edge_traces = []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=2, color='gray'),
                hoverinfo='text',
                text=graph.edges[edge].get('relation', ''),
                mode='lines'
            )
            edge_traces.append(edge_trace)
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title="Interactive Knowledge Graph",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=600
        )
        
        return fig

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'graph_rag_system' not in st.session_state:
        st.session_state.graph_rag_system = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'knowledge_base_built' not in st.session_state:
        st.session_state.knowledge_base_built = False
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""
    if 'graph_data' not in st.session_state:
        st.session_state.graph_data = None

def safe_query(system, prompt):
    """Safely execute query with proper error handling"""
    try:
        # First try multi-hop reasoning for complex queries
        if any(keyword in prompt.lower() for keyword in 
              ['compare', 'relationship', 'connection', 'how', 'why']):
            result = system.multi_hop_reasoning(prompt)
        else:
            result = system.query(prompt)
        
        # Ensure result has the expected structure
        if 'retrieval_info' not in result:
            result['retrieval_info'] = {
                'chunks_found': 0,
                'entities_found': 0,
                'graph_nodes': 0
            }
        
        return result
        
    except Exception as e:
        # Fallback to simple response if complex query fails
        try:
            result = system.query(prompt)
            if 'retrieval_info' not in result:
                result['retrieval_info'] = {
                    'chunks_found': 0,
                    'entities_found': 0,
                    'graph_nodes': 0
                }
            return result
        except Exception as fallback_error:
            return {
                'answer': f"I apologize, but I encountered an error processing your query: {str(fallback_error)}",
                'retrieval_info': {
                    'chunks_found': 0,
                    'entities_found': 0,
                    'graph_nodes': 0
                }
            }

def display_chat_interface():
    """Display the chat interface"""
    st.header("💬 Chat with Your Documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View Sources"):
                    st.json(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = safe_query(st.session_state.graph_rag_system, prompt)
                    
                    # Display answer
                    st.markdown(result['answer'])
                    
                    # Display retrieval information
                    with st.expander("Retrieval Details"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Chunks Used", result['retrieval_info']['chunks_found'])
                        with col2:
                            st.metric("Entities Found", result['retrieval_info']['entities_found'])
                        with col3:
                            st.metric("Graph Nodes", result['retrieval_info']['graph_nodes'])
                        
                        if 'reasoning_paths' in result:
                            st.write("**Reasoning Paths:**")
                            for i, path in enumerate(result['reasoning_paths'][:3]):
                                path_text = " → ".join([step['entity'] for step in path['path']])
                                st.write(f"Path {i+1}: {path_text}")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result['answer'],
                        "sources": result['retrieval_info']
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

def main():
    st.set_page_config(
        page_title="Graph RAG Chat",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 Graph RAG Chat System")
    st.markdown("Upload PDF documents, visualize the knowledge graph, and chat with your data!")
    
    initialize_session_state()
    
    # Sidebar for PDF upload and controls
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to build the knowledge graph"
        )
        
        if uploaded_files:
            if st.button("Build Knowledge Graph", type="primary"):
                with st.spinner("Processing documents and building knowledge graph..."):
                    try:
                        # Extract text from all PDFs
                        all_texts = []
                        for uploaded_file in uploaded_files:
                            text = PDFProcessor.extract_text_from_pdf(uploaded_file)
                            if text and len(text.strip()) > 0:
                                all_texts.append(text)
                                st.success(f"✅ Processed {uploaded_file.name}")
                            else:
                                st.warning(f"⚠️ No text extracted from {uploaded_file.name}")
                        
                        if all_texts:
                            # Initialize and build Graph RAG system
                            system = GraphRAGSystem()
                            system.build_knowledge_base(all_texts)
                            
                            # Store in session state
                            st.session_state.graph_rag_system = system
                            st.session_state.knowledge_base_built = True
                            st.session_state.pdf_text = "\n\n".join(all_texts)
                            st.session_state.graph_data = system.kg.graph
                            
                            st.success(f"🎉 Knowledge graph built successfully! Found {len(system.kg.graph.nodes())} entities and {len(system.kg.graph.edges())} relationships.")
                        else:
                            st.error("No text could be extracted from the uploaded PDFs.")
                            
                    except Exception as e:
                        st.error(f"Error building knowledge graph: {str(e)}")
        
        if st.session_state.knowledge_base_built:
            st.success("✅ Knowledge base ready!")
            st.divider()
            
            # Graph visualization options
            st.header("📊 Graph Visualization")
            viz_type = st.radio(
                "Choose visualization type:",
                ["Static", "Interactive"]
            )
            
            # Graph statistics
            if st.session_state.graph_data:
                graph = st.session_state.graph_data
                st.subheader("Graph Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Nodes", len(graph.nodes()))
                    st.metric("Edges", len(graph.edges()))
                with col2:
                    try:
                        density = nx.density(graph) if len(graph.nodes()) > 1 else 0
                        st.metric("Density", f"{density:.3f}")
                        components = nx.number_weakly_connected_components(graph)
                        st.metric("Components", components)
                    except:
                        st.metric("Density", "0.000")
                        st.metric("Components", 0)
        
        st.divider()
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Main content area
    if not st.session_state.knowledge_base_built:
        # Welcome and instructions
        st.markdown("""
        ## Welcome to Graph RAG Chat!
        
        **How to use:**
        1. 📁 Upload one or more PDF documents using the sidebar
        2. 🏗️ Click "Build Knowledge Graph" to process your documents
        3. 📊 View the generated knowledge graph visualization
        4. 💬 Start chatting with your documents using the chat interface
        
        **What makes Graph RAG special:**
        - Extracts entities and relationships to build a knowledge graph
        - Uses graph structure for better reasoning and connections
        - Supports multi-hop reasoning across documents
        - Provides transparent source attribution
        """)
        
        # Placeholder for demo
        st.info("👈 Upload PDF documents to get started!")
        
    else:
        # Display knowledge graph
        st.header("📊 Knowledge Graph Visualization")
        
        if st.session_state.graph_data and len(st.session_state.graph_data.nodes()) > 0:
            graph = st.session_state.graph_data
            
            if viz_type == "Static":
                # Static matplotlib visualization
                fig = GraphVisualizer.plot_networkx_graph(graph)
                st.pyplot(fig)
            else:
                # Interactive Plotly visualization
                fig = GraphVisualizer.plot_interactive_graph(graph)
                st.plotly_chart(fig, use_container_width=True)
            
            # Entity list
            with st.expander("View All Entities"):
                entities_by_type = {}
                for node in graph.nodes():
                    node_type = graph.nodes[node].get('type', 'OTHER')
                    if node_type not in entities_by_type:
                        entities_by_type[node_type] = []
                    entities_by_type[node_type].append(node)
                
                for entity_type, entities in entities_by_type.items():
                    st.subheader(f"{entity_type} Entities ({len(entities)})")
                    st.write(", ".join(entities[:20]))  # Limit display
                    if len(entities) > 20:
                        st.write(f"... and {len(entities) - 20} more")
        else:
            st.warning("No entities were extracted from the documents. The graph may be empty.")
        
        # Chat interface
        display_chat_interface()
        
        # Document preview
        with st.expander("📄 View Processed Document Text"):
            st.text_area("Extracted Text", st.session_state.pdf_text, height=300)

if __name__ == "__main__":
    main()