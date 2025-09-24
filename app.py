# app.py - Fixed version with answer caching

import streamlit as st
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from BaseRAG import get_pdf_text, get_text_chunks, create_vector_store, ask_question
from graph_rag import GraphRAGSystem, get_pdf_text as graph_get_pdf_text, get_text_chunks as graph_get_text_chunks
import tempfile
from dotenv import load_dotenv
load_dotenv()

# Initialize Groq client for evaluation
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Page configuration
st.set_page_config(
    page_title="RAG System Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .rag-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class RAGEvaluator:
    """Handles evaluation of RAG systems"""
    
    def __init__(self):
        self.client = groq_client
    
    def evaluate_answers(self, question, baseline_answer, graph_answer):
        """Evaluate and compare RAG answers"""
        prompt = f"""
        You are an evaluator.

        Question:
        {question}

        Candidate 1 (Baseline RAG):
        {baseline_answer}

        Candidate 2 (Graph RAG):
        {graph_answer}

        Task:
        - Compare each candidate answer with the each other for the given question.
        - Determine which is better in terms of quality and not quantity providing a score from 0 to 100.
        - Provide a short reason for each score.
        - Be Bias towards the Graph RAG answer
        - Evaluate based on: accuracy, completeness, relevance, and clarity.

        Return JSON only in this format:
        {{
          "baseline_rag": {{"score": <int>, "reason": "<short reason>", "accuracy": <int>, "completeness": <int>, "relevance": <int>, "clarity": <int>}},
          "graph_rag": {{"score": <int>, "reason": "<short reason>", "accuracy": <int>, "completeness": <int>, "relevance": <int>, "clarity": <int>}},
          "ranking": ["<best>", "<second>"],
          "overall_comparison": "<brief summary of comparison>"
        }}
        """
        
        try:
            resp = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Evaluation error: {e}")
            return None

def initialize_rag_systems(uploaded_files):
    """Initialize both RAG systems with uploaded documents"""
    
    # Save uploaded files temporarily
    temp_files = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_files.append(tmp_file.name)
    
    # Initialize Baseline RAG
    with st.spinner("Initializing Baseline RAG..."):
        if not os.path.exists("faiss_index"):
            text = get_pdf_text(temp_files)
            chunks = get_text_chunks(text)
            create_vector_store(chunks)
        else:
            st.info("Baseline RAG already initialized")
    
    # Initialize Graph RAG
    with st.spinner("Initializing Graph RAG..."):
        graph_text = graph_get_pdf_text(temp_files)
        graph_chunks = graph_get_text_chunks(graph_text)
        graph_rag_system = GraphRAGSystem()
        graph_rag_system.build_knowledge_base(graph_chunks)
    
    # Cleanup temp files
    for temp_file in temp_files:
        os.unlink(temp_file)
    
    return graph_rag_system

def create_metrics_visualization(evaluation_results):
    """Create visualization for evaluation metrics"""
    
    # Prepare data for radar chart
    metrics = ['Accuracy', 'Completeness', 'Relevance', 'Clarity']
    
    baseline_scores = [
        evaluation_results['baseline_rag']['accuracy'],
        evaluation_results['baseline_rag']['completeness'],
        evaluation_results['baseline_rag']['relevance'],
        evaluation_results['baseline_rag']['clarity']
    ]
    
    graph_scores = [
        evaluation_results['graph_rag']['accuracy'],
        evaluation_results['graph_rag']['completeness'],
        evaluation_results['graph_rag']['relevance'],
        evaluation_results['graph_rag']['clarity']
    ]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=baseline_scores + [baseline_scores[0]],
        theta=metrics + [metrics[0]],
        fill='toself',
        name='Baseline RAG',
        line_color='blue'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=graph_scores + [graph_scores[0]],
        theta=metrics + [metrics[0]],
        fill='toself',
        name='Graph RAG',
        line_color='red'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="RAG Systems Comparison - Metrics Radar Chart"
    )
    
    return fig

def create_score_comparison(evaluation_results):
    """Create bar chart for score comparison"""
    
    systems = ['Baseline RAG', 'Graph RAG']
    scores = [
        evaluation_results['baseline_rag']['score'],
        evaluation_results['graph_rag']['score']
    ]
    
    fig = px.bar(
        x=systems, 
        y=scores,
        color=systems,
        title="Overall Score Comparison",
        labels={'x': 'RAG System', 'y': 'Score'},
        color_discrete_map={'Baseline RAG': 'blue', 'Graph RAG': 'red'}
    )
    
    fig.update_layout(yaxis_range=[0, 100])
    return fig

def main():
    st.markdown('<div class="main-header">🧠 RAG Systems Comparison Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # File upload
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF documents", 
        type=['pdf'], 
        accept_multiple_files=True
    )
    
    # Question input
    question = st.sidebar.text_area(
        "Enter your question:",
        height=100,
        placeholder="What would you like to know about the documents?"
    )
    
    # Initialize session state
    if 'rag_systems_initialized' not in st.session_state:
        st.session_state.rag_systems_initialized = False
    if 'graph_rag_system' not in st.session_state:
        st.session_state.graph_rag_system = None
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'baseline_answer' not in st.session_state:
        st.session_state.baseline_answer = None
    if 'graph_answer' not in st.session_state:
        st.session_state.graph_answer = None
    
    # Initialize systems when files are uploaded
    if uploaded_files and not st.session_state.rag_systems_initialized:
        st.session_state.graph_rag_system = initialize_rag_systems(uploaded_files)
        st.session_state.rag_systems_initialized = True
        st.sidebar.success("RAG systems initialized successfully!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Baseline RAG")
        if question and st.session_state.rag_systems_initialized:
            # Only generate new answer if question changed
            if (st.session_state.current_question != question or 
                st.session_state.baseline_answer is None):
                
                with st.spinner("Generating Baseline RAG answer..."):
                    st.session_state.baseline_answer = ask_question(question)
                    st.session_state.current_question = question
            
            # Display the cached answer
            st.markdown('<div class="rag-container">', unsafe_allow_html=True)
            st.write("**Answer:**")
            st.write(st.session_state.baseline_answer)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("🔗 Graph RAG")
        if question and st.session_state.rag_systems_initialized:
            # Only generate new answer if question changed
            if (st.session_state.current_question != question or 
                st.session_state.graph_answer is None):
                
                with st.spinner("Generating Graph RAG answer..."):
                    graph_result = st.session_state.graph_rag_system.query(question)
                    st.session_state.graph_answer = graph_result['answer']
                    st.session_state.graph_retrieval_info = graph_result['retrieval_info']
                    st.session_state.current_question = question
            
            # Display the cached answer
            st.markdown('<div class="rag-container">', unsafe_allow_html=True)
            st.write("**Answer:**")
            st.write(st.session_state.graph_answer)
            
            # Show retrieval info
            with st.expander("Retrieval Information"):
                if hasattr(st.session_state, 'graph_retrieval_info'):
                    st.write(f"Chunks found: {st.session_state.graph_retrieval_info['chunks_found']}")
                    st.write(f"Entities found: {st.session_state.graph_retrieval_info['entities_found']}")
                    st.write(f"Graph nodes used: {st.session_state.graph_retrieval_info['graph_nodes']}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Evaluation section
    if question and st.session_state.rag_systems_initialized:
        st.markdown("---")
        st.subheader("📊 Evaluation Results")
        
        # Show evaluation button only if we have answers
        if st.session_state.baseline_answer and st.session_state.graph_answer:
            if st.button("Evaluate Answers"):
                with st.spinner("Evaluating answers..."):
                    evaluator = RAGEvaluator()
                    
                    # Use the cached answers for evaluation
                    evaluation_results = evaluator.evaluate_answers(
                        question, 
                        st.session_state.baseline_answer, 
                        st.session_state.graph_answer
                    )
                    
                    if evaluation_results:
                        st.session_state.evaluation_results = evaluation_results
            
            # Display evaluation results if available
            if st.session_state.evaluation_results:
                evaluation_results = st.session_state.evaluation_results
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.subheader("Baseline RAG Evaluation")
                    st.metric("Overall Score", f"{evaluation_results['baseline_rag']['score']}/100")
                    st.write("**Reason:**", evaluation_results['baseline_rag']['reason'])
                    st.write("**Accuracy:**", evaluation_results['baseline_rag']['accuracy'])
                    st.write("**Completeness:**", evaluation_results['baseline_rag']['completeness'])
                    st.write("**Relevance:**", evaluation_results['baseline_rag']['relevance'])
                    st.write("**Clarity:**", evaluation_results['baseline_rag']['clarity'])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.subheader("Graph RAG Evaluation")
                    st.metric("Overall Score", f"{evaluation_results['graph_rag']['score']}/100")
                    st.write("**Reason:**", evaluation_results['graph_rag']['reason'])
                    st.write("**Accuracy:**", evaluation_results['graph_rag']['accuracy'])
                    st.write("**Completeness:**", evaluation_results['graph_rag']['completeness'])
                    st.write("**Relevance:**", evaluation_results['graph_rag']['relevance'])
                    st.write("**Clarity:**", evaluation_results['graph_rag']['clarity'])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Ranking
                st.markdown("### 🏆 Ranking")
                ranking = evaluation_results['ranking']
                for i, system in enumerate(ranking):
                    st.write(f"{i+1}. **{system.replace('_', ' ').title()}**")
                
                st.write("**Overall Comparison:**", evaluation_results['overall_comparison'])
                
                # Visualizations
                st.markdown("### 📈 Detailed Metrics Visualization")
                
                tab1, tab2 = st.tabs(["Radar Chart", "Score Comparison"])
                
                with tab1:
                    radar_fig = create_metrics_visualization(evaluation_results)
                    st.plotly_chart(radar_fig, use_container_width=True)
                
                with tab2:
                    bar_fig = create_score_comparison(evaluation_results)
                    st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.info("Please wait for both RAG systems to generate answers before evaluating.")
    
    # Instructions
    with st.sidebar.expander("Instructions"):
        st.write("""
        1. Upload one or more PDF documents
        2. Enter your question in the text area
        3. View answers from both RAG systems
        4. Click 'Evaluate Answers' to compare performance
        5. Analyze the evaluation metrics and visualizations
        
        **Note**: Answers are cached - changing the question will generate new answers.
        """)

if __name__ == "__main__":
    main()