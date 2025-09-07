# app.py
import streamlit as st
from robust import RobustGraphRAG, create_production_config, validate_system_requirements
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components

# ==============================
# Streamlit Frontend for RobustGraphRAG
# ==============================

st.set_page_config(page_title="GraphRAG Dashboard", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Configuration")
config = create_production_config()
fallback = st.sidebar.checkbox("Enable Fallback (Simple RAG)", value=True)

# Initialize system
if "system" not in st.session_state:
    st.session_state.system = RobustGraphRAG(fallback_to_simple_rag=fallback)

system = st.session_state.system

# App header
st.title("🧠 Graph RAG Frontend")
st.markdown("Interact with your **Robust Graph RAG** system. Upload documents, build a knowledge graph, query it, and inspect reasoning.")

# Step 1: Document Upload
st.header("📄 1. Upload Documents")
uploaded_docs = st.text_area("Paste your documents here (separate by ---)", height=200)
if st.button("Build Knowledge Base"):
    docs = [doc.strip() for doc in uploaded_docs.split("---") if doc.strip()]
    result = system.safe_build_knowledge_base(docs)

    if result['status'] == "success":
        st.success(f"✅ Knowledge Base built successfully with {result['documents_processed']} documents")
        st.json(result)
    else:
        st.error("❌ Knowledge Base construction failed")
        st.json(result)

# Step 2: Graph Overview
if system.kg.graph.nodes:
    st.header("🌐 2. Knowledge Graph Overview")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write(f"Entities: {len(system.kg.graph.nodes)} | Relations: {len(system.kg.graph.edges)}")

        # Create interactive Pyvis graph
        net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
        net.from_nx(system.kg.graph)
        net.force_atlas_2based()  # Better layout for large graphs

        # Save and display inside Streamlit
        net.save_graph("graph.html")
        html_file = open("graph.html", "r", encoding="utf-8").read()
        components.html(html_file, height=550)

    with col2:
        st.subheader("Sample Entities")
        sample_nodes = list(system.kg.graph.nodes(data=True))[:10]
        for node, data in sample_nodes:
            st.markdown(f"**{node}** → type: `{data.get('type','unknown')}`")

# Step 3: Query Interface
st.header("❓ 3. Ask Questions")
query = st.text_input("Enter your query")
if st.button("Run Query"):
    result = system.safe_query(query)
    st.subheader("🔎 Answer")
    st.write(result['answer'])

    st.subheader("📖 Context Used")
    st.text(result.get("context_used", ""))

    st.subheader("ℹ️ Retrieval Info")
    st.json(result.get("retrieval_info", {}))

# Step 4: Error Log
st.header("⚠️ 4. Error Log")
if system.error_log:
    st.write("Errors encountered during processing:")
    for err in system.error_log:
        st.error(f"{err['timestamp']} → {err['error']}")
else:
    st.success("No errors logged ✅")

# Step 5: System Stats
st.header("📊 5. System Statistics")
if system.kg.graph.nodes:
    st.metric("Entities", len(system.kg.graph.nodes))
    st.metric("Relations", len(system.kg.graph.edges))
    st.metric("Text Chunks", len(system.kg.text_chunks))
    st.metric("Errors Logged", len(system.error_log))
