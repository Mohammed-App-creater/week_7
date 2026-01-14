"""
Streamlit Interface for CFPB RAG System

This application provides a user-friendly interface for querying the
CFPB RAG pipeline.

Run with:
    streamlit run ui/app.py
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path to allow importing src
# Assumes this file is in ui/app.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.rag_pipeline import RAGPipeline

# Page Configuration
st.set_page_config(
    page_title="CFPB Complaint Analysis RAG",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    
if "pipeline_loaded" not in st.session_state:
    st.session_state.pipeline_loaded = False


def load_rag_pipeline():
    """Load the RAG pipeline components."""
    try:
        if not st.session_state.pipeline_loaded:
            with st.spinner("Loading RAG Pipeline... (This may take a moment)"):
                # Initialize pipeline
                # Make sure these paths are correct relative to where styling is run
                # If running from root, vector_store/ is correct
                pipeline = RAGPipeline(
                    vector_store_path=os.path.join(PROJECT_ROOT, "vector_store"),
                    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                    llm_model="google/flan-t5-base",
                    device="cpu"  # Force CPU for compatibility
                )
                pipeline.load_vector_store()
                pipeline.load_llm()
                
                st.session_state.pipeline = pipeline
                st.session_state.pipeline_loaded = True
                st.success("Pipeline Loaded Successfully!")
    except Exception as e:
        st.error(f"Error loading pipeline: {str(e)}")
        st.error("Please ensure you have run `src/build_vector_store.py` first.")


def clear_conversation():
    """Reset conversation history."""
    st.session_state.messages = []


# ==========================================
# UI Layout
# ==========================================

# Sidebar
with st.sidebar:
    st.title("⚙️ RAG Settings")
    
    st.markdown("### Retrieval Parameters")
    k_docs = st.slider(
        "Number of Sources (k)", 
        min_value=1, 
        max_value=10, 
        value=5,
        help="Number of documents to retrieve for context."
    )
    
    st.markdown("### Generation Parameters")
    max_tokens = st.slider(
        "Max Answer Length",
        min_value=64,
        max_value=512,
        value=256,
        step=32
    )
    
    st.markdown("---")
    if st.button("Clear Conversation", type="primary"):
        clear_conversation()
        st.rerun()

    st.markdown("""
    ### About
    This tool allows you to query the CFPB Consumer Complaint Database using RAG.
    
    **Pipeline:**
    - Embeddings: all-MiniLM-L6-v2
    - Store: ChromaDB
    - LLM: FLAN-T5 Base
    """)

# Main Content
col1, col2 = st.columns([6, 1])
with col1:
    st.title("🔍 CFPB Complaint Assistant")
with col2:
    # Add vertical spacing to align button
    st.write("")
    st.write("")
    if st.button("Clear", key="clear_main_chat"):
        clear_conversation()
        st.rerun()

st.markdown("Ask questions about consumer financial complaints.")

# Load Pipeline on Startup
if not st.session_state.pipeline_loaded:
    load_rag_pipeline()

# Display Conversation History (Optional, if we want chat style)
# For this specific requirement ("Input box -> Ask -> Answer"), 
# a simple form is often clearer than a chat history, but the requirement 
# mentioned "interactive chat interface" and "Clear conversation".
# We will use a chat-like display for answers but a persistent input area.

# Container for chat history
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # If there are sources associated with this message
            if "sources" in message:
                with st.expander("📚 Retrieved Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        meta = source['metadata']
                        product = meta.get('product', 'Unknown Product')
                        issue = meta.get('issue', 'Unknown Issue')
                        company = meta.get('company', 'Unknown Company')
                        
                        st.markdown(f"**{i}. {product} - {issue}**")
                        st.caption(f"Company: {company}")
                        st.markdown(f"> {source['text']}")
                        st.divider()

# Input Area
if prompt := st.chat_input("Ask a question about financial complaints..."):
    # 1. Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Generate response
    if st.session_state.pipeline_loaded:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing complaints..."):
                try:
                    pipeline = st.session_state.pipeline
                    result = pipeline.answer_question(
                        question=prompt,
                        k=k_docs,
                        max_new_tokens=max_tokens
                    )
                    
                    answer = result['answer']
                    sources = result['sources']
                    
                    st.markdown(answer)
                    
                    with st.expander("📚 Retrieved Sources"):
                        for i, source in enumerate(sources, 1):
                            meta = source['metadata']
                            product = meta.get('product', 'Unknown Product')
                            issue = meta.get('issue', 'Unknown Issue')
                            company = meta.get('company', 'Unknown Company')
                            
                            st.markdown(f"**{i}. {product} - {issue}**")
                            st.caption(f"Company: {company}")
                            st.markdown(f"> {source['text']}")
                            st.divider()
                    
                    # Save context to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Pipeline is not loaded. Please check the logs.")
