"""
Quick test of RAG pipeline core functionality.
Tests vector store loading and retrieval without LLM generation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline import RAGPipeline

def test_vector_store_and_retrieval():
    """Test vector store loading and retrieval."""
    print("=" * 80)
    print("QUICK RAG PIPELINE TEST")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        vector_store_path="vector_store/",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Test 1: Load vector store
    print("\n[Test 1] Loading vector store...")
    pipeline.load_vector_store()
    print("✓ Vector store loaded successfully")
    
    # Test 2: Retrieve documents
    print("\n[Test 2] Testing retrieval...")
    test_questions = [
        "What are common credit card billing complaints?",
        "What issues do customers face with mortgage loans?",
        "What complaints mention fraud?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        docs = pipeline.retrieve_documents(question, k=3)
        print(f"✓ Retrieved {len(docs)} documents")
        
        # Show first result
        if docs:
            print(f"\n  Top Result:")
            print(f"    Product: {docs[0]['metadata'].get('product', 'Unknown')}")
            print(f"    Issue: {docs[0]['metadata'].get('issue', 'Unknown')}")
            print(f"    Text: {docs[0]['text'][:150]}...")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
    print("\nNote: Full evaluation with LLM generation requires downloading")
    print("the FLAN-T5 model (~990MB). Run 'python src/evaluate_rag.py' to")
    print("execute the complete evaluation and generate the report.")

if __name__ == "__main__":
    test_vector_store_and_retrieval()
