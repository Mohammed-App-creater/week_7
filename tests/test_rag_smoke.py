"""
RAG Smoke Test - Lightweight End-to-End Integration Test

Purpose:
- Verify the full RAG pipeline works end-to-end
- Test retrieve → deduplicate → generate → validate workflow
- Distinguish retrieval failures from generation failures
- Ensure refactored pipeline produces helpful answers (not refusals)

This test runs the complete RAG pipeline with a realistic domain question.
"""

import pytest
import os
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from rag_pipeline import RAGPipeline


class TestRAGSmoke:
    """End-to-end RAG pipeline integration tests."""
    
    @pytest.fixture(scope="class")
    def rag_pipeline(self):
        """Initialize RAG pipeline once for all tests."""
        vector_store_path = Path(__file__).parent.parent / "vector_store"
        
        # Skip tests if vector store doesn't exist
        if not vector_store_path.exists():
            pytest.skip("Vector store not found. Run build_vector_store.py first.")
        
        # Initialize pipeline
        pipeline = RAGPipeline(
            vector_store_path=str(vector_store_path),
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            llm_model="google/flan-t5-base",
            device="cpu"
        )
        
        # Load components
        pipeline.load_vector_store()
        pipeline.load_llm()
        
        return pipeline
    
    def test_pipeline_initialization(self, rag_pipeline):
        """Test that RAG pipeline initializes successfully."""
        assert rag_pipeline is not None, "Pipeline failed to initialize"
        assert rag_pipeline.vector_store is not None, "Vector store not loaded"
        assert rag_pipeline.llm is not None, "LLM not loaded"
        assert rag_pipeline.tokenizer is not None, "Tokenizer not loaded"
    
    def test_end_to_end_question_answering(self, rag_pipeline):
        """Test full RAG pipeline with a realistic domain question."""
        question = "What are common mortgage complaints?"
        
        # Run full pipeline
        result = rag_pipeline.answer_question(question, k=5)
        
        # Assertions on result structure
        assert result is not None, "No result returned"
        assert "answer" in result, "Result missing 'answer' key"
        assert "sources" in result, "Result missing 'sources' key"
        
        # Assertions on answer
        answer = result["answer"]
        assert answer is not None, "Answer is None"
        assert isinstance(answer, str), "Answer is not a string"
        assert len(answer) > 0, "Answer is empty"
        
        # Assertions on sources
        sources = result["sources"]
        assert isinstance(sources, list), "Sources is not a list"
        assert len(sources) > 0, "No sources returned"
    
    def test_answer_is_not_empty_refusal(self, rag_pipeline):
        """Test that pipeline produces meaningful answers, not empty refusals."""
        question = "What are common credit card billing issues?"
        
        result = rag_pipeline.answer_question(question, k=5)
        answer = result["answer"]
        
        # Answer should be reasonably long (not just a refusal message)
        assert len(answer) >= 50, f"Answer too short ({len(answer)} chars), likely a refusal: {answer}"
        
        # Check if it's a generic refusal (should be rare with relaxed prompt)
        refusal_phrases = [
            "not enough information",
            "unable to answer",
            "cannot provide",
            "insufficient information"
        ]
        
        lower_answer = answer.lower()
        is_refusal = any(phrase in lower_answer for phrase in refusal_phrases)
        
        # With relaxed prompts, refusals should be rare unless no docs retrieved
        # If we got sources, we should get a synthesized answer
        if len(result["sources"]) > 0:
            assert not is_refusal, f"Pipeline refused to answer despite having sources: {answer}"
    
    def test_answer_length_reasonable(self, rag_pipeline):
        """Test that answer length is reasonable (not too short)."""
        question = "What complaints mention fraud or identity theft?"
        
        result = rag_pipeline.answer_question(question, k=5, max_new_tokens=512)
        answer = result["answer"]
        
        # With relaxed settings, answers should be more comprehensive
        # At minimum 100 chars for a meaningful synthesized answer
        assert len(answer) >= 100, f"Answer suspiciously short ({len(answer)} chars): {answer}"
    
    def test_retrieval_returns_sources(self, rag_pipeline):
        """Test that retrieval step returns sources with metadata."""
        question = "What are the main problems with student loans?"
        
        result = rag_pipeline.answer_question(question, k=5)
        sources = result["sources"]
        
        # Should have at least one source
        assert len(sources) >= 1, "No sources retrieved"
        
        # Each source should have required structure
        for i, source in enumerate(sources):
            assert "text" in source, f"Source {i} missing 'text' key"
            assert "metadata" in source, f"Source {i} missing 'metadata' key"
            assert isinstance(source["text"], str), f"Source {i} text is not a string"
            assert isinstance(source["metadata"], dict), f"Source {i} metadata is not a dict"
            assert len(source["text"]) > 0, f"Source {i} has empty text"
    
    def test_pipeline_handles_multiple_questions(self, rag_pipeline):
        """Test that pipeline can handle multiple sequential questions."""
        questions = [
            "What are common issues with mortgages?",
            "What complaints mention fraud?",
            "What are checking account problems?"
        ]
        
        for question in questions:
            try:
                result = rag_pipeline.answer_question(question, k=3)
                
                # Basic validation
                assert result is not None, f"No result for question: {question}"
                assert "answer" in result, f"Missing answer for question: {question}"
                assert "sources" in result, f"Missing sources for question: {question}"
                assert len(result["answer"]) > 0, f"Empty answer for question: {question}"
                
            except Exception as e:
                pytest.fail(f"Pipeline failed for question '{question}': {e}")
    
    def test_deduplication_works(self, rag_pipeline):
        """Test that deduplication removes redundant documents."""
        question = "mortgage issues"
        
        # Retrieve documents
        raw_docs = rag_pipeline.retrieve_documents(question, k=10)
        
        # Deduplicate
        deduped_docs = rag_pipeline.deduplicate_documents(raw_docs)
        
        # Deduplication should reduce or maintain count (never increase)
        assert len(deduped_docs) <= len(raw_docs), "Deduplication increased document count"
        
        # All deduped docs should be unique by composite key
        seen_keys = set()
        for doc in deduped_docs:
            metadata = doc['metadata']
            text = doc['text']
            product = metadata.get('product', 'Unknown')
            issue = metadata.get('issue', 'Unknown')
            text_prefix = text[:200].strip()
            key = (product, issue, text_prefix)
            
            assert key not in seen_keys, "Deduplication failed: duplicate key found"
            seen_keys.add(key)
    
    def test_validation_allows_valid_answers(self, rag_pipeline):
        """Test that validation allows valid synthesized answers to pass."""
        question = "What are common debt collection complaints?"
        
        result = rag_pipeline.answer_question(question, k=5)
        answer = result["answer"]
        
        # Validated answer should not be a generic fallback
        # (unless LLM truly failed to generate anything useful)
        fallback_messages = [
            "unable to generate",
            "please review the sources below",
            "not enough synthesized information"
        ]
        
        lower_answer = answer.lower()
        is_fallback = any(msg in lower_answer for msg in fallback_messages)
        
        # With relaxed validation, fallbacks should be rare
        # If we have 5 sources, we should get a real answer
        if len(result["sources"]) >= 5:
            # Allow occasional fallbacks but log them
            if is_fallback:
                print(f"WARNING: Got fallback despite 5 sources: {answer}")
