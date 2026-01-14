"""
RAG Pipeline for CFPB Financial Complaint Analysis
Task 3: Complete RAG core logic with retrieval and generation.

REFACTORED VERSION - Produces synthesized, evidence-backed answers instead of verbatim complaint text.

Author: Data & AI Engineer
Date: 2026-01-13
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for financial complaint analysis.
    
    Components:
    - Vector Store Loader: Load persisted ChromaDB
    - Retriever: Semantic search over complaint narratives
    - Deduplicator: Remove duplicate/similar complaints (NEW)
    - Context Formatter: Format context with metadata and truncation (NEW)
    - Prompt Template: Strong instruction-style prompts for synthesis (IMPROVED)
    - Generator: FLAN-T5 model with deterministic settings (IMPROVED)
    - Quality Validator: Detect and fix bad answers (NEW)
    - Orchestrator: End-to-end question answering
    """
    
    def __init__(
        self,
        vector_store_path: str = "vector_store/",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "google/flan-t5-base",
        device: str = "cpu"
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store_path: Path to persisted ChromaDB
            embedding_model: HuggingFace embedding model (must match indexing)
            llm_model: HuggingFace LLM for generation
            device: Device for model inference ('cpu' or 'cuda')
        """
        self.vector_store_path = Path(vector_store_path)
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.device = device
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.tokenizer = None
        self.llm = None
        
        logger.info(f"Initializing RAG Pipeline (REFACTORED)")
        logger.info(f"  Vector Store: {self.vector_store_path}")
        logger.info(f"  Embedding Model: {self.embedding_model_name}")
        logger.info(f"  LLM Model: {self.llm_model_name}")
        logger.info(f"  Device: {self.device}")
    
    def load_vector_store(self):
        """
        Load persisted ChromaDB vector store.
        
        Uses the SAME embedding model as indexing to ensure compatibility.
        """
        logger.info("Loading vector store...")
        
        if not self.vector_store_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.vector_store_path}. "
                "Please run build_vector_store.py first."
            )
        
        # Initialize embeddings (must match indexing configuration)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}  # Match indexing
        )
        
        logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        
        # Load ChromaDB
        self.vector_store = Chroma(
            persist_directory=str(self.vector_store_path),
            embedding_function=self.embeddings
        )
        
        logger.info("✓ Vector store loaded successfully")
        
        # Verify collection
        collection_count = self.vector_store._collection.count()
        logger.info(f"  Collection size: {collection_count:,} embeddings")
    
    def load_llm(self):
        """
        Load HuggingFace LLM for answer generation.
        
        Uses FLAN-T5-base (780M params) for fast CPU inference.
        Configured for DETERMINISTIC generation (do_sample=False, temperature=0.0).
        """
        logger.info(f"Loading LLM: {self.llm_model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        
        # Load model
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True
        )
        
        # Move to device
        self.llm.to(self.device)
        
        logger.info(f"✓ LLM loaded successfully on {self.device}")
    
    def retrieve_documents(
        self,
        question: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant documents for a question.
        
        Args:
            question: User question
            k: Number of documents to retrieve
        
        Returns:
            List of dicts with 'text' and 'metadata' keys
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not loaded. Call load_vector_store() first.")
        
        logger.info(f"Retrieving documents for: '{question}'")
        logger.info(f"  k={k}")
        
        # Perform similarity search
        results = self.vector_store.similarity_search(question, k=k)
        
        # Format results
        documents = []
        for doc in results:
            documents.append({
                'text': doc.page_content,
                'metadata': doc.metadata
            })
        
        logger.info(f"✓ Retrieved {len(documents)} documents")
        
        return documents
    
    def filter_by_product_category(
        self, 
        question: str, 
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter retrieved documents based on product category implied by the question.
        
        IMPROVEMENT: Prevents irrelevant product complaints from polluting context.
        e.g., prevents "Credit reporting" results when user asks about "Credit cards".
        
        Args:
            question: User question
            documents: List of retrieved documents
            
        Returns:
            Filtered list of documents (or original list if fallback triggered)
        """
        # Map keywords to product categories (based on CFPB dataset products)
        product_map = {
            "credit card": ["Credit card", "Prepaid card"],
            "mortgage": ["Mortgage"],
            "loan": ["Mortgage", "Student loan", "Vehicle loan or lease", "Payday loan"],
            "bank account": ["Bank account or service", "Checking or savings account"],
            "saving": ["Bank account or service", "Checking or savings account"],
            "checking": ["Bank account or service", "Checking or savings account"],
            "debt": ["Debt collection"],
            "collection": ["Debt collection"],
            "credit report": ["Credit reporting", "Credit reporting, credit repair services, or other personal consumer reports"],
            "score": ["Credit reporting", "Credit reporting, credit repair services, or other personal consumer reports"]
        }
        
        # Identify implied products
        implied_products = set()
        question_lower = question.lower()
        
        for keyword, categories in product_map.items():
            if keyword in question_lower:
                implied_products.update(categories)
        
        # If no product inferred, return original documents
        if not implied_products:
            return documents
            
        # Filter documents
        filtered_docs = []
        for doc in documents:
            doc_product = doc['metadata'].get('product', 'Unknown') + " " + doc['metadata'].get('product_category', '')
            # Check if doc product matches any implied category
            if any(cat.lower() in doc_product.lower() for cat in implied_products):
                filtered_docs.append(doc)
        
        # Fallback: If filtering is too aggressive (removes >80% of docs), return original
        if len(filtered_docs) < len(documents) * 0.2 and len(documents) > 0:
            logger.warning(f"⚠ Filtering removed too many documents ({len(documents)} -> {len(filtered_docs)}). Falling back to unfiltered.")
            return documents
            
        if len(filtered_docs) < len(documents):
            logger.info(f"✓ Product Filtering: Kept {len(filtered_docs)}/{len(documents)} docs matching {implied_products}")
            
        return filtered_docs

    def clean_complaint_text(self, text: str) -> str:
        """
        Clean complaint text by removing boilerplate, signatures, and legal noise.
        
        IMPROVEMENT: Reduces noise in context specifically for CFPB complaints.
        """
        import re
        
        # 1. Remove common signatures and closings
        # Match "Sincerely", "Regards", etc., followed by name/footer to the end of the text
        # Using \b to ensure word boundary, allowing for inline signatures (e.g. "...end. Sincerely, Bob")
        text = re.sub(r'(?i)(\b(sincerely|regards|thank you|respectfully|yours truly)[\s,]+[a-z ]+.*)', '', text, flags=re.DOTALL)
        
        # Also clean specific "Sent from my..." footers if they appear alone
        text = re.sub(r'(?i)\n\s*sent from my.*$', '', text)
        
        # 2. Remove URLS
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        
        # 3. Remove legal citations (Section X, Paragraph Y)
        text = re.sub(r'(?i)section \d+[\(\)\w]*|paragraph \d+', '', text)
        
        # 4. Remove common company form response markers
        text = re.sub(r'(?i)(we apologize|our records show|please contact us|customer service)', '', text)
        
        # 5. Collapse multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()

        # 6. Remove common legal boilerplate
        text = re.sub(
            r'(?i)(reasonable policies and procedures|required under|pursuant to|under \d+ u\.s\.c)',
            '',
            text
        )
        
        return text
    
    def deduplicate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate retrieved documents based on product, issue, and text prefix.
        
        IMPROVEMENT: Removes duplicate or near-duplicate complaints to avoid wasting
        context window and confusing the LLM with redundant information.
        
        Deduplication strategy:
        - Uses composite key: (product, issue, text[:200])
        - Removes exact duplicates and near-duplicates (same issue + similar text)
        - Preserves order (keeps first occurrence)
        
        Args:
            documents: List of retrieved documents with text and metadata
        
        Returns:
            Deduplicated list of documents (preserves order)
        """
        seen_keys = set()
        deduplicated = []
        
        for doc in documents:
            metadata = doc['metadata']
            text = doc['text']
            
            # Create composite deduplication key
            # Use product, issue, and first 200 chars of text for similarity detection
            product = metadata.get('product', 'Unknown')
            issue = metadata.get('issue', 'Unknown')
            text_prefix = text[:200].strip()  # First 200 chars for near-duplicate detection
            
            dedup_key = (product, issue, text_prefix)
            
            # Only add if not seen before
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                deduplicated.append(doc)
        
        # Log deduplication results
        removed_count = len(documents) - len(deduplicated)
        if removed_count > 0:
            logger.info(f"✓ Deduplication: {len(documents)} → {len(deduplicated)} documents ({removed_count} duplicates removed)")
        else:
            logger.info(f"✓ Deduplication: No duplicates found ({len(documents)} documents)")
        
        return deduplicated
    
    def format_context(
        self,
        documents: List[Dict[str, Any]],
        max_excerpt_length: int = 200
    ) -> str:
        """
        Format retrieved documents as numbered, truncated excerpts with metadata.
        
        IMPROVEMENT: Structured context formatting to:
        - Provide clear source numbering for evidence attribution
        - Include key metadata (product, issue, company) for each excerpt
        - Truncate text to avoid token overflow (context window limit)
        
        Args:
            documents: List of retrieved documents with text and metadata
            max_excerpt_length: Maximum characters per excerpt (default: 200)
        
        Returns:
            Formatted context string with numbered excerpts
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc['metadata']
            text = doc['text']
            
            # Extract key metadata for context
            product = metadata.get('product', 'Unknown')
            issue = metadata.get('issue', 'Unknown')
            company = metadata.get('company', 'Unknown')
            
            # Clean text (NEW)
            clean_text = self.clean_complaint_text(text)
            
            # Truncate text to avoid token overflow
            # This prevents context window exhaustion and forces LLM to synthesize
            if len(clean_text) > max_excerpt_length:
                truncated_text = clean_text[:max_excerpt_length].strip() + "..."
            else:
                truncated_text = clean_text.strip()
            
            # Format as numbered excerpt with metadata header
            context_parts.append(
                f"[{i}] Product: {product} | Issue: {issue}\n"
                f"    {truncated_text}"
            )
        
        return "\n\n".join(context_parts)
    
    def create_prompt(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Create a robust synthesis-focused prompt optimized for FLAN-T5.
        
        IMPROVED VERSION:
        - Explicitly enforces synthesis (patterns vs examples)
        - Forbids listing or quoting individual complaints
        - Hard-locks output format to numbered bullet points
        """
        
        # Strict System Instructions
        system_instructions = """
You are an expert financial analyst.

TASK TYPE: SYNTHESIS (NOT SELECTION).
You must analyze multiple complaint excerpts together and identify recurring GENERALIZED PATTERNS.
Do NOT choose, rank, or summarize individual complaints.

STRICT OUTPUT RULES:
1. Output MUST be 3–7 numbered bullet points.
2. Each bullet point must describe a GENERALIZED billing issue or pattern observed across complaints.
3. Each bullet point must be supported by information from AT LEAST TWO DIFFERENT excerpts.
4. DO NOT mention specific complaint numbers, dates, companies, or individuals.
5. DO NOT quote, paraphrase, or restate any single complaint.
6. DO NOT say phrases like "the complaints mention" or "one customer said".
7. Ignore company responses, legal text, signatures, or boilerplate. Focus only on consumer problems.

"""

        context = self.format_context(
            retrieved_docs,
            max_excerpt_length=400 # Increased length for better context with cleaning
        )

        prompt = f"""{system_instructions}

CONTEXT (RAW EVIDENCE — NOT ANSWERS):
The text below is provided only as evidence. It may be noisy, repetitive, or incomplete.
Do NOT select or repeat any single excerpt as an answer.
You must reason ACROSS excerpts to identify shared patterns.
{context}

USER QUESTION:
{question}



Generate the synthesized analysis now:
"""
        return prompt

    
    def generate_answer(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.0
    ) -> str:
        """
        Generate answer using LLM based on retrieved documents.
        
        REFACTORED: Relaxed generation settings for more comprehensive answers:
        - do_sample=False: Greedy decoding (deterministic, no randomness)
        - temperature=0.0: No sampling temperature
        - max_new_tokens: Increased to 512 (default), supports up to 2000 for complex questions
        
        Args:
            question: User question
            retrieved_docs: Retrieved documents with text and metadata
            max_new_tokens: Maximum tokens to generate (default: 512, max: 2000)
            temperature: Sampling temperature (default: 0.0 for deterministic output)
        
        Returns:
            Generated answer string (only the generated text, no prompt echo)
        """
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("LLM not loaded. Call load_llm() first.")
        
        logger.info("Generating answer...")
        
        # Create prompt with improved template
        prompt = self.create_prompt(question, retrieved_docs)
        
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,  # Context window limit
            truncation=True
        ).to(self.device)
        
        # Deterministic generation settings
        # - do_sample=False: Greedy decoding (most probable tokens)
        # - temperature=0.0: No randomness (though ignored when do_sample=False)
        # - num_beams=1: Single beam (greedy search, fastest)
        # - max_new_tokens: Allows longer, more comprehensive answers (512-2000)
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # Flexible length (default 512, up to 2000)
                do_sample=False,                 # CRITICAL: Deterministic greedy decoding
                num_beams=1,                     # Greedy search (single beam)
            )
        
        # Decode only the generated tokens (skip_special_tokens removes <pad>, </s>, etc.)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"✓ Generated answer ({len(answer)} chars)")
        logger.info(f" The answer is: {answer}")
        return answer.strip()
    
    def validate_answer(
        self,
        answer: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Validate generated answer and apply quality guardrails.
        
        REFACTORED: Relaxed post-generation validation to reduce false rejections:
        
        Quality checks:
        1. Minimum length (>20 characters) - prevents empty/broken outputs
        2. First-person language detection - WARNING ONLY (not hard failure)
        3. Verbatim complaint text check - prevents copy-paste
        
        Args:
            answer: Generated answer to validate
            retrieved_docs: Retrieved documents for verbatim check
        
        Returns:
            Validated answer or fallback message if critical quality issues detected
        """
        # Strip whitespace for validation
        clean_answer = answer.strip()
        
        # QUALITY CHECK 1: Minimum length (relaxed from 30 to 20 chars)
        # Allows brief but valid answers like "No relevant complaints found."
        if len(clean_answer) < 20:
            logger.warning("⚠ Answer too short (< 20 chars), likely incomplete")
            return "Unable to generate a meaningful summary. Please review the sources below."
        
        # QUALITY CHECK 2: First-person language (WARNING ONLY - not hard failure)
        # The LLM may occasionally use first-person in valid synthesis ("I observe that...")
        # Log warning but allow answer to pass through
        first_person_terms = [
            "I ", "my ", "me ", "I'm", "I've", "I was", "I am", "I have",
            "I received", "I called", "I tried", "I contacted"
        ]
        for term in first_person_terms:
            if term.lower() in clean_answer.lower():
                logger.warning(f"⚠ First-person language detected ('{term}') - may indicate verbatim text")
                # CHANGED: Continue validation instead of returning fallback
                break
        
        # QUALITY CHECK 3: Verbatim complaint text (substring match)
        # Check if answer is a substantial substring of any retrieved complaint
        for doc in retrieved_docs:
            doc_text = doc['text']
            # Only check for substantial matches (50+ chars) to avoid false positives
            if len(clean_answer) > 50:
                # Case-insensitive substring match
                if clean_answer.lower() in doc_text.lower():
                    logger.warning("⚠ Answer appears to be verbatim complaint text, rejected")
                    return "Unable to synthesize information. Please review the sources below."
        
        # All critical quality checks passed
        logger.info("✓ Answer passed quality validation")
        return answer
    
    def answer_question(
        self,
        question: str,
        k: int = 5,
        max_new_tokens: int = 512,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        End-to-end RAG pipeline: retrieve + deduplicate + generate + validate.
        
        REFACTORED WORKFLOW:
        1. Retrieve top-k documents from vector store
        2. Deduplicate documents (remove redundant complaints)
        3. Generate answer with relaxed prompt and flexible token limits (512-2000)
        4. Validate answer quality (relaxed validation, warnings instead of hard failures)
        5. Return validated answer + sources
        
        Args:
            question: User question
            k: Number of documents to retrieve
            max_new_tokens: Maximum tokens for generation (default: 512, max: 2000)
            temperature: Sampling temperature (default: 0.0 for deterministic output)
        
        Returns:
            Dict with 'answer' and 'sources' keys:
            {
                "answer": str,  # Validated, synthesized answer
                "sources": [    # Original sources for UI display
                    {
                        "text": str,
                        "metadata": dict
                    }
                ]
            }
        """
        logger.info("=" * 80)
        logger.info(f"RAG QUERY: {question}")
        logger.info("=" * 80)
        
        # STEP 1: Retrieve documents
        retrieved_docs = self.retrieve_documents(question, k=k)
        
        # STEP 1b: Filter by Product Category (NEW)
        # Filters based on product keywords in user question
        filtered_docs = self.filter_by_product_category(question, retrieved_docs)
        
        # STEP 2: Deduplicate documents
        # Removes duplicate/similar complaints to improve context quality
        deduplicated_docs = self.deduplicate_documents(filtered_docs)
        
        # STEP 3: Generate answer with deduplicated context (IMPROVED)
        # Uses improved prompt template and deterministic generation settings
        answer = self.generate_answer(
            question,
            deduplicated_docs,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        # STEP 4: Validate answer quality (NEW)
        # Detects and replaces bad answers (verbatim text, first-person, too short)
        validated_answer = self.validate_answer(answer, deduplicated_docs)
        
        # STEP 5: Format response
        # Return validated answer + original sources (not deduplicated) for UI display
        response = {
            "answer": validated_answer,
            "sources": retrieved_docs  # Original sources for transparency
        }
        
        logger.info("=" * 80)
        logger.info("RAG PIPELINE COMPLETE")
        logger.info("=" * 80)
        
        return response


# Convenience functions for standalone usage

def load_vector_store(
    vector_store_path: str = "vector_store/",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Chroma:
    """
    Load vector store (standalone function).
    
    Args:
        vector_store_path: Path to persisted ChromaDB
        embedding_model: HuggingFace embedding model
    
    Returns:
        Loaded Chroma vector store
    """
    pipeline = RAGPipeline(
        vector_store_path=vector_store_path,
        embedding_model=embedding_model
    )
    pipeline.load_vector_store()
    return pipeline.vector_store


def retrieve_documents(
    question: str,
    k: int = 5,
    vector_store_path: str = "vector_store/"
) -> List[Dict[str, Any]]:
    """
    Retrieve documents (standalone function).
    
    Args:
        question: User question
        k: Number of documents to retrieve
        vector_store_path: Path to persisted ChromaDB
    
    Returns:
        List of dicts with 'text' and 'metadata'
    """
    pipeline = RAGPipeline(vector_store_path=vector_store_path)
    pipeline.load_vector_store()
    return pipeline.retrieve_documents(question, k=k)


def answer_question(
    question: str,
    k: int = 5,
    vector_store_path: str = "vector_store/",
    llm_model: str = "google/flan-t5-base"
) -> Dict[str, Any]:
    """
    Answer question using RAG (standalone function).
    
    Args:
        question: User question
        k: Number of documents to retrieve
        vector_store_path: Path to persisted ChromaDB
        llm_model: HuggingFace LLM model
    
    Returns:
        Dict with 'answer' and 'sources'
    """
    pipeline = RAGPipeline(
        vector_store_path=vector_store_path,
        llm_model=llm_model
    )
    pipeline.load_vector_store()
    pipeline.load_llm()
    return pipeline.answer_question(question, k=k)


def main():
    """Demo usage of refactored RAG pipeline."""
    # Initialize pipeline
    pipeline = RAGPipeline(
        vector_store_path="vector_store/",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="google/flan-t5-base",
        device="cpu"
    )
    
    # Load components
    pipeline.load_vector_store()
    pipeline.load_llm()
    
    # Test questions
    test_questions = [
        "What are common issues with mortgages?",
        "What complaints mention fraud or identity theft?",
        "What are the main problems with credit card billing?"
    ]
    
    # Run queries
    for question in test_questions:
        print("\n" + "=" * 80)
        print(f"Question: {question}")
        print("=" * 80)
        
        result = pipeline.answer_question(question, k=5)
        
        print(f"\nAnswer:\n{result['answer']}\n")
        print(f"Sources ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"\n  [{i}] {source['metadata'].get('product', 'Unknown')} - {source['metadata'].get('issue', 'Unknown')}")
            print(f"      {source['text'][:150]}...")


if __name__ == "__main__":
    main()
