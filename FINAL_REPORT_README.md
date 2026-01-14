# Intelligent Complaint Analysis System (RAG)

## 1. Abstract

This project presents a robust Retrieval-Augmented Generation (RAG) system designed to automate the analysis of consumer financial complaints from the Consumer Financial Protection Bureau (CFPB) database. By integrating semantic search with a generative Large Language Model (LLM), the system overcomes the limitations of traditional keyword-based querying, enabling analysts to extract synthesized insights from unstructured narrative data. The architecture combines a ChromaDB vector store for high-precision retrieval with a FLAN-T5-base model for grounded answer generation. Key innovations include a product-category filtering mechanism to reduce retrieval noise, a deduplication layer to maximize context utility, and a strict synthesis-focused prompt design that enforces evidence-backed summarization. Evaluation on a set of representative financial inquiries demonstrates the system's ability to produce accurate, context-aware responses, achieving an average quality score of 3.7/5 across diverse complaint categories.

## 2. Introduction

The Consumer Financial Protection Bureau (CFPB) collects millions of complaints regarding financial products and services, serving as a critical dataset for identifying systemic risks and ensuring regulatory compliance. Financial institutions and oversight bodies face a significant challenge in processing this volume; manual review is unscalable, while distinct narratives often contain complex, jargon-heavy descriptions of specific grievances.

This project investigates the application of Retrieval-Augmented Generation (RAG) to democratize access to this data. Unlike traditional dashboards that aggregate structured metadata (e.g., "count of complaints by product"), this system targets the unstructured *narrative* text, allowing Product Managers, Compliance Officers, and Customer Support leads to ask natural language questions (e.g., "What are the emerging fraud patterns in mortgage servicing?"). The objective is to deliver a tool that offers grounded, explainable AI responses, bridging the gap between raw data and actionable intelligence.

## 3. Problem Statement

Standard analytical approaches to the CFPB dataset rely heavily on exact keyword matching or high-level statistical aggregation. These methods suffer from two primary limitations:
1.  **Semantic Gap**: Keyword searches fail to capture synonymous concepts (e.g., "billing error" vs. "wrong amount charged") or context-dependent meanings.
2.  **Information Overload**: Retrieving hundreds of raw complaint texts places the burden of synthesis on the analyst, slowing down decision-making.

Furthermore, applying off-the-shelf generative AI models directly to the raw data is prone to hallucinations and lacks traceability. There is a critical need for a grounded system that not only answers questions but also cites specific evidence from the underlying data, ensuring trust and verification in a regulated industry context.

## 4. System Architecture Overview

The system follows a modular RAG architecture designed for local execution and high explainability.

### Data Flow Architecture

1.  **Data Ingestion**: Raw CSV data is ingested, cleaned, and filtered to remove records without narratives.
2.  **Embedding**: Text is segmented into chunks and converted into dense vector embeddings.
3.  **Vector Storage**: Embeddings are indexed in ChromaDB with rich metadata for hybrid filtering.
4.  **Retrieval**: User queries trigger a semantic search to retrieve the top-k most relevant text chunks.
5.  **Refinement**: Retrieved results undergo product-category filtering and deduplication.
6.  **Prompt Construction**: A strict template combines system instructions, retrieved context, and the user query.
7.  **Generation**: The LLM (FLAN-T5) synthesizes a response based *only* on the provided context.
8.  **Evaluation/Interaction**: The output is presented via a Streamlit UI, alongside the raw source citations.

### Text-Based Architecture Diagram

```
[CFPB Dataset] 
      │
      ▼
[Preprocessing & Cleaning] ──> [Task 1: Gold Dataset]
      │
      ▼
[Chunking & Embedding] ────> [Sentence-Transformers]
      │
      ▼
[Vector Store (ChromaDB)] <──> [Retrieval Logic]
                                      ^
                                      │
                                [User Query]
                                      │
      ┌───────────────────────────────┴───────────────────────────────┐
      ▼                                                               ▼
[Retrieval Refinement]                                      [Interactive UI]
(Product Filter -> Deduplication)                               (Streamlit)
      │                                                               ▲
      ▼                                                               │
[Prompt Engineering] ──> [LLM (FLAN-T5)] ──> [Synthesized Answer] ────┘
```

## 5. Methodology

### 5.1 Data Preparation (Task 1)
The project utilized the public CFPB dataset, filtering the original 9.6 million records down to ~2.98 million rows that contained non-null consumer narratives. 
*   **Cleaning**: Removal of PII placeholders (e.g., `XXXX`) and standardization of whitespace.
*   **Sampling**: To balance computational feasibility with representation, a stratified sample of 12,000 records was created, ensuring coverage of minority product categories alongside dominant ones like "Credit reporting".
*   **Chunking**: Narratives were segmented into 500-character chunks with a 50-character overlap to preserve semantic context across boundaries.

### 5.2 Vector Store Construction (Task 2)
The vector store was implemented using **ChromaDB**, chosen for its efficiency and local persistence capabilities.
*   **Embeddings**: The `sentence-transformers/all-MiniLM-L6-v2` model was selected for its high performance-to-size ratio (384 dimensions), making it suitable for CPU inference.
*   **Metadata**: Key fields (`product`, `issue`, `company`, `date`) were stored alongside embeddings, enabling the system to perform pre-filtering or post-retrieval validation.
*   **Persistence**: The database is persisted to disk, allowing instant reload of the 12,000+ vectors without re-indexing.

### 5.3 Retrieval & Evaluation (Task 2)
Retrieval quality was optimized through a multi-stage process:
1.  **Similarity Search**: Using cosine similarity to find the nearest semantic neighbors.
2.  **Product Filtering**: A heuristic layer that maps query terms (e.g., "mortgage") to dataset categories, filtering out irrelevant hits (e.g., credit card complaints) before they reach the LLM.
3.  **Refinement**: A deduplication step removes near-identical chunks to maximize the information density presented to the model.

### 5.4 RAG Pipeline Design (Task 3)
The core generation logic resides in `src/rag_pipeline.py`.
*   **Prompt Philosophy**: A "Synthesis-First" approach was adopted. The system prompt explicitly instructs the model to identify *patterns* across documents rather than summarizing individual texts. It enforces a structured output format (numbered list) to improve readability.
*   **LLM Selection**: `google/flan-t5-base` was chosen for its strong instruction-following capabilities relative to its size (250M parameters), enabling completely local execution without external API dependencies.
*   **Guardrails**: The pipeline includes post-processing checks to detect hallucinations or verbatim copying, ensuring that the output is a true synthesis.

### 5.5 Interactive UI (Task 4)
A **Streamlit** application (`ui/app.py`) serves as the front-end.
*   **Goals**: To verify system performance in real-time and demonstrate usability.
*   **Design**: Features a chat interface where users can adjust retrieval parameters (k-sources, max token length).
*   **Transparency**: Crucially, the UI displays the "Retrieved Sources" in an expandable section below every answer, allowing users to verify the AI's claims against the ground truth data.

## 6. Implementation Details

The system is built in Python 3.10 and modularized into clear components:
*   `src/build_vector_store.py`: Handles data ingestion and indexing.
*   `src/rag_pipeline.py`: The central engine containing the `RAGPipeline` class, which manages loading, retrieval, and generation.
*   `ui/app.py`: The presentation layer.

**Key Design Decisions:**
*   **Deterministic Generation**: The LLM generation parameters are set to `do_sample=False` and `temperature=0.0`. This ensures that the same query always yields the same response, a critical requirement for compliance and reproducibility.
*   **Open-Source Strategy**: By using FLAN-T5 and ChromaDB, the entire stack is open-source and free to run, avoiding data privacy issues associated with sending financial data to commercial APIs.

## 7. Evaluation & Results

The RAG pipeline was evaluated qualitatively using a diverse set of 10 financial domain questions.
*   **Scoring Criteria**: Responses were rated on a 1-5 scale based on **Relevance** to the query, **Groundedness** in the retrieved context, and **Clarity** of synthesis.
*   **Results**: The system achieved an average score of **3.70/5.0**.
    *   **Strengths**: Strong performance in identifying generalized issues (e.g., "common mortgage complaints"). The product filtering successfully removed irrelevant context.
    *   **Weaknesses**: Occasional brevity in responses due to the conservative token limit of the base model.
*   **Example Outcome**: When asked about "unauthorized credit card charges," the system correctly synthesized multiple reports of liability disputes and merchant errors into a cohesive summary, rather than listing disparate anecdotes.

## 8. Testing Strategy

Quality assurance was integrated throughout the development lifecycle:
*   **LLM Sanity Testing**: Simple prompts were run against the loaded model to verify instruction following before connecting the retrieval layer.
*   **RAG Smoke Testing**: Automated scripts (`tests/test_rag_quality.py`) run the full pipeline against known queries to ensure no regressions in retrieving expected documents.
*   **Validation**: The pipeline includes runtime validation to catch empty responses or failures in the vector store connection, providing graceful error messages to the user.

## 9. Limitations

*   **Model Constraints**: `FLAN-T5-base` has a limited context window (typically 512-1024 tokens). This restricts the number of documents (k) that can be fed into the prompt, potentially omitting relevant details.
*   **Retrieval Noise**: Despite filtering, semantic search sometimes retrieves "noise" (e.g., legal disclaimers included in complaint text) which can distract the model.
*   **Evaluation Subjectivity**: The current evaluation metric relies on human judgment. A larger-scale automated evaluation (e.g., RAGAS) would provide more statistical rigor.

## 10. Future Work

*   **Advanced Filtering**: Implementing metadata filtering at the database query level (Hybrid Search) rather than post-retrieval would improve efficiency.
*   **Stronger Models**: Upgrading to `FLAN-T5-large` or a quantized 7B parameter model (like Mistral or Llama-2) would significantly enhance reasoning capabilities and allowable context length.
*   **Quantitative Metrics**: Integrating automated metrics like BLEU or ROUGE scores compared against a set of "golden answers" would refine the evaluation process.
*   **Production Readiness**: Containerizing the application using Docker and exposing the RAG pipeline via a REST API (FastAPI) would be the next step for deployment.

## 11. Conclusion

This project successfully demonstrates the viability of a local, privacy-preserving RAG system for financial data analysis. By combining effective preprocessing strategies with a tuned retrieval pipeline, we achieved a system that transforms raw, unstructured noise into clear, actionable insights. The work highlights the critical importance of data quality and retrieval precision in RAG systems; even a modest LLM can perform exceptionally well when fed high-quality, relevant context. This architecture provides a solid foundation for enterprise-grade complaint analytics tools.

## 12. Repository Structure

```
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for EDA and prototyping
├── reports/                # Evaluation reports and figures
├── src/                    # Source code for the core pipeline
│   ├── build_vector_store.py   # Script to populate ChromaDB
│   ├── rag_pipeline.py         # Main RAG logic and class definitions
│   ├── evaluate_rag.py         # Evaluation scripts
│   └── task1_eda.py            # Data processing modules
├── tests/                  # Automated test suite
├── ui/                     # Streamlit frontend application
├── vector_store/           # Persisted ChromaDB embeddings
├── README.md               # Quickstart guide
└── requirements.txt        # Python dependencies
```
