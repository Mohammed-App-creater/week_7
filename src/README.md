# Task 2: Chunking & Vector Store Creation

## 1. Objective

Task 2 focuses on converting the preprocessed textual data into a machine-understandable format. This involves chunking long narratives into manageable pieces, generating vector embeddings, and persisting them in a specialized vector database (ChromaDB) to enable efficient semantic retrieval.

## 2. Stratified Sampling Strategy

Due to potential computational constraints and the high volume of data:
*   We employ **Stratified Sampling** based on the `Product` category.
*   This ensures that the vector store maintains a representative distribution of all four business lines (Credit card, Personal loan, Savings account, Money transfer), preventing any single category from dominating the retrieval results.

## 3. Text Chunking Approach

Complaint narratives can be lengthy. To optimize context retrieval for the LLM:

*   **Chunk Size:** 512 characters.
    *   *Justification:* This size captures sufficient context to understand a specific issue without overwhelming the LLM's context window with irrelevant details.
*   **Overlap:** 50 characters.
    *   *Justification:* Provides continuity between chunks, ensuring that context is not lost if a sentence is split across boundaries.

## 4. Embedding Model Choice

*   **Model:** `sentence-transformers/all-MiniLM-L6-v2`
*   **Reasoning:**
    *   **Performance:** Offers an excellent balance between speed and semantic accuracy.
    *   **Dimensions:** Produces 384-dimensional vectors, which are storage-efficient.
    *   **Open Source:** Fully accessible via Hugging Face, ensuring reproducibility.

## 5. Vector Store Choice

*   **Database:** ChromaDB
*   **Type:** Embedded, serverless vector database.
*   **Why ChromaDB?**
    *   Easy integration with Python and LangChain.
    *   Persistent storage capabilities without needing a separate container or extensive infrastructure.
    *   Fast similarity search performance for datasets of this scale.

## 6. Metadata Stored with Embeddings

To facilitate precise filtering and reference generation, each embedding is stored with the following metadata:
*   `complaint_id`: Unique identifier for the original record.
*   `product_category`: The financial product related to the complaint.
*   `issue`: The specific sub-issue reported.
*   `source_text`: The raw text chunk (for reconstruction during retrieval).

## 7. Output Artifact

*   **Directory:** `vector_store/`
*   **Contents:** The persistent ChromaDB database files (SQLite and binary index files). This directory stores the learned embeddings and index, allowing the application to start without re-ingesting data.

## 8. How to Run

1.  Ensure `data/filtered_complaints.csv` exists (output of Task 1).
2.  Run the build script from the project root:
    ```bash
    python src/build_vector_store.py
    ```
3.  Upon completion, verify that the `vector_store/` directory has been populated.
