# Task 2 & 3: Vector Store & RAG Pipeline

## 🎯 Overview
This directory contains the core backend logic for the RAG system. It covers **Task 2 (Vector Store Creation)** and **Task 3 (RAG Pipeline Implementation)**.

## 🏗️ Architecture Components

### Task 2: Vector Store (`build_vector_store.py`)
Responsible for converting text into embeddings and persisting them.
- **Chunking:** 512 characters with 50-character overlap.
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
- **Storage:** ChromaDB (Serverless mode).
- **Metadata:** Stores `product_category` and `issue` alongside vectors for filtering.

### Task 3: RAG Core (`rag_pipeline.py`)
Responsible for answering user queries.
1.  **Retrieval:** Fetches top-k (k=5) semantic matches from ChromaDB.
2.  **Product Filtering:** (New) Heuristically matches user query terms to products (e.g., "card" -> "Credit card") to filter results.
3.  **Deduplication:** Removes duplicate complaints to maximize context window usage.
4.  **Generation:** Uses `google/flan-t5-base` to synthesize an answer.
    - **Prompt:** "Synthesis-focused" template forcing bullet-point verification.
    - **Guardrails:** Checks for "Not enough info" or hallucinations.

## 📂 Key Files
| File | Description |
| :--- | :--- |
| `build_vector_store.py` | Script to chunk data and populate ChromaDB. |
| `rag_pipeline.py` | The main `RAGPipeline` class containing retrieval and generation logic. |
| `evaluate_rag.py` | Evaluation script to run qualitatve tests and generate reports. |

## 🚀 How to Run

### Build the Vector Store (Task 2)
**Prerequisite:** Ensure `data/filtered_complaints.csv` exists (from Task 1).
```bash
python src/build_vector_store.py
```

### Run Evaluation (Task 3)
To verify the RAG pipeline performance:
```bash
python src/evaluate_rag.py
```
*This will generate `reports/task3_evaluation.md`.*

### Interactive Demo
You can run a quick interactive CLI demo by executing the pipeline script directly:
```bash
python src/rag_pipeline.py
```

## 📊 Evaluation Results (Task 3)
The qualitative evaluation (`evaluate_rag.py`) assesses the system on 10 standardized questions.
- **Average Score:** ~4.0/5.0
- **Strengths:** Strong performance on direct product questions ("What are common issues with mortgages?").
- **Weaknesses:** Occasional synthesis limits due to `flan-t5-base` context window.
