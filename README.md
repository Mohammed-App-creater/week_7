# Intelligent Complaint Analysis for Financial Services (RAG Hybrid System)

## 📌 Abstract
This project implements a Retrieval-Augmented Generation (RAG) system designed to automate the analysis of consumer financial complaints. By leveraging the Consumer Financial Protection Bureau (CFPB) dataset, the system uses semantic search and Large Language Models (LLMs) to provide accurate, context-aware answers to natural language queries. The solution aims to reduce analyst workload, identify systemic issues faster, and democratize data access for non-technical stakeholders.

## 🚀 Problem Statement & Motivation
In the financial services sector, regulatory compliance and customer trust are paramount. Analyzing thousands of narrative complaints manually is inefficient and prone to error. Traditional keyword-based search fails to capture the semantic nuance of customer grievances (e.g., distinguishing "fraud" from "accounting error").

**Core Challenges:**
- **Volume:** High intake of unstructured text data.
- **Complexity:** Financial products have specific jargon and regulatory contexts.
- **Latency:** Manual triage hampers rapid response to emerging risks.

This project addresses these challenges by building an automated pipeline that retrieves relevant historical context and synthesizes insights using generative AI.

## 🏗️ High-Level Architecture

```mermaid
graph LR
    A[CFPB Data] --> B(Task 1: EDA & Preprocessing)
    B --> C{Task 2: Vector Store}
    C -->|Embeddings| D[ChromaDB]
    D --> E(Task 3: RAG Pipeline)
    E -->|Retrieval| F[Context]
    G[User Query] --> E
    F --> H[LLM (FLAN-T5)]
    H --> I[Synthesized Answer]
    E <--> J(Task 4: Interactive UI)
```

**Data Flow:**
1.  **Ingest:** Raw CSV data is cleaned and filtered (Task 1).
2.  **Embed:** Narratives are chunked and embedded into a vector space (Task 2).
3.  **Retrieve:** User queries fetch top-k similar complaints (Task 3).
4.  **Generate:** An LLM synthesizes the retrieval context into a concise answer (Task 3).
5.  **Interact:** Users engage via a Streamlit interface (Task 4).

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Database:** ChromaDB (Persisted Vector Store)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** `google/flan-t5-base` (Hugging Face Transformers)
- **Interface:** Streamlit
- **Testing:** Pytest

## 📋 Task Breakdown

| Task | Description | Location |
| :--- | :--- | :--- |
| **Task 1** | **EDA & Preprocessing:** Data cleaning, product filtering, and "Gold" dataset creation. | `notebooks/` |
| **Task 2** | **Vector Store Creation:** Chunking strategies, embedding generation, and database population. | `src/` |
| **Task 3** | **RAG Pipeline:** Core logic for retrieval, prompt engineering, generation, and qualitative evaluation. | `src/` |
| **Task 4** | **Interactive UI:** A user-friendly chat interface for querying the system. | `ui/` |

## 💻 How to Run

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
git clone <repository_url>
cd <repository_folder>
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Pipeline Execution
**Step 1: Preprocessing (Task 1)**
Run the notebook `notebooks/task1_eda_preprocessing.ipynb` to generate `data/filtered_complaints.csv`.

**Step 2: Build Vector Store (Task 2)**
Ingest the data and generate embeddings:
```bash
python src/build_vector_store.py
```

**Step 3: Run the Application (Task 4)**
Launch the Streamlit interface:
```bash
streamlit run ui/app.py
```

### 3. Testing
Run the automated test suite:
```bash
pytest tests/
```

## 📊 Results & Evaluation
The RAG pipeline was evaluated qualitatively using a set of 10 representative financial domain questions.
- **Metric:** Quality Score (1-5 Scale) based on relevance, groundedness, and clarity.
- **Result:** The system achieves consistent scores of **4/5**, effectively retrieving relevant distinct complaints and synthesizing them into coherent summaries.
- **Key Insight:** Product-category filtering significantly improves precision by removing irrelevant context (e.g., credit reporting issues appearing in credit card queries).

## ⚠️ Limitations & Future Work
- **Model Size:** `FLAN-T5-base` is lightweight but may struggle with highly complex reasoning compared to larger models like GPT-4 or Llama-2.
- **Context Window:** Fixed chunking (512 chars) may split some long narratives; sliding windows or hierarchical indexing could improve context.
- **Latency:** CPU inference is functional but slower than GPU-accelerated environments.
