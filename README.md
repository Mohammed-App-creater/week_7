# Intelligent Complaint Analysis for Financial Services (RAG-Powered Chatbot)

## 1. Project Overview

**Client:** CrediTrust Financial

In the highly regulated financial services sector, timely and accurate responses to customer complaints are critical for maintaining trust and compliance. CrediTrust Financial receives a high volume of consumer complaints across various channels. Manual triage and analysis are time-consuming and prone to human error, leading to delayed resolutions and missed insights.

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot designed to intelligently query, analyze, and summarize consumer complaints. By leveraging Large Language Models (LLMs) grounded in a vector database of historical complaints, the system provides accurate, context-aware answers to analysts' queries, significantly reducing the time required to understand complaint trends and specifics.

## 2. Business Objectives & KPIs

The primary goal is to transform complaint data into actionable intelligence.

*   **Time-to-Insight Reduction:** Decrease the time analysts spend searching for relevant past complaints by 60%.
*   **Empowerment of Non-Technical Teams:** Enable product managers and compliance officers to query complex datasets using natural language without needing SQL or Python skills.
*   **Proactive Issue Detection:** Identify emerging issues and systemic problems faster through semantic similarity search rather than keyword matching.

## 3. System Architecture

The solution follows a modern RAG architecture:

1.  **Data Ingestion:** Loading raw CSV data from the Consumer Financial Protection Bureau (CFPB).
2.  **Preprocessing:** Cleaning text, handling missing values, and filtering for relevant product categories.
3.  **Embedding & Vector Store:** Converting complaint narratives into high-dimensional vectors using `sentence-transformers` and persisting them in `ChromaDB` for efficient similarity search.
4.  **Retrieval-Augmented Generation:**
    *   **Retrieval:** Querying the vector store to find the most relevant context chunks for a user's question.
    *   **Generation:** Passing the retrieved context and the user's query to an LLM (via Hugging Face) to generate a coherent, grounded response.
5.  **User Interface:** An interactive frontend built with Gradio/Streamlit to facilitate easy interaction for end-users.

## 4. Dataset

The system utilizes the **Consumer Financial Protection Bureau (CFPB) Complaints Dataset**.

*   **Source:** Publicly available financial complaint data.
*   **Target Scope:** The project focuses on four high-impact product categories:
    *   Credit card
    *   Personal loan
    *   Savings account
    *   Money transfer
*   **Embeddings:** We utilize pre-built embeddings (`sentence-transformers/all-MiniLM-L6-v2`) to ensure high-quality semantic representation without the computational cost of training models from scratch.

## 5. Project Structure

```bash
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for EDA and prototyping (Task 1)
├── src/                    # Source code for vector store and RAG pipeline (Task 2 & 3)
├── tests/                  # Unit tests for preprocessing and vector store
├── vector_store/           # Persisted ChromaDB vector database
├── requirements.txt        # Python dependency definitions
└── README.md               # Project documentation
```

## 6. Tasks Breakdown

*   **Task 1: EDA & Preprocessing:** Data exploration, cleaning, and preparation of the "Gold" dataset.
*   **Task 2: Chunking & Vector Store:** Implementing text chunking strategies and populating the ChromaDB vector database.
*   **Task 3: RAG Pipeline:** Developing the retrieval logic and integrating the LLM for answer generation.
*   **Task 4: Interactive UI:** Building the frontend interface for users to interact with the system.

## 7. Tech Stack

*   **Language:** Python 3.10+
*   **Data Manipulation:** pandas, numpy
*   **Vector Operations:** sentence-transformers, ChromaDB
*   **LLM Integration:** Hugging Face Transformers
*   **Interface:** Gradio / Streamlit
*   **Testing:** pytest

## 8. How to Run

### Environment Setup

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Pipeline

1.  **Preprocessing (Task 1):**
    Run the preprocessing notebook or script to generate `data/filtered_complaints.csv`.

2.  **Build Vector Store (Task 2):**
    Execute the script to ingest data and create embeddings:
    ```bash
    python src/build_vector_store.py
    ```

3.  **Run the App (Task 3 & 4):**
    Launch the Gradio/Streamlit interface:
    ```bash
    python src/app.py
    ```

### Running Tests

Execute the unit test suite to verify system integrity:
```bash
pytest tests/
```

## 9. Testing & CI

The project maintains high code quality standards through:

*   **Unit Testing:** Comprehensive tests implementation using `pytest` covering data integrity, processing logic, and vector store operations.
*   **CI/CD:** A GitHub Actions workflow (`.github/workflows/unittests.yml`) triggers on every push and pull request to ensure no regression is introduced.

## 10. Future Improvements

*   **Advanced Reranking:** Implement a Cross-Encoder step to re-rank retrieved results for higher precision.
*   **Evaluation Metrics:** Integrate RAGAS or similar frameworks to quantitatively evaluate the faithfulness and answer relevance of the generation.
*   **Role-Based Access:** Add authentication to restrict access to sensitive complaint data.
*   **Streaming & Caching:** Implement response streaming for better UX and semantic caching to reduce API costs and latency.
