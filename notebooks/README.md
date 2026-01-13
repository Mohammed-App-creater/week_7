# Task 1: Exploratory Data Analysis & Preprocessing

## 🎯 Objective
To transform the raw CFPB complaint dataset into a clean, high-quality "Gold" dataset suitable for semantic embedding. This involves identifying key columns, analyzing data distribution, and applying strict filtering rules to remove unrelated noises.

## 📊 Key Implementation Differences
- **Notebook-Driven:** Unlike other tasks, this stage is primarily explorative and executed via Jupyter Notebook.
- **Strict Filtering:** We drop ~80% of data to focus ONLY on 4 key product categories, ensuring the RAG system specializes in high-value banking domains.

## 🧹 Preprocessing Steps
1.  **Ingestion:** Load raw CSV data.
2.  **Schema Validation:** Ensure `Consumer complaint narrative` and `Product` columns exist.
3.  **Filtration:**
    *   **Keep only:** "Credit card", "Personal loan", "Savings account", "Money transfer".
    *   **Drop:** Rows with missing narratives.
4.  **Text Cleaning:**
    *   Lowercase transformation.
    *   Redaction handling (preserving "XXXX" masks).
    *   Whitespace normalization.

## 📁 Files
- `task1_eda_preprocessing.ipynb`: The main notebook containing all logic and visualizations.

## 🚀 How to Run
1.  Navigate to this directory:
    ```bash
    cd notebooks/
    ```
2.  Launch Jupyter:
    ```bash
    jupyter notebook
    ```
3.  Open and run all cells in `task1_eda_preprocessing.ipynb`.
4.  **Output:** The script will save `filtered_complaints.csv` to the `data/` directory.
