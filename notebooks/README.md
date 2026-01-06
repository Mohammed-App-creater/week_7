# Task 1: Exploratory Data Analysis & Preprocessing

## 1. Objective

The primary objective of Task 1 is to prepare a clean, high-quality dataset suitable for semantic search and RAG operations. This involves ingesting the raw CFPB complaints data, exploring its characteristics, and applying rigorous cleaning and filtering rules to produce a "Gold" dataset.

## 2. Data Loading & EDA Steps

The analysis begins by loading the raw dataset and performing an initial assessment:
*   **Schema Verification:** Confirming the presence of critical columns (Consumer Complaint Narrative, Product, Issue).
*   **Distribution Analysis:** Visualizing the distribution of complaints across products and states.
*   **Null Value Assessment:** Quantifying missing data in the narrative column which is essential for the embeddings.

## 3. Filtering Logic

To ensure the RAG system focuses on relevant business units, the data is filtered based on strict criteria:

*   **Product Allowed List:**
    *   Credit card
    *   Personal loan
    *   Savings account
    *   Money transfer
*   **Narrative Existence:** Rows with missing or empty `Consumer complaint narrative` fields are dropped, as they provide no context for the LLM.

## 4. Text Cleaning Approach

Text preprocessing is minimal yet effective to preserve semantic meaning while normalizing the inputs:
*   **Lowercasing:** All narratives are converted to lowercase to standardize the vocabulary.
*   **Whitespace Normalization:** Excess whitespace is trimmed.
*   **PII Handling:** The dataset assumes PII has been redacted by the source (CFPB) via 'X' masking (e.g., "XXXX"). Further regex-based cleaning is applied if additional artifacts are detected.

## 5. Output Artifact

*   **File:** `data/filtered_complaints.csv`
*   **Content:** A CSV containing only valid, non-empty, allowed-product complaints with normalized narrative text.

## 6. How to Run

1.  Ensure `requirements.txt` dependencies are installed.
2.  Navigate to the `notebooks/` directory.
3.  Open `Task1_EDA_Preprocessing.ipynb`.
4.  Run all cells sequentially.
5.  Verify that `data/filtered_complaints.csv` is generated in the parent `data/` directory.

## 7. Key Observations

*   **Data Imbalance:** "Credit card" complaints significantly outnumber "Savings account" and "Personal loan" complaints.
*   **Narrative Length:** The length of complaint narratives varies widely, indicating a need for effective chunking strategies in Task 2.
*   **Redaction patterns:** Frequent use of "XXXX" for dates and account numbers ensures privacy but may affect sentence structure slightly.
