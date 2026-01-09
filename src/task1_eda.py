
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_task1_eda_and_filtering():
    """
    Executes Task 1: EDA and Data Preprocessing.
    1. Loads raw data.
    2. Generates and saves EDA figures.
    3. Filters data for narratives.
    4. Saves filtered data.
    """
    logger.info("Starting Task 1: EDA and Preprocessing...")

    # Define paths
    raw_data_path = Path("data/raw/complaints.csv")
    output_data_dir = Path("data")
    output_figures_dir = Path("reports/figures")
    
    # Ensure directories exist
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    if not raw_data_path.exists():
        logger.error(f"Raw data not found at {raw_data_path}")
        # In a real scenario we might stop here, but for this task I'll create a dummy file if missing just to let the code 'run' via unit tests or similar, 
        # BUT the user context says data/filtered_complaints.csv is produced from Task 1. 
        # I must assume the USER has put the file there as per instructions "Load the raw CFPB dataset".
        # If it doesn't exist, I'll error out.
        # However, checking the file list, I see 'data' dir but didn't list 'data/raw'. Let's assume it might be there or I need to warn.
        # Actually list_dir of 'data' showed numChildren=1. Let's check what is in data.
        raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")

    logger.info(f"Loading data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    logger.info(f"Loaded {len(df):,} rows.")

    # 2. EDA & Figures
    
    # Figure 1: Complaints by Product
    logger.info("Generating 'complaints_by_product.png'...")
    plt.figure(figsize=(12, 6))
    product_counts = df['Product'].value_counts().head(15) # Top 15
    sns.barplot(x=product_counts.values, y=product_counts.index, palette='viridis')
    plt.title('Top 15 Products by Complaint Count')
    plt.xlabel('Number of Complaints')
    plt.tight_layout()
    plt.savefig(output_figures_dir / "complaints_by_product.png")
    plt.close()

    # Figure 2: Narratives Missing vs Present
    logger.info("Generating 'narratives_missing_vs_present.png'...")
    plt.figure(figsize=(8, 6))
    # Standardize column name just in case
    narrative_col = 'Consumer complaint narrative'
    if narrative_col not in df.columns:
         # Try to find it
         cols = [c for c in df.columns if 'narrative' in c.lower()]
         if cols:
             narrative_col = cols[0]
    
    has_narrative = df[narrative_col].notna()
    counts = has_narrative.value_counts()
    counts.index = ['Present' if x else 'Missing' for x in counts.index]
    sns.barplot(x=counts.index, y=counts.values, palette='pastel')
    plt.title('Count of Complaints with vs. without Narratives')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_figures_dir / "narratives_missing_vs_present.png")
    plt.close()

    # Figure 3: Narrative Length Distribution
    logger.info("Generating 'narrative_length_distribution.png'...")
    df_narratives = df[has_narrative].copy()
    # simple word count split by whitespace
    df_narratives['word_count'] = df_narratives[narrative_col].astype(str).apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df_narratives['word_count'], bins=50, kde=True, color='purple')
    plt.title('Distribution of Complaint Narrative Lengths (Word Count)')
    plt.xlabel('Word Count')
    plt.xlim(0, 1000) # Limit x-axis to see the bulk
    plt.tight_layout()
    plt.savefig(output_figures_dir / "narrative_length_distribution.png")
    plt.close()

    # 3. Preprocessing (Filtering)
    logger.info("Filtering for records with narratives...")
    df_filtered = df_narratives  # We already created this subset
    
    # Save Filtered Data
    output_path = output_data_dir / "filtered_complaints.csv"
    df_filtered.to_csv(output_path, index=False)
    logger.info(f"Saved filtered data to {output_path}. Rows: {len(df_filtered):,}")

    logger.info("Task 1 Completed successfully.")
    
    # 4. Return metrics map for report generation
    return {
        "raw_count": len(df),
        "filtered_count": len(df_filtered),
        "products_retained": list(df_filtered['Product'].unique())[:5] # Just sample
    }

if __name__ == "__main__":
    run_task1_eda_and_filtering()
