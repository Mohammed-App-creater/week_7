"""
Vector Store Builder for CFPB Complaints
Task 2: Build ChromaDB vector store with stratified sampling and text chunking.

Author: Data & AI Engineer
Date: 2026-01-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorStoreBuilder:
    """Build and persist ChromaDB vector store for complaint narratives."""
    
    def __init__(
        self,
        data_path: str = "data/filtered_complaints.csv",
        vector_store_path: str = "vector_store/",
        sample_size: int = 12000,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize VectorStoreBuilder.
        
        Args:
            data_path: Path to cleaned complaints dataset
            vector_store_path: Path to persist vector store
            sample_size: Number of complaints to sample (10000-15000)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: HuggingFace embedding model name
        """
        self.data_path = Path(data_path)
        self.vector_store_path = Path(vector_store_path)
        self.sample_size = sample_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model
        
        # Validate sample size
        if not 10000 <= sample_size <= 15000:
            raise ValueError("Sample size must be between 10,000 and 15,000")
        
        logger.info(f"Initialized VectorStoreBuilder with sample_size={sample_size}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load cleaned complaints dataset.
        
        Returns:
            DataFrame with complaints data
        """
        logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df):,} complaints")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    
    def create_stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create stratified sample by product category.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Stratified sample DataFrame
        """
        logger.info(f"Creating stratified sample of {self.sample_size:,} complaints")
        
        # Identify product column (case-insensitive)
        product_col = None
        for col in df.columns:
            if 'product' in col.lower():
                product_col = col
                break
        
        if product_col is None:
            raise ValueError("Product column not found in dataset")
        
        logger.info(f"Using product column: {product_col}")
        
        # Calculate sample size per product (proportional stratification)
        product_counts = df[product_col].value_counts()
        logger.info(f"Product distribution:\n{product_counts}")
        
        # Calculate samples per product
        total = len(df)
        samples_per_product = {}
        
        for product, count in product_counts.items():
            proportion = count / total
            sample_count = int(self.sample_size * proportion)
            # Ensure at least 1 sample per product
            samples_per_product[product] = max(1, sample_count)
        
        logger.info(f"Samples per product: {samples_per_product}")
        
        # Perform stratified sampling
        sampled_dfs = []
        for product, n_samples in samples_per_product.items():
            product_df = df[df[product_col] == product]
            
            # Sample with replacement if needed
            if len(product_df) < n_samples:
                logger.warning(f"Product '{product}' has only {len(product_df)} rows, sampling with replacement")
                sample = product_df.sample(n=n_samples, replace=True, random_state=42)
            else:
                sample = product_df.sample(n=n_samples, replace=False, random_state=42)
            
            sampled_dfs.append(sample)
        
        # Combine samples
        stratified_sample = pd.concat(sampled_dfs, ignore_index=True)
        
        # Shuffle
        stratified_sample = stratified_sample.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Created stratified sample: {len(stratified_sample):,} complaints")
        logger.info(f"Sample product distribution:\n{stratified_sample[product_col].value_counts()}")
        
        return stratified_sample
    
    def create_chunks(self, df: pd.DataFrame) -> List[Document]:
        """
        Create text chunks with metadata.
        
        Args:
            df: DataFrame with complaints
        
        Returns:
            List of LangChain Document objects
        """
        logger.info(f"Creating text chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Identify relevant columns
        narrative_col = None
        for col in df.columns:
            if 'cleaned_narrative' in col.lower():
                narrative_col = col
                break
        
        if narrative_col is None:
            # Fallback to narrative column
            for col in df.columns:
                if 'narrative' in col.lower():
                    narrative_col = col
                    break
        
        if narrative_col is None:
            raise ValueError("Narrative column not found in dataset")
        
        logger.info(f"Using narrative column: {narrative_col}")
        
        # Identify metadata columns
        metadata_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['product', 'issue', 'company', 'date', 'id']):
                metadata_columns.append(col)
        
        logger.info(f"Metadata columns: {metadata_columns}")
        
        # Create documents with chunks
        documents = []
        total_chunks = 0
        
        for idx, row in df.iterrows():
            narrative = str(row[narrative_col])
            
            # Skip empty narratives
            if not narrative or narrative.strip() == '':
                continue
            
            # Split text into chunks
            chunks = text_splitter.split_text(narrative)
            
            # Create document for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                # Build metadata
                metadata = {
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'complaint_id': str(idx),
                }
                
                # Add available metadata fields
                for col in metadata_columns:
                    col_lower = col.lower()
                    value = row[col]
                    
                    # Map to standardized field names
                    if 'product' in col_lower and 'product_category' not in metadata:
                        if 'sub' not in col_lower and 'category' not in col_lower:
                            metadata['product'] = str(value) if pd.notna(value) else 'Unknown'
                        metadata['product_category'] = str(value) if pd.notna(value) else 'Unknown'
                    elif 'issue' in col_lower:
                        if 'sub' in col_lower:
                            metadata['sub_issue'] = str(value) if pd.notna(value) else 'Unknown'
                        else:
                            metadata['issue'] = str(value) if pd.notna(value) else 'Unknown'
                    elif 'company' in col_lower:
                        metadata['company'] = str(value) if pd.notna(value) else 'Unknown'
                    elif 'date' in col_lower and 'received' in col_lower:
                        metadata['date_received'] = str(value) if pd.notna(value) else 'Unknown'
                
                # Ensure all required metadata fields exist
                for field in ['product_category', 'product', 'issue', 'sub_issue', 'company', 'date_received']:
                    if field not in metadata:
                        metadata[field] = 'Unknown'
                
                # Create document
                doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                documents.append(doc)
                total_chunks += 1
            
            # Log progress every 1000 complaints
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1:,} complaints, created {total_chunks:,} chunks")
        
        logger.info(f"Created {total_chunks:,} chunks from {len(df):,} complaints")
        logger.info(f"Average chunks per complaint: {total_chunks / len(df):.2f}")
        
        return documents
    
    def build_vector_store(self, documents: List[Document]) -> Chroma:
        """
        Build ChromaDB vector store with embeddings.
        
        Args:
            documents: List of Document objects
        
        Returns:
            Chroma vector store
        """
        logger.info(f"Building vector store with {len(documents):,} documents")
        logger.info(f"Using embedding model: {self.embedding_model_name}")
        
        # Initialize embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info("Embedding model loaded successfully")
        
        # Create ChromaDB vector store
        logger.info("Creating embeddings and building vector store...")
        
        # Process in batches to show progress
        batch_size = 1000
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        vector_store = None
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            if vector_store is None:
                # Create vector store with first batch
                vector_store = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=str(self.vector_store_path)
                )
            else:
                # Add subsequent batches
                vector_store.add_documents(documents=batch)
        
        logger.info("Vector store creation complete")
        
        return vector_store
    
    def persist_vector_store(self, vector_store: Chroma):
        """
        Persist vector store to disk.
        
        Args:
            vector_store: Chroma vector store
        """
        logger.info(f"Persisting vector store to {self.vector_store_path}")
        
        # Create directory if it doesn't exist
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Persist
        vector_store.persist()
        
        logger.info("Vector store persisted successfully")
        logger.info(f"Vector store location: {self.vector_store_path.absolute()}")
    
    def build(self):
        """Execute full pipeline: load, sample, chunk, embed, and persist."""
        logger.info("=" * 80)
        logger.info("STARTING VECTOR STORE BUILD PIPELINE")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Step 1: Load data
        df = self.load_data()
        
        # Step 2: Create stratified sample
        sampled_df = self.create_stratified_sample(df)
        
        # Step 3: Create chunks
        documents = self.create_chunks(sampled_df)
        
        # Step 4: Build vector store
        vector_store = self.build_vector_store(documents)
        
        # Step 5: Persist vector store
        self.persist_vector_store(vector_store)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("=" * 80)
        logger.info("VECTOR STORE BUILD PIPELINE COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total complaints sampled: {len(sampled_df):,}")
        logger.info(f"Total chunks/embeddings created: {len(documents):,}")
        logger.info(f"Vector store location: {self.vector_store_path.absolute()}")
        logger.info(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info("=" * 80)
        
        return vector_store


def main():
    """Main execution function."""
    # Initialize builder
    builder = VectorStoreBuilder(
        data_path="data/filtered_complaints.csv",
        vector_store_path="vector_store/",
        sample_size=12000,  # 10,000 - 15,000 range
        chunk_size=500,
        chunk_overlap=50,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Build vector store
    vector_store = builder.build()
    
    # Test query
    logger.info("\nTesting vector store with sample query...")
    test_query = "credit card unauthorized charges"
    results = vector_store.similarity_search(test_query, k=3)
    
    logger.info(f"Query: '{test_query}'")
    logger.info(f"Retrieved {len(results)} results:")
    for i, doc in enumerate(results, 1):
        logger.info(f"\nResult {i}:")
        logger.info(f"  Content: {doc.page_content[:150]}...")
        logger.info(f"  Metadata: {doc.metadata}")
    
    logger.info("\nVector store build and test completed successfully!")


if __name__ == "__main__":
    main()
