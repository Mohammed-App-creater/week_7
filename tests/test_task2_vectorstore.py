# tests/test_task2_vectorstore.py

import pytest
import chromadb
from pathlib import Path


@pytest.fixture
def vector_store_path():
    """Fixture for vector store directory path."""
    return Path("vector_store")


@pytest.fixture
def chroma_client(vector_store_path):
    """Fixture to initialize ChromaDB client."""
    return chromadb.PersistentClient(path=str(vector_store_path))


def test_vector_store_directory_exists(vector_store_path):
    """Test that the vector_store/ directory exists."""
    assert vector_store_path.exists(), f"Vector store directory not found at {vector_store_path}"
    assert vector_store_path.is_dir(), f"{vector_store_path} is not a directory"


def test_chromadb_persistence_files_exist(vector_store_path):
    """Test that ChromaDB persistence files exist inside vector_store/."""
    # ChromaDB creates a chroma.sqlite3 file in the persistence directory
    persistence_files = list(vector_store_path.glob("*"))
    
    assert len(persistence_files) > 0, \
        "No persistence files found in vector_store/ directory"


def test_vector_store_loads_without_errors(chroma_client):
    """Test that the vector store can be loaded without errors."""
    try:
        collections = chroma_client.list_collections()
        assert collections is not None
    except Exception as e:
        pytest.fail(f"Failed to load vector store: {e}")


def test_at_least_one_document_retrievable(chroma_client):
    """Test that at least one document can be retrieved from the vector store."""
    collections = chroma_client.list_collections()
    
    assert len(collections) > 0, "No collections found in vector store"
    
    # Get the first collection
    collection = collections[0]
    
    # Query for documents (limit to 1 for speed)
    results = collection.get(limit=1)
    
    assert results is not None, "Failed to retrieve documents"
    assert "ids" in results, "Results missing 'ids' field"
    assert len(results["ids"]) > 0, "No documents found in collection"


def test_retrieved_metadata_contains_required_fields(chroma_client):
    """Test that retrieved metadata contains complaint_id, product_category, and issue."""
    collections = chroma_client.list_collections()
    
    assert len(collections) > 0, "No collections found in vector store"
    
    # Get the first collection
    collection = collections[0]
    
    # Retrieve one document
    results = collection.get(limit=1)
    
    assert "metadatas" in results, "Results missing 'metadatas' field"
    assert len(results["metadatas"]) > 0, "No metadata found"
    
    metadata = results["metadatas"][0]
    
    required_fields = ["complaint_id", "product_category", "issue"]
    
    for field in required_fields:
        assert field in metadata, \
            f"Required field '{field}' not found in metadata. Available fields: {list(metadata.keys())}"
