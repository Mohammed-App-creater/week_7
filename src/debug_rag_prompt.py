
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline import RAGPipeline

def debug_prompt():
    pipeline = RAGPipeline(
        vector_store_path="dummy",
        embedding_model="dummy",
        llm_model="dummy"
    )
    
    # Mock documents
    mock_docs = [
        {
            'text': "I had a huge issue with my credit card bill. They charged me double interest. Sincerely, Bob.",
            'metadata': {'product': 'Credit card', 'issue': 'Billing dispute'}
        },
        {
            'text': "My experience was also bad. \n\nWe apologize for the inconvenience.\n\nRegards,\nBank Corp",
            'metadata': {'product': 'Credit card', 'issue': 'Customer service'}
        }
    ]
    
    question = "What are the common billing issues?"
    
    print("=" * 80)
    print("DEBUGGING PROMPT GENERATION")
    print("=" * 80)
    
    prompt = pipeline.create_prompt(question, mock_docs)
    
    print(prompt)
    print("=" * 80)

if __name__ == "__main__":
    debug_prompt()
