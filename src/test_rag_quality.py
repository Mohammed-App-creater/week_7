
import unittest
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline import RAGPipeline

class TestRAGQuality(unittest.TestCase):
    def setUp(self):
        # Initialize pipeline without loading heavy models
        self.pipeline = RAGPipeline(
            vector_store_path="dummy",
            embedding_model="dummy",
            llm_model="dummy"
        )
        
    def test_product_filtering(self):
        """Test that product filtering keeps only relevant documents."""
        print("\nTest: Product Filtering")
        
        # Mock documents with different products
        documents = [
            {'metadata': {'product': 'Credit card'}, 'text': 'Card issue'},
            {'metadata': {'product': 'Mortgage'}, 'text': 'Loan issue'},
            {'metadata': {'product': 'Bank account or service'}, 'text': 'Account issue'},
            {'metadata': {'product': 'Credit reporting'}, 'text': 'Report issue'}
        ]
        
        # Case 1: "credit card" query -> Should keep Credit card
        filtered_cc = self.pipeline.filter_by_product_category("issues with credit card", documents)
        self.assertEqual(len(filtered_cc), 1)
        self.assertEqual(filtered_cc[0]['metadata']['product'], 'Credit card')
        print("✓ Credit card filtering passed")
        
        # Case 2: "mortgage" query -> Should keep Mortgage
        filtered_mort = self.pipeline.filter_by_product_category("mortgage rates", documents)
        self.assertEqual(len(filtered_mort), 1)
        self.assertEqual(filtered_mort[0]['metadata']['product'], 'Mortgage')
        print("✓ Mortgage filtering passed")
        
        # Case 3: Ambiguous query -> Should keep all
        filtered_all = self.pipeline.filter_by_product_category("general complaint", documents)
        self.assertEqual(len(filtered_all), 4)
        print("✓ Fallback (no keyword) passed")
        
    def test_context_cleaning(self):
        """Test that boilerplate is removed from context."""
        print("\nTest: Context Cleaning")
        
        raw_text = """
        I am writing to complain about my bill.
        I visited https://example.com and found errors.
        
        Sincerely,
        John Doe
        
        Sent from my iPhone
        """
        
        cleaned = self.pipeline.clean_complaint_text(raw_text)
        
        # Check signatures are gone
        self.assertNotIn("Sincerely", cleaned)
        self.assertNotIn("John Doe", cleaned)
        
        # Check URL replacement
        self.assertIn("[URL]", cleaned)
        self.assertNotIn("https://example.com", cleaned)
        
        # Check core text remains
        self.assertIn("complain about my bill", cleaned)
        
        print(f"✓ Cleaning passed. Result: '{cleaned}'")

if __name__ == '__main__':
    unittest.main()
