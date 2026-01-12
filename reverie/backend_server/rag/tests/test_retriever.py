import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np

# Add path to find modules
sys.path.append(os.getcwd())

from reverie.backend_server.rag.retriever import Retriever

class TestRetriever(unittest.TestCase):
    def setUp(self):
        # Create a mock store with some data
        self.mock_docs = [
            {"text": "Apple is a fruit", "embedding": [1.0, 0.0, 0.0]},
            {"text": "Car is a vehicle", "embedding": [0.0, 1.0, 0.0]},
            {"text": "Banana is yellow", "embedding": [0.9, 0.1, 0.0]}
        ]

    @patch('reverie.backend_server.rag.retriever.VectorStore')
    @patch('reverie.backend_server.rag.retriever.get_embedding')
    def test_retrieve(self, mock_embed, mock_store_cls):
        # Setup
        mock_store = MagicMock()
        mock_store.documents = self.mock_docs
        mock_store_cls.return_value = mock_store

        # Mock embedding for query "fruit" -> closer to [1,0,0]
        mock_embed.return_value = [0.95, 0.05, 0.0]

        retriever = Retriever("dummy_path", "index.json")
        results = retriever.retrieve("fruit", k=2)

        self.assertEqual(len(results), 2)
        # Should match Apple (high similarity) and Banana (high similarity)
        # Verify text content, not exact score
        self.assertIn("Apple", results[0]["text"])

if __name__ == '__main__':
    unittest.main()
