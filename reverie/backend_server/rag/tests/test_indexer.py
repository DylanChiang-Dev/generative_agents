import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add path to find modules
sys.path.append(os.getcwd())

from reverie.backend_server.rag.indexer import Indexer

class TestIndexer(unittest.TestCase):
    @patch('reverie.backend_server.rag.indexer.get_embedding')
    @patch('reverie.backend_server.rag.indexer.chunk_file')
    @patch('reverie.backend_server.rag.indexer.VectorStore')
    def test_build_index(self, mock_store_cls, mock_chunk, mock_embed):
        # Setup mocks
        mock_chunk.return_value = [{"text": "chunk1", "source": "file.txt"}]
        mock_embed.return_value = [0.1, 0.2, 0.3]

        mock_store_instance = MagicMock()
        mock_store_cls.return_value = mock_store_instance

        # Run
        indexer = Indexer(storage_path="./")
        indexer.build_index("dummy_path.txt", "index.json")

        # Verify
        mock_chunk.assert_called_once()
        mock_embed.assert_called_once_with("chunk1")
        mock_store_instance.add_documents.assert_called_once()
        mock_store_instance.save.assert_called_once_with("index.json")

if __name__ == '__main__':
    unittest.main()
