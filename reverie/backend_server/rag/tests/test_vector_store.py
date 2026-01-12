import unittest
import os
import json
import shutil
from reverie.backend_server.rag.vector_store import VectorStore

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = "reverie/backend_server/rag/tests/temp_data"
        os.makedirs(self.test_dir, exist_ok=True)
        self.store = VectorStore(storage_path=self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_add_and_save(self):
        self.store.add_documents([
            {"text": "doc1", "embedding": [0.1, 0.2]},
            {"text": "doc2", "embedding": [0.3, 0.4]}
        ])
        self.store.save("index.json")

        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "index.json")))

        # Load and verify
        new_store = VectorStore(storage_path=self.test_dir)
        new_store.load("index.json")
        self.assertEqual(len(new_store.documents), 2)
        self.assertEqual(new_store.documents[0]["text"], "doc1")
        self.assertEqual(new_store.documents[0]["embedding"], [0.1, 0.2])

if __name__ == '__main__':
    unittest.main()
