import json
import os
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.documents: List[Dict[str, Any]] = []

    def add_documents(self, docs: List[Dict[str, Any]]):
        """
        Add documents to the store.
        Each doc should have 'text' and 'embedding' keys.
        """
        self.documents.extend(docs)

    def save(self, filename: str):
        """Save documents and embeddings to a JSON file."""
        file_path = os.path.join(self.storage_path, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def load(self, filename: str):
        """Load documents from a JSON file."""
        file_path = os.path.join(self.storage_path, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
