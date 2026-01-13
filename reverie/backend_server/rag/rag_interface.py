import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from .retriever import Retriever

class RAGSystem:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            # Determine paths relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            storage_path = os.path.join(base_dir, "data")
            index_name = "legal_index.json"
            
            # Check if index exists, if not warn
            index_path = os.path.join(storage_path, index_name)
            if not os.path.exists(index_path):
                print(f"[RAG Warning] Index not found at {index_path}")
                return None
                
            cls._instance = Retriever(storage_path, index_name)
        return cls._instance

    @staticmethod
    def query(text: str, k: int = 3):
        retriever = RAGSystem.get_instance()
        if retriever:
            return retriever.retrieve(text, k)
        return []
