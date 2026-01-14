import sys
import os
import json
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from .retriever import Retriever

class RAGSystem:
    _instance = None
    _log_filepath = None

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

    @classmethod
    def set_log_filepath(cls, filepath):
        """Set the file path for logging RAG interactions."""
        cls._log_filepath = filepath
        # Ensure directory exists
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

    @staticmethod
    def query(text: str, k: int = 3):
        retriever = RAGSystem.get_instance()
        results = []
        if retriever:
            results = retriever.retrieve(text, k)

        # Log the interaction if a log file is configured
        if RAGSystem._log_filepath:
            try:
                log_entry = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": text,
                    "results_count": len(results),
                    "results": results
                }
                with open(RAGSystem._log_filepath, "a", encoding='utf-8') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[RAG Logging Error] Could not write to log: {e}")

        return results
