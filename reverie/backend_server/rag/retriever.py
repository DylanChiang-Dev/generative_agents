import sys
import os
import numpy as np

# Add backend_server directory to sys.path to allow importing persona
curr_dir = os.path.dirname(__file__)
backend_server_dir = os.path.abspath(os.path.join(curr_dir, '../'))
sys.path.append(backend_server_dir)

from reverie.backend_server.rag.vector_store import VectorStore
from persona.prompt_template.gpt_structure import get_embedding

class Retriever:
    def __init__(self, storage_path: str, index_name: str):
        self.store = VectorStore(storage_path)
        self.store.load(index_name)

    def retrieve(self, query: str, k: int = 3):
        """
        Retrieve top-k relevant documents for the query.
        """
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []

        # Calculate cosine similarity
        query_vec = np.array(query_embedding)
        norm_query = np.linalg.norm(query_vec)

        scored_docs = []
        for doc in self.store.documents:
            doc_vec = np.array(doc["embedding"])
            norm_doc = np.linalg.norm(doc_vec)

            if norm_query == 0 or norm_doc == 0:
                score = 0
            else:
                score = np.dot(query_vec, doc_vec) / (norm_query * norm_doc)

            scored_docs.append((score, doc))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Return top k docs (excluding embedding to save space)
        results = []
        for score, doc in scored_docs[:k]:
            result = doc.copy()
            if "embedding" in result:
                del result["embedding"]
            result["score"] = float(score)
            results.append(result)

        return results

if __name__ == "__main__":
    # Demo
    retriever = Retriever("reverie/backend_server/rag/data", "legal_index.json")
    try:
        results = retriever.retrieve("离婚时财产如何分割？")
        for r in results:
            print(f"[Score: {r['score']:.4f}] {r['text'][:50]}...")
    except Exception as e:
        print(f"Error running demo: {e}")
