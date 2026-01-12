import sys
import os

# Calculate absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_dir = current_dir
backend_server_dir = os.path.dirname(rag_dir)
reverie_dir = os.path.dirname(backend_server_dir)
root_dir = os.path.dirname(reverie_dir)

# Add root to sys.path to allow importing 'reverie' package
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Add backend_server to sys.path to allow importing 'persona' legacy modules if needed
if backend_server_dir not in sys.path:
    sys.path.append(backend_server_dir)

from reverie.backend_server.rag.chunker import chunk_file
from reverie.backend_server.rag.vector_store import VectorStore

# Import get_embedding from the existing util
try:
    from persona.prompt_template.gpt_structure import get_embedding
except ImportError:
    from reverie.backend_server.persona.prompt_template.gpt_structure import get_embedding

class Indexer:
    def __init__(self, storage_path: str):
        self.store = VectorStore(storage_path)

    def build_index(self, source_file: str, index_name: str):
        """
        1. Chunk the file
        2. Get embeddings for each chunk
        3. Store in VectorStore
        4. Save to disk
        """
        print(f"Chunking {source_file}...")
        # Resolve absolute path for source file if it's relative
        if not os.path.isabs(source_file):
            source_file = os.path.join(root_dir, source_file)
            
        chunks = chunk_file(source_file, chunk_size=512)

        processed_docs = []
        print(f"Generating embeddings for {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk["text"])
            if embedding:
                doc = chunk.copy()
                doc["embedding"] = embedding
                processed_docs.append(doc)

            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(chunks)} chunks")

        self.store.add_documents(processed_docs)
        self.store.save(index_name)
        print(f"Index saved to {index_name}")

if __name__ == "__main__":
    # Demo usage
    # Ensure data directory exists
    data_dir = os.path.join(rag_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    indexer = Indexer(data_dir)
    
    # Check if the source file exists
    source_path = os.path.join(data_dir, "marriage_law.txt")
    if not os.path.exists(source_path):
        print(f"Error: Source file not found at {source_path}")
        # Try relative path from root for the input
        input_path = "reverie/backend_server/rag/data/marriage_law.txt"
    else:
        input_path = source_path
        
    indexer.build_index(input_path, "legal_index.json")
