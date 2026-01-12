# RAG System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a RAG (Retrieval-Augmented Generation) system for the `taochiang` agent to access legal knowledge from "Marriage Law" documents using Volcengine embeddings.

**Architecture:** A custom, lightweight RAG pipeline. Offline indexing chunking text and storing vectors in JSON. Runtime retrieval using cosine similarity triggered by keywords in agent's thought process.

**Tech Stack:** Python, Numpy (vector math), Volcengine API (Embeddings), JSON (Storage).

---

### Task 1: Create Legal Data Directory and File

**Files:**
- Create: `reverie/backend_server/rag/data/marriage_law.txt`

**Step 1: Create directory**

```bash
mkdir -p reverie/backend_server/rag/data
```

**Step 2: Create legal text file**

```bash
cat <<EOF > reverie/backend_server/rag/data/marriage_law.txt
第五编　婚姻家庭
第一章　一般规定
第一千零四十条　本编调整因婚姻家庭产生的民事关系。
第一千零四十一条　婚姻家庭受国家保护。
实行婚姻自由、一夫一妻、男女平等的婚姻制度。
保护妇女、未成年人、老年人、残疾人的合法权益。
第一千零四十二条　禁止包办、买卖婚姻和其他干涉婚姻自由的行为。禁止借婚姻索取财物。
禁止重婚。禁止有配偶者与他人同居。
禁止家庭暴力。禁止家庭成员间的虐待和遗弃。
第一千零四十三条　家庭应当树立优良家风，弘扬家庭美德，重视家庭文明建设。
夫妻应当互相忠实，互相尊重，互相关爱；家庭成员应当敬老爱幼，互相帮助，维护平等、和睦、文明的婚姻家庭关系。
第一千零四十六条　结婚应当男女双方完全自愿，禁止任何一方对另一方加以强迫，禁止任何组织或者个人加以干涉。
第一千零四十七条　结婚年龄，男不得早于二十二周岁，女不得早于二十周岁。
第一千零六十二条　夫妻在婚姻关系存续期间所得的下列财产，为夫妻的共同财产，归夫妻共同所有：
（一）工资、奖金、劳务报酬；
（二）生产、经营、投资的收益；
（三）知识产权的收益；
（四）继承或者受赠的财产，但是本法第一千零六十三条第三项规定的除外；
（五）其他应当归共同所有的财产。
夫妻对共同财产，有平等的处理权。
第一千零七十六条　夫妻双方自愿离婚的，应当签订书面离婚协议，并亲自到婚姻登记机关申请离婚登记。
第一千零七十九条　夫妻一方要求离婚的，可以由有关组织进行调解或者直接向人民法院提起离婚诉讼。
人民法院审理离婚案件，应当进行调解；如果感情确已破裂，调解无效的，应当准予离婚。
EOF
```

**Step 3: Commit**

```bash
git add reverie/backend_server/rag/data/marriage_law.txt
git commit -m "feat: add marriage law text data"
```

### Task 2: Implement Vector Store

**Files:**
- Create: `reverie/backend_server/rag/vector_store.py`
- Test: `reverie/backend_server/rag/tests/test_vector_store.py`

**Step 1: Create test directory**

```bash
mkdir -p reverie/backend_server/rag/tests
touch reverie/backend_server/rag/tests/__init__.py
```

**Step 2: Write the failing test**

```python
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
```

**Step 3: Run test to verify it fails**

Run: `python3 reverie/backend_server/rag/tests/test_vector_store.py`
Expected: FAIL (ModuleNotFoundError)

**Step 4: Write minimal implementation**

```python
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
```

**Step 5: Run test to verify it passes**

Run: `python3 reverie/backend_server/rag/tests/test_vector_store.py`
Expected: PASS

**Step 6: Commit**

```bash
git add reverie/backend_server/rag/vector_store.py reverie/backend_server/rag/tests/test_vector_store.py
git commit -m "feat: implement vector store for RAG"
```

### Task 3: Implement Indexer

**Files:**
- Create: `reverie/backend_server/rag/indexer.py`
- Modify: `reverie/backend_server/rag/indexer.py` (Add embedding integration)
- Test: `reverie/backend_server/rag/tests/test_indexer.py`

**Step 1: Write the failing test**

```python
import unittest
from unittest.mock import patch, MagicMock
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
```

**Step 2: Run test to verify it fails**

Run: `python3 reverie/backend_server/rag/tests/test_indexer.py`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from reverie.backend_server.rag.chunker import chunk_file
from reverie.backend_server.rag.vector_store import VectorStore
# Import get_embedding from the existing util
from persona.prompt_template.gpt_structure import get_embedding

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
    indexer = Indexer("reverie/backend_server/rag/data")
    indexer.build_index("reverie/backend_server/rag/data/marriage_law.txt", "legal_index.json")
```

**Step 4: Run test to verify it passes**

Run: `python3 reverie/backend_server/rag/tests/test_indexer.py`
Expected: PASS

**Step 5: Commit**

```bash
git add reverie/backend_server/rag/indexer.py reverie/backend_server/rag/tests/test_indexer.py
git commit -m "feat: implement document indexer with embedding integration"
```

### Task 4: Implement Retriever

**Files:**
- Create: `reverie/backend_server/rag/retriever.py`
- Test: `reverie/backend_server/rag/tests/test_retriever.py`

**Step 1: Write the failing test**

```python
import unittest
from unittest.mock import patch, MagicMock
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
        self.assertIn("Apple", results[0]["text"])

if __name__ == '__main__':
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python3 reverie/backend_server/rag/tests/test_retriever.py`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

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
    results = retriever.retrieve("离婚时财产如何分割？")
    for r in results:
        print(f"[Score: {r['score']:.4f}] {r['text'][:50]}...")
```

**Step 4: Run test to verify it passes**

Run: `python3 reverie/backend_server/rag/tests/test_retriever.py`
Expected: PASS

**Step 5: Commit**

```bash
git add reverie/backend_server/rag/retriever.py reverie/backend_server/rag/tests/test_retriever.py
git commit -m "feat: implement retriever with cosine similarity"
```

### Task 5: Build Index and Run Verification Demo

**Files:**
- Run: `reverie/backend_server/rag/indexer.py`
- Run: `reverie/backend_server/rag/retriever.py`

**Step 1: Build the real index**

```bash
python3 reverie/backend_server/rag/indexer.py
```

**Step 2: Verify retrieval works**

```bash
python3 reverie/backend_server/rag/retriever.py
```

**Step 3: Capture output for verification**

Ensure the output shows relevant legal text for the query.

### Task 6: RAG Agent Interface

**Files:**
- Create: `reverie/backend_server/rag/rag_interface.py`

**Step 1: Create interface class**

```python
from reverie.backend_server.rag.retriever import Retriever

class RAGSystem:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            # Hardcoded paths for simplicity in this demo
            storage_path = "reverie/backend_server/rag/data"
            index_name = "legal_index.json"
            cls._instance = Retriever(storage_path, index_name)
        return cls._instance

    @staticmethod
    def query(text: str, k: int = 3):
        retriever = RAGSystem.get_instance()
        return retriever.retrieve(text, k)
```

**Step 2: Commit**

```bash
git add reverie/backend_server/rag/rag_interface.py
git commit -m "feat: add singleton interface for agent access"
```

### Task 7: Integrate with Agent (Keyword Trigger)

**Files:**
- Modify: `reverie/backend_server/persona/persona.py`

**Step 1: Locate `plan` method in Persona**

We'll add a check before planning actions.

**Step 2: Add trigger logic**

Modify `reverie/backend_server/persona/persona.py`:

```python
# Import at top
from rag.rag_interface import RAGSystem

# In Persona class
def check_legal_context(self, current_thought):
    keywords = ["婚姻", "离婚", "财产", "抚养", "收养", "夫妻", "子女"]
    for kw in keywords:
        if kw in current_thought:
            print(f"[RAG Triggered] Found keyword: {kw}")
            results = RAGSystem.query(current_thought, k=2)
            context = "\n".join([f"- {r['text']}" for r in results])
            return context
    return None
```

> **Note:** The exact integration point depends on where `taochiang` makes decisions. We will assume for now we just add the capability method to `Persona`, and the user (you) will invoke it manually or we hook it into `plan.py`. For this plan, adding the method is the goal.

**Step 3: Commit**

```bash
git add reverie/backend_server/persona/persona.py
git commit -m "feat: add RAG capability to Persona class"
```

