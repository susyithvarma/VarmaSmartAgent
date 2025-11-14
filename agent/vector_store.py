"""
Simple FAISS-based vector store with metadata persistence.
- Stores vectors in FAISS IndexFlatIP (cosine via normalized vectors).
- Keeps metadata (id -> dict) persisted to disk as JSON or pickle (simple approach).
"""

import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Tuple

DATA_DIR = os.getenv("VARMA_DATA_DIR", "./varma_data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "meta.pkl")

class VectorStore:
    def __init__(self, dim: int):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.dim = dim
        # Use inner product on normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        # metadata: list of dicts aligned with index vectors
        self.metadata: List[Dict] = []
        # load if exists
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            try:
                self.index = faiss.read_index(INDEX_PATH)
                with open(META_PATH, "rb") as f:
                    self.metadata = pickle.load(f)
            except Exception:
                # fallback: re-create
                self.index = faiss.IndexFlatIP(dim)
                self.metadata = []

    def _persist(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

    def add(self, vectors: List[List[float]], metas: List[Dict]):
        """
        Add vectors and metadata. Ensure vectors are normalized for cosine similarity.
        """
        if not vectors:
            return
        arr = np.array(vectors).astype("float32")
        # normalize rows
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        arr = arr / norms
        self.index.add(arr)
        self.metadata.extend(metas)
        self._persist()

    def search(self, query_vector: List[float], k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search by single vector; returns list of (metadata, score).
        """  
        if self.index.ntotal == 0:
            return []
        q = np.array(query_vector).astype("float32")
        q = q.reshape(1, -1)
        # normalize
        q = q / (np.linalg.norm(q) + 1e-10)
        D, I = self.index.search(q, k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def count(self) -> int:
        return self.index.ntotal

    def clear(self):
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        self._persist()