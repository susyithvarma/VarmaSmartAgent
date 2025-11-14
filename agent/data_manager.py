"""
Data manager responsible for ingesting URLs / raw texts, chunking content,
computing embeddings, and adding to the vector store.
Also exposes simple reindex/update operations for new/changed sources.
"""

import os
import uuid
import json
from typing import List, Dict
from agent.tools import fetch_url_text
from agent.embeddings import embed_texts
from agent.vector_store import VectorStore, DATA_DIR
from concurrent.futures import ThreadPoolExecutor

# chunking config
CHUNK_SIZE = 800  # characters per chunk approx (tune as needed)
CHUNK_OVERLAP = 200

# persistent sources map
SOURCES_PATH = os.path.join(DATA_DIR, "sources.json")
executor = ThreadPoolExecutor(max_workers=2)

class DataManager:
    def __init__(self, embed_dim: int = 1536):
        # embed_dim should match the embedding model dimension (OpenAI's embedding dims vary by model).
        self.embed_dim = embed_dim
        self.vs = VectorStore(dim=embed_dim)
        self._load_sources()

    def _load_sources(self):
        if os.path.exists(SOURCES_PATH):
            try:
                with open(SOURCES_PATH, "r", encoding="utf-8") as f:
                    self.sources = json.load(f)
            except Exception:
                self.sources = {}
        else:
            self.sources = {}

    def _persist_sources(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(SOURCES_PATH, "w", encoding="utf-8") as f:
            json.dump(self.sources, f, indent=2, ensure_ascii=False)

    def _chunk_text(self, text: str) -> List[str]:
        """
        Simple character-based chunker with overlap.
        """
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + CHUNK_SIZE, length)
            chunk = text[start:end]
            chunks.append(chunk)
            start = max(end - CHUNK_OVERLAP, end)
        return chunks

    def ingest_url(self, url: str, source_name: str = None):
        """
        Fetches URL, chunks, embeds, adds to vector store and records the source entry.
        Runs synchronously; you can call this in a background task.
        Returns the source_id.
        """
        title, text = fetch_url_text(url)
        source_id = str(uuid.uuid4())
        chunks = self._chunk_text(text)
        metas = []
        for i, chunk in enumerate(chunks):
            meta = {
                "id": f"{source_id}::{i}",
                "source_id": source_id,
                "source_url": url,
                "source_title": title,
                "chunk_index": i,
                "chunk_text": chunk,
            }
            metas.append(meta)
        # embed
        vectors = embed_texts(chunks)
        # ensure dimension correct
        self.vs.add(vectors, metas)
        # record source
        self.sources[source_id] = {
            "id": source_id,
            "url": url,
            "title": title,
            "name": source_name or title or url,
            "chunks": len(chunks)
        }
        self._persist_sources()
        return source_id

    def ingest_url_async(self, url: str, source_name: str = None):
        """
        Submit ingestion to thread-pool for background processing.
        Returns a future.
        """
        return executor.submit(self.ingest_url, url, source_name)

    def search(self, query: str, k: int = 5):
        """
        Embeds the query and searches the vector store.
        Returns list of (metadata, score).
        """
        q_vec = embed_texts([query])[0]
        return self.vs.search(q_vec, k=k)

    def list_sources(self):
        return list(self.sources.values())

    def reindex_all(self):
        """
        Re-ingest every source. Simple naive approach: clear index and re-ingest.
        Production: do incremental or update-by-diff to avoid duplicates.
        """
        all_sources = list(self.sources.values())
        # clear store
        self.vs.clear()
        for s in all_sources:
            self.ingest_url(s["url"], source_name=s.get("name"))
        return len(all_sources)