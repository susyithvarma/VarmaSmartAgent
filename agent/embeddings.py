"""
Wrapper around OpenAI embeddings (or alternate model).
Exposes embed_texts(texts: list[str]) -> list[list[float]]
"""

import os
import openai
from typing import List

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Synchronous wrapper for embeddings. For bulk calls, OpenAI supports batching.
    Returns list of vectors (floats).
    """
    if not texts:
        return []
    # OpenAI's embeddings endpoint accepts up to certain size per call;
    # production: batch appropriately and handle rate limits / retries.
    resp = openai.Embedding.create(model=EMBED_MODEL, input=texts)
    vectors = [item["embedding"] for item in resp["data"]]
    return vectors
