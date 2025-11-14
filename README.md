````markdown
# Varma's Agent — Personal Assistant with RAG (starter)

This scaffold provides:
- FastAPI WebSocket streaming chat for real-time responses.
- Retrieval-augmented generation (RAG) with OpenAI embeddings and FAISS vector store.
- HTTP fetch + HTML extraction tool for ingesting web pages (no JavaScript execution).
- Data manager to ingest, chunk, embed, and index sources, and to reindex/update them.
- Endpoints to manage data sources: /ingest-url, /sources, /reindex (admin only).

Quickstart (dev)
1. Copy `.env.example` -> `.env` and fill OPENAI_API_KEY, VARMA_AGENT_API_KEY and VARMA_ADMIN_KEY.
2. python -m venv .venv && source .venv/bin/activate
3. pip install -r requirements.txt
4. Start server:
   uvicorn app.main:app --reload
5. Connect a WebSocket client:
   ws://localhost:8000/ws/chat?api_key=<VARMA_AGENT_API_KEY>
   Send JSON frames like {"message":"summarize https://example.com"}.

Ingesting sources
- Admin POST /ingest-url (header X-ADMIN-KEY) with JSON {"url":"https://..."} will start ingestion in background.
- The agent also supports a simple user-driven ingest command: messages beginning with "ingest <url>" or "remember <url>" trigger immediate ingestion (runs sync in background; for heavy loads use admin endpoint).

Design notes
- Embeddings: uses OpenAI embeddings by default (configure OPENAI_EMBEDDING_MODEL).
- Vector store: FAISS IndexFlatIP with normalized vectors for cosine similarity.
- Grounded generation: top-k snippets are included in the prompt as system-context and the model is instructed to cite URLs.
- Performance: ingestion/embedding is done in background threads to avoid blocking the main event loop. Tune CHUNK_SIZE, TOP_K, and MAX_CONTEXT_CHARS for latency/accuracy tradeoffs.

Security
- Current auth is demonstration-only (simple API keys). Replace with OAuth2/JWT, per-user sessions, and strict rate limits / quotas before production.
- Carefully sanitize and whitelist domains for ingestion in production to avoid indexing malicious content.

Next steps
- Add incremental updates instead of naive reindex_all to avoid duplicate content.
- Store chunk text outside metadata (on-disk) and only persist pointers in metadata to reduce memory use.
- Add per-user conversation history and encrypted storage.
- Add observability: logs, metrics, and usage quotas.
````