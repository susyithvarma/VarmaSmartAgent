"""
Core agent updated to perform retrieval-augmented generation (RAG) with:
- DataManager retriever for top-k snippets
- Grounded prompt assembly (with citations)
- Streaming and non-streaming responses using OpenAI chat with streaming
- Simple tool invocation pattern: if the user requests ingestion, call DataManager.ingest_url_async
"""

import os
import openai
import asyncio
import functools
from typing import AsyncGenerator, List
from dotenv import load_dotenv
from agent.data_manager import DataManager

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are Varma's personal assistant. Be concise and helpful. When you use retrieved documents, always cite the URL in the response and mark the text as [source]."""

# RAG config
TOP_K = 4
EMBED_DIM = 1536  # set to the dimension of your embedding model; adjust if needed
MAX_CONTEXT_CHARS = 4000  # cap the context included to avoid too big prompts

# instantiate DataManager once

data_manager = DataManager(embed_dim=EMBED_DIM)

class AIAgent:
    def __init__(self):
        self.system = SYSTEM_PROMPT

    async def _call_openai_stream(self, messages):
        """
        Async wrapper to call OpenAI ChatCompletion with stream=True and yield tokens.
        """
        loop = asyncio.get_event_loop()
        def blocking():
            return openai.ChatCompletion.create(model=MODEL, messages=messages, stream=True, temperature=0.2, max_tokens=800)
        resp_iter = await loop.run_in_executor(None, blocking)
        for chunk in resp_iter:
            for c in chunk.get("choices", []):
                delta = c.get("delta", {})
                text = delta.get("content")
                if text:
                    yield text

    async def _call_openai(self, messages):
        """
        Non-streaming call.
        """
        loop = asyncio.get_event_loop()
        def blocking():
            return openai.ChatCompletion.create(model=MODEL, messages=messages, max_tokens=800, temperature=0.2)
        resp = await loop.run_in_executor(None, blocking)
        return resp["choices"][0]["message"]["content"]

    def _assemble_rag_prompt(self, user_message: str, retrieved: List[dict]) -> List[dict]:
        """
        Build messages list with system prompt, retrieved contexts, and user query.
        Each retrieved item is included with a short citation.
        """
        system_msgs = [{"role": "system", "content": self.system}]
        if retrieved:
            # include top-K snippets up to a character budget
            ctxs = []
            chars = 0
            for meta, score in retrieved:
                snippet = meta.get("chunk_text") or meta.get("text") or ""
                if not snippet:
                    continue
                addition = f"[{meta.get('source_title','')}]({meta.get('source_url')})\n{snippet}\n---\n"
                chars += len(addition)
                if chars > MAX_CONTEXT_CHARS:
                    break
                ctxs.append(addition)
            if ctxs:
                ctx_content = "Use the following retrieved snippets to ground your answer. Cite the URL in your answer.\n\n" + "\n".join(ctxs)
                system_msgs.append({"role": "system", "content": ctx_content})
        system_msgs.append({"role": "user", "content": user_message})
        return system_msgs

    async def stream_reply(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        Stream reply with RAG. Detect quick ingestion commands and handle them.
        """
        # naive command handling: ingest command
        if user_message.strip().lower().startswith("ingest ") or user_message.strip().lower().startswith("remember "):
            # expect: "ingest https://..."
            parts = user_message.split()
            url = next((p for p in parts if p.startswith("http")), None)
            if url:
                # run ingestion in background
                loop = asyncio.get_event_loop()
                future = loop.run_in_executor(None, functools.partial(data_manager.ingest_url, url))
                source_id = await future
                yield f"[INFO] Ingested {url} as source {source_id}\n"
                yield "[DONE]"
                return

        # run retriever
        retrieved = data_manager.search(user_message, k=TOP_K)
        messages = self._assemble_rag_prompt(user_message, retrieved)
        async for token in self._call_openai_stream(messages):
            yield token

    async def handle_message(self, user_message: str) -> str:
        """
        Non-streaming: perform RAG and return final text.
        """
        retrieved = data_manager.search(user_message, k=TOP_K)
        messages = self._assemble_rag_prompt(user_message, retrieved)
        try:
            resp = await self._call_openai(messages)
            return resp
        except Exception as e:
            return f"Error contacting OpenAI: {e}"
