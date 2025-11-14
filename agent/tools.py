"""
Lightweight HTTP fetcher + HTML extractor using httpx + BeautifulSoup.
Used as a safe, headless page fetcher for RAG ingestion and short summaries.
"""

import httpx
from bs4 import BeautifulSoup
import logging
from typing import Tuple

logger = logging.getLogger("varma-agent-tools")
logger.setLevel(logging.INFO)

def fetch_url_text(url: str, timeout: int = 10) -> Tuple[str, str]:
    """
    Fetch URL via httpx (no JS), extract title and visible text.
    Returns (title, text) where text is trimmed to a reasonable length.
    This function intentionally avoids executing JS or fetching subresources.
    """
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "VarmaAgent/1.0"})
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        logger.exception("Error fetching URL")
        return "", f"[Error fetching {url}: {e}]"

    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    # prefer article or main elements
    main = soup.find("article") or soup.find("main")
    if main:
        text = main.get_text(separator="\n", strip=True)
    else:
        # fallback: extract paragraphs
        paragraphs = soup.find_all("p")
        text = "\n".join([p.get_text(strip=True) for p in paragraphs])
        if not text:
            text = soup.get_text(separator="\n", strip=True)
    # sanitize and trim
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    snippet = "\n".join(lines[:200])  # keep up to 200 lines (approx)
    if len(snippet) > 30000:
        snippet = snippet[:30000] + "..."
    return title, snippet
