"""SQLite-backed semantic cache for the capstone.

Layer 1 of the 3-layer caching stack (Layer 2 = OpenAI prompt cache
auto-fires ≥1024 tokens; Layer 3 = embedding cache from W7).
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
import struct
import time
from pathlib import Path

CACHE_DB_DEFAULT = "results.db"   # reuse the file W2 created
DEFAULT_THRESHOLD = 0.95           # deck default; do not lower without measuring


def _connect(db_path: str = CACHE_DB_DEFAULT) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
            id             TEXT PRIMARY KEY,
            question       TEXT NOT NULL,
            embedding_blob BLOB NOT NULL,
            answer         TEXT NOT NULL,
            hit_count      INTEGER NOT NULL DEFAULT 0,
            created_ts     REAL NOT NULL,
            last_hit_ts    REAL
        )
    """)
    conn.commit()
    print("neleesh connecting to sqlite")
    return conn


def _key(question: str) -> str:
    return hashlib.sha256(question.lower().strip().encode()).hexdigest()[:16]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na  = math.sqrt(sum(x*x for x in a))
    nb  = math.sqrt(sum(y*y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def lookup(question_embedding: list[float],
           threshold: float = DEFAULT_THRESHOLD,
           db_path: str = CACHE_DB_DEFAULT) -> dict | None:
    """Return {answer, similarity, matched_question} on hit; None on miss."""
    conn = _connect(db_path)
    try:
        best_score, best = -1.0, None
        for row in conn.execute(
            "SELECT id, question, embedding_blob, answer FROM query_cache"
        ):
            row_id, row_q, blob, ans = row
            n = len(blob) // 4
            emb = list(struct.unpack(f"{n}f", blob))
            s = _cosine(question_embedding, emb)
            if s > best_score:
                best_score, best = s, (row_id, row_q, ans)
        if best is None or best_score < threshold:
            return None
        row_id, matched_q, cached_ans = best
        conn.execute(
            "UPDATE query_cache SET hit_count=hit_count+1, last_hit_ts=? WHERE id=?",
            (time.time(), row_id),
        )
        conn.commit()
        return {
            "answer":            cached_ans,
            "matched_question":  matched_q,
            "similarity":        best_score,
        }
    finally:
        conn.close()


def put(question: str, embedding: list[float], answer: str,
        db_path: str = CACHE_DB_DEFAULT) -> None:
    conn = _connect(db_path)
    try:
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        conn.execute("""
            INSERT OR REPLACE INTO query_cache
            (id, question, embedding_blob, answer, hit_count, created_ts, last_hit_ts)
            VALUES (?, ?, ?, ?, 0, ?, NULL)
        """, (_key(question), question, blob, answer, time.time()))
        conn.commit()
    finally:
        conn.close()


def clear(db_path: str = CACHE_DB_DEFAULT) -> int:
    """Wipe all cache entries. Call this after any document update."""
    conn = _connect(db_path)
    try:
        cur = conn.execute("DELETE FROM query_cache")
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def stats(db_path: str = CACHE_DB_DEFAULT) -> dict:
    """For dashboards: total entries, total hits, average similarity of hits."""
    conn = _connect(db_path)
    try:
        row = conn.execute("""
            SELECT COUNT(*), COALESCE(SUM(hit_count), 0)
            FROM query_cache
        """).fetchone()
        return {"n_entries": row[0], "total_hits": row[1]}
    finally:
        conn.close()
"""```

**Wire into `qdrant_rag.py`** — cache-first pattern:

```python"""
# In src/rag/qdrant_rag.py — REPLACE ask_rag()

from openai import OpenAI
from src.rag.retrieval import retrieve      # from W9
from src.rag import cache                    # NEW W10

_openai = None
def _get_openai():
    global _openai
    if _openai is None:
        _openai = OpenAI()
    return _openai


def ask_rag(question: str, k: int = 3,
            cache_threshold: float = 0.95) -> dict:
    """Full RAG with 3-layer caching.

    Layer 1 (semantic cache) checked first — return early on hit.
    Layer 2 (OpenAI prompt cache) fires automatically on the LLM call.
    Layer 3 (embedding cache) transparent from W7.
    """
    client = _get_openai()

    # ── Layer 1: semantic cache ──
    q_vec = client.embeddings.create(
        model="text-embedding-3-small", input=[question]
    ).data[0].embedding

    hit = cache.lookup(q_vec, threshold=cache_threshold)
    if hit:
        return {
            "answer":     hit["answer"],
            "cache_hit":  True,
            "similarity": hit["similarity"],
            "matched":    hit["matched_question"],
        }

    # ── Miss: full RAG pipeline ──
    retrieved = retrieve(question, k=k)  # from W9: hybrid + rerank
    prompt = _build_prompt(question, retrieved)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = resp.choices[0].message.content

    # Store in cache for next time
    cache.put(question, q_vec, answer)

    return {
        "answer":     answer,
        "cache_hit":  False,
        "n_input_tokens":         resp.usage.prompt_tokens,
        "n_output_tokens":        resp.usage.completion_tokens,
        "n_cached_input_tokens":  getattr(resp.usage, 'prompt_tokens_details', None)
                                   and resp.usage.prompt_tokens_details.cached_tokens or 0,
        "retrieved":              retrieved,
    }


def _build_prompt(question: str, retrieved: list[dict]) -> str:
    """Stable prefix FIRST (cacheable), question + context LAST."""
    system = ("You are a helpful assistant. Answer using ONLY the context. "
              "If the context does not contain the answer, say 'I don't know'.")
    ctx = "\n\n".join(f"[{i+1}] {h['doc'].get('text', '')}"
                       for i, h in enumerate(retrieved))
    return f"{system}\n\n=== Context ===\n{ctx}\n\n=== Question ===\n{question}\n\n=== Answer ==="