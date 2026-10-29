"""Hybrid retrieval + cross-encoder reranking for the capstone.

Replaces W8's pure-dense retrieval with:
  1. Hybrid retrieval (BM25 + Dense via RRF) - top-10
  2. Cross-encoder reranking - top-3

Public API:
  hybrid_retrieve(query, k=10) -> list[dict]
  rerank(query, candidates, top_k=3) -> list[dict]
  retrieve(query, k=3) -> list[dict]   # full stack, use this from qdrant_rag.py
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

from openai import OpenAI
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from src.rag.qdrant_store import _get_client, COLLECTION_NAME

# Cross-encoder lazy-init (heavy)
_reranker = None
def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

_openai = None
def _get_openai():
    global _openai
    if _openai is None:
        _openai = OpenAI()
    return _openai


def simple_tokenize(text: str) -> list[str]:
    """Match W9 Day 1 tokenization: keeps hyphens/slashes in tokens.

    'AC-1042' stays as one token so BM25 can catch exact IDs.
    """
    return re.findall(r'[a-z0-9][a-z0-9\-/_]*', text.lower())


@lru_cache(maxsize=1)
def _load_bm25():
    """Load ALL chunks from Qdrant, build BM25 index over their text.
    
    Cached — pays the cost once per process. For very large corpora
    (>10K chunks), persist this to disk instead. That's a W10 topic.
    """
    client = _get_client()
    all_points = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=200,
            offset=offset,
            with_payload=True,
        )
        all_points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
    
    # Preserve order — index positions map to point ids
    corpus_texts = [p.payload.get("text", "") for p in all_points]
    corpus_meta  = [{**p.payload, "point_id": p.id} for p in all_points]
    tokenized    = [simple_tokenize(t) for t in corpus_texts]
    return BM25Okapi(tokenized), corpus_meta


def bm25_search(query: str, k: int = 10) -> list[dict]:
    """BM25 retrieval — top-K by score. Returns list of dicts with 'point_id'."""
    bm25, meta = _load_bm25()
    scores = bm25.get_scores(simple_tokenize(query))
    ranked = sorted(zip(scores, meta), key=lambda p: p[0], reverse=True)
    return [
        {"id": str(m["point_id"]), "score": float(s), "doc": m}
        for s, m in ranked[:k]
    ]


def dense_search(query: str, k: int = 10) -> list[dict]:
    """Dense retrieval — Qdrant cosine top-K."""
    openai = _get_openai()
    client = _get_client()
    q_vec = openai.embeddings.create(
        model="text-embedding-3-small", input=[query]).data[0].embedding
    hits = client.query_points(
        collection_name=COLLECTION_NAME, query=q_vec, limit=k).points
    return [
        {"id": str(h.id), "score": h.score, "doc": {**h.payload, "point_id": h.id}}
        for h in hits
    ]


def rrf_fuse(ranked_lists: list[list[dict]], k: int = 60, top_n: int = 10) -> list[dict]:
    """Reciprocal Rank Fusion — same as W9 Day 1."""
    scores = {}
    docs = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            doc_id = hit["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in docs:
                docs[doc_id] = hit
    fused = sorted(scores.items(), key=lambda p: p[1], reverse=True)[:top_n]
    return [{**docs[doc_id], "rrf_score": s} for doc_id, s in fused]


def hybrid_retrieve(query: str, k_per_retriever: int = 10, k_final: int = 10) -> list[dict]:
    """BM25 + Dense + RRF combined."""
    bm25_hits  = bm25_search(query, k=k_per_retriever)
    dense_hits = dense_search(query, k=k_per_retriever)
    return rrf_fuse([bm25_hits, dense_hits], k=60, top_n=k_final)


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """Cross-encoder reranking — score (query, candidate.text) pairs."""
    reranker = _get_reranker()
    pairs = [(query, c["doc"].get("text", "")) for c in candidates]
    scores = reranker.predict(pairs)
    scored = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)]
    scored.sort(key=lambda h: h["rerank_score"], reverse=True)
    return scored[:top_k]


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Full W9 stack: hybrid → rerank → top-k. Use this from qdrant_rag.py."""
    candidates = hybrid_retrieve(query, k_per_retriever=10, k_final=10)
    return rerank(query, candidates, top_k=k)
