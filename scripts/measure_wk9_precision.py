"""Measure precision@3 and precision@5 for the golden set.

Compares W9 (hybrid+rerank) vs W8 baseline (pure dense).
"""
import json
import sqlite3
from pathlib import Path

from src.rag.retrieval import retrieve, dense_search

GOLDEN_SET = Path("data/golden_set.jsonl")
DB_PATH = Path("results.db")

# Load golden set: {question, expected_chunk_ids: [...]}
golden = [json.loads(line) for line in GOLDEN_SET.read_text().splitlines() if line]
print(f"Loaded {len(golden)} golden questions.")

def precision_at_k(retrieved_ids, expected_ids, k):
    """1.0 if any expected chunk_id is in top-k retrieved; else 0.0.
    (Programme convention: single-answer questions; extend to multi-answer if needed.)"""
    top_k = retrieved_ids[:k]
    return 1.0 if any(eid in top_k for eid in expected_ids) else 0.0

def run_and_score(retriever_fn, label):
    p3_scores, p5_scores = [], []
    for q in golden:
        hits = retriever_fn(q["question"], k=5)
        retrieved_ids = [h["doc"].get("chunk_id") for h in hits]
        p3_scores.append(precision_at_k(retrieved_ids, q["expected_chunk_ids"], 3))
        p5_scores.append(precision_at_k(retrieved_ids, q["expected_chunk_ids"], 5))
    return {
        "label": label,
        "p_at_3": sum(p3_scores) / len(p3_scores),
        "p_at_5": sum(p5_scores) / len(p5_scores),
        "per_q_p3": p3_scores,
        "per_q_p5": p5_scores,
    }

# Baseline: pure dense (what W8 used)
baseline = run_and_score(lambda q, k: dense_search(q, k), "W8 dense-only")

# New: full W9 hybrid + rerank stack
v9 = run_and_score(retrieve, "W9 hybrid+rerank")

print(f"\n══ Precision comparison ══")
print(f"  {'Metric':<10s} {'W8 baseline':>12s} {'W9 hybrid+rerank':>18s} {'Delta':>10s}")
print(f"  {'------':<10s} {'-----------':>12s} {'-----------------':>18s} {'-----':>10s}")
for m in ["p_at_3", "p_at_5"]:
    delta = v9[m] - baseline[m]
    print(f"  {m:<10s} {baseline[m]:>12.3f} {v9[m]:>18.3f} {delta:>+10.3f}")

# Save per-question delta to docs/wk9-precision-delta.md
out = Path("docs/wk9-precision-delta.md")
lines = ["# W9 Precision Delta — per-question breakdown\n"]
lines.append(f"| Q# | Question | W8 P@3 | W9 P@3 | Change |")
lines.append(f"|---|---|---|---|---|")
for i, (q, p8_3, p9_3) in enumerate(zip(golden, baseline["per_q_p3"], v9["per_q_p3"])):
    change = "→ FIXED" if p9_3 > p8_3 else ("→ BROKE" if p9_3 < p8_3 else "same")
    lines.append(f"| {i+1} | {q['question'][:60]}... | {p8_3:.0f} | {p9_3:.0f} | {change} |")
out.write_text("\n".join(lines))
print(f"\nSaved per-question delta to {out}")
