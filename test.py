from src.rag.qdrant_rag import ask_rag
from src.rag import cache
cache.clear()  # start cold
r1 = ask_rag("what is the leave policy?")
print(f"Cold: hit={r1['cache_hit']}, answer={r1['answer'][:80]}")
r2 = ask_rag("what is the leave policy?")
print(f"Warm: hit={r2['cache_hit']}, similarity={r2.get('similarity', 'n/a')}")