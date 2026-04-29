from retrieval.bm25_search import search as bm25_search
from retrieval.semantic_search import semantic_search

def hybrid_search(query):
    bm25_results = bm25_search(query, top_k=5)
    semantic_results = semantic_search(query, top_k=5)

    combined = bm25_results + semantic_results

    # simple merge
    unique = list({doc: score for doc, score in combined}.items())

    ranked = sorted(unique, key=lambda x: x[1], reverse=True)

    return ranked[:5]


# test
results = hybrid_search("learn AI")
for r in results:
    print(r)