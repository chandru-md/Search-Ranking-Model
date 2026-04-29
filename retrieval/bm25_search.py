from rank_bm25 import BM25Okapi

# sample documents
documents = [
    "learn machine learning and AI",
    "python programming basics",
    "deep learning neural networks"
]

tokenized_docs = [doc.split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)

def search(query, top_k=3):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        list(zip(documents, scores)),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]


# test
results = search("learn python")
for r in results:
    print(r)