from rank_bm25 import BM25Okapi
from data.load_documents import load_documents

# load dataset
df = load_documents()
documents = df["text"].tolist()

# tokenize
tokenized_docs = [doc.lower().split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)

def search(query, top_k=5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        list(zip(documents, scores)),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]


if __name__ == "__main__":
    results = search("learn python")
    for r in results:
        print(r)