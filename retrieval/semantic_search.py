from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "learn machine learning and AI",
    "python programming basics",
    "deep learning neural networks"
]

doc_embeddings = model.encode(documents)

def semantic_search(query, top_k=3):
    query_embedding = model.encode([query])[0]

    scores = np.dot(doc_embeddings, query_embedding)

    ranked = sorted(
        list(zip(documents, scores)),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]