import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def build_features(query, documents):
    query_embedding = model.encode([query])[0]
    doc_embeddings = model.encode(documents)

    similarity_scores = cosine_similarity(
        [query_embedding],
        doc_embeddings
    )[0]

    features = []

    for i, doc in enumerate(documents):
        doc_lower = doc.lower()
        query_lower = query.lower()

        features.append([
            similarity_scores[i],                    # semantic similarity
            len(doc.split()),                        # doc length
            doc_lower.count(query_lower),           # keyword match count
            1 if query_lower in doc_lower else 0,   # exact match
            len(set(query_lower.split()) & set(doc_lower.split())),  # word overlap
        ])

    return np.array(features)