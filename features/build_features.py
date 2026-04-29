from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def build_features(query, documents):
    query_embedding = model.encode([query])[0]
    doc_embeddings = model.encode(documents)

    similarity_scores = cosine_similarity(
        [query_embedding],
        doc_embeddings
    )[0]

    # simple feature set (can expand later)
    features = []

    for i, doc in enumerate(documents):
        features.append([
            similarity_scores[i],   # semantic similarity
            len(doc.split()),       # doc length
        ])

    return np.array(features)