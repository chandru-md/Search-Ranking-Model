import pandas as pd
import numpy as np
from features.build_features import build_features
from data.load_documents import load_documents

def generate_data():
    df = load_documents()

    queries = [
    "machine learning",
    "python programming",
    "deep learning",
    "data science",
    "artificial intelligence",
    "neural networks",
    "learn coding",
    "statistics basics"
]

    X, y, qid = [], [], []
    query_id = 0

    for query in queries:
        documents = df["text"].tolist()

        features = build_features(query, documents)

        # simulate relevance (simple logic)
        relevance = []

    for doc in documents:
        doc_lower = doc.lower()
        query_lower = query.lower()

        score = 0

        if query_lower in doc_lower:
            score += 2

        score += len(set(query_lower.split()) & set(doc_lower.split()))

        relevance.append(min(score, 4))

        X.extend(features)
        y.extend(relevance)
        qid.extend([query_id] * len(documents))

        query_id += 1

    return np.array(X), np.array(y), qid