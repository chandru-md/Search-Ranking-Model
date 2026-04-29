import lightgbm as lgb
import numpy as np

from retrieval.hybrid_search import hybrid_search
from features.build_features import build_features

# load trained model
model = lgb.Booster(model_file="models/ltr_model.txt")

def search(query):
    # Step 1: Retrieve documents
    retrieved = hybrid_search(query)

    documents = [doc for doc, _ in retrieved]

    # Step 2: Build features
    X = build_features(query, documents)

    # Step 3: Rank using model
    scores = model.predict(X)

    ranked = sorted(
        list(zip(documents, scores)),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


# test
if __name__ == "__main__":
    results = search("learn machine learning")

    print("\nFinal Ranked Results:")
    for r in results:
        print(r)