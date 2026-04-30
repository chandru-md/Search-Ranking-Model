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
def display_results(query, results):
    print("\n" + "="*50)
    print(f"🔍 Query: {query}")
    print("="*50)

    print("\nTop Results:\n")

    for i, (doc, score) in enumerate(results, start=1):
        print(f"{i}. {doc} (Score: {float(score):.2f})")

    print("\n" + "="*50)
if __name__ == "__main__":
    query = input("Enter your search query: ")
    results = search(query)

    display_results(query, results)
    