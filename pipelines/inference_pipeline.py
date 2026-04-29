import numpy as np
import lightgbm as lgb
from data.load_istella import load_istella

# Load model
model = lgb.Booster(model_file="models/ltr_model.txt")

# Load data (simulate query-based ranking)
X, y, qid = load_istella("data/raw/test.txt", limit=10000)

X = np.array(X)
y = np.array(y)

def rank_query(query_id):
    # get indices for this query
    indices = [i for i, q in enumerate(qid) if q == query_id]

    X_query = X[indices]
    y_query = y[indices]

    # predict scores
    scores = model.predict(X_query)

    # sort results
    ranked = sorted(
        zip(scores, y_query),
        key=lambda x: x[0],
        reverse=True
    )

    return ranked[:10]  # top 10


# Example usage
sample_query = qid[0]
results = rank_query(sample_query)

print("Top Results (score, relevance):")
for r in results:
    print(r)