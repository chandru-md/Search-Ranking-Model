import numpy as np
import lightgbm as lgb

from data.load_istella import load_istella
from data.grouping import get_group
from evaluation.metrics import ndcg_at_k

# Load validation data
X_val, y_val, qid_val = load_istella("data/raw/test.txt", limit=10000)

X_val = np.array(X_val)
y_val = np.array(y_val)

# Load model
model = lgb.Booster(model_file="models/ltr_model.txt")

# Predict scores
y_pred = model.predict(X_val)

# Evaluate per query
ndcg_scores = []
current_qid = qid_val[0]
start = 0

for i in range(1, len(qid_val)):
    if qid_val[i] != current_qid:
        ndcg = ndcg_at_k(
            y_val[start:i],
            y_pred[start:i],
            k=10
        )
        ndcg_scores.append(ndcg)

        current_qid = qid_val[i]
        start = i

# last query
ndcg_scores.append(
    ndcg_at_k(y_val[start:], y_pred[start:], k=10)
)

print("Average NDCG@10:", np.mean(ndcg_scores))