import numpy as np

def dcg(relevance_scores):
    return sum(
        (2**rel - 1) / np.log2(idx + 2)
        for idx, rel in enumerate(relevance_scores)
    )

def ndcg_at_k(y_true, y_pred, k=10):
    # sort by predicted scores
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = np.take(y_true, order[:k])

    dcg_val = dcg(y_true_sorted)

    # ideal ranking
    ideal_order = np.argsort(y_true)[::-1]
    ideal_sorted = np.take(y_true, ideal_order[:k])

    idcg_val = dcg(ideal_sorted)

    return dcg_val / idcg_val if idcg_val > 0 else 0.0