import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def per_query(frame, scores, k=10):
    d = frame[["query_id", "item_id", "label"]].copy()
    d["score"] = scores
    d = d.sort_values(["query_id", "score", "item_id"], ascending=[True, False, True])
    result = []
    for q, g in d.groupby("query_id", sort=True):
        rel = g.label.to_numpy()
        total = rel.sum()
        top = rel[:k]
        discount = 1 / np.log2(np.arange(len(top)) + 2)
        ideal = (1 / np.log2(np.arange(min(int(total), k)) + 2)).sum()
        result.append({"query_id": q, "recall": float(top.sum() / total),
                       "ndcg": float((top * discount).sum() / ideal),
                       "map": float((np.cumsum(top) / np.arange(1, len(top)+1) * top).sum() / min(total, k))})
    return pd.DataFrame(result)


def ranking_summary(frame, scores, probabilities=None, k=10):
    p = per_query(frame, scores, k)
    result = {f"{m}@{k}": float(p[m].mean()) for m in ["recall", "ndcg", "map"]}
    if probabilities is not None:
        result["sampled_logloss"] = float(log_loss(frame.label, probabilities))
        result["sampled_roc_auc"] = float(roc_auc_score(frame.label, probabilities))
    return result, p


def paired_bootstrap(a, b, seed=42, repeats=2000):
    """User/query paired CI for one final test comparison, not model selection."""
    merged = a.merge(b, on="query_id", suffixes=("_a", "_b"), validate="one_to_one")
    rng = np.random.default_rng(seed)
    out = {}
    for metric in ["ndcg", "recall", "map"]:
        delta = (merged[f"{metric}_b"] - merged[f"{metric}_a"]).to_numpy()
        draws = [rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(repeats)]
        out[metric] = {"delta": float(delta.mean()), "ci95": np.quantile(draws, [.025,.975]).tolist()}
    return out
