"""Restricted feature grammar: entity path x time window x aggregation.

    Each feature is a point-in-time group-by, never a target encoding.
"""
import numpy as np
import pandas as pd
from shared.data import before

MANUAL_WINDOWS = [7, 30, 90, 0]
AUTO_WINDOWS = [2, 3, 5, 10, 14, 21, 45, 60]
PATHS = {"u": ["user_id"], "i": ["item_id"], "ui": ["user_id", "item_id"],
         "ug": ["user_id", "genre"], "ud": ["user_id", "director"]}
CATEGORIES = ["content_type", "genre", "director", "country", "age_rating", "for_kids",
              "age", "income", "sex", "kids_flg"]


def specifications():
    return [{"window_days": w, "path": name, "keys": keys, "aggregations": ["count", "watched_mean"]}
            for w in MANUAL_WINDOWS + AUTO_WINDOWS for name, keys in PATHS.items()]


def build_features(events, items, users, queries, cutoff, include_auto=True):
    past = before(events, cutoff).merge(items[["item_id", "genre", "director"]], on="item_id", how="left")
    x = queries[["user_id", "item_id"]].copy().merge(items, on="item_id", how="left", validate="many_to_one")
    x = x.merge(users, on="user_id", how="left", validate="many_to_one")
    # Preserve candidate order through a series of left joins.
    for col in CATEGORIES:
        if col not in x:
            x[col] = "unknown"
        x[col] = x[col].astype("string").fillna("unknown").astype(str)
    for name, keys in PATHS.items():
        recent = past.groupby(keys, dropna=False).timestamp.max().rename(f"{name}_last")
        x = x.merge(recent, on=keys, how="left", sort=False)
        x[f"{name}_recency"] = (pd.Timestamp(cutoff) - x.pop(f"{name}_last")).dt.total_seconds().div(86400).fillna(999)
    windows = MANUAL_WINDOWS + (AUTO_WINDOWS if include_auto else [])
    derived = {}
    for w in windows:
        h = past if w == 0 else past[past.timestamp >= pd.Timestamp(cutoff) - pd.Timedelta(days=w)]
        for name, keys in PATHS.items():
            agg = h.groupby(keys, dropna=False).agg(n=("item_id", "size"), mean=("watched_pct", "mean"))
            agg = agg.rename(columns={"n": f"{name}_count_w{w}", "mean": f"{name}_watch_w{w}"})
            x = x.merge(agg, on=keys, how="left", sort=False)
        for path in ["ug", "ud", "ui"]:
            derived[f"{path}_share_w{w}"] = x[f"{path}_count_w{w}"].fillna(0) / (x[f"u_count_w{w}"].fillna(0)+1)
        derived[f"i_rate_w{w}"] = x[f"i_count_w{w}"].fillna(0) / max(len(h), 1)
    derived["i_trend_7_30"] = (x.i_count_w7.fillna(0)+1)/(x.i_count_w30.fillna(0)+1)
    derived["u_trend_7_30"] = (x.u_count_w7.fillna(0)+1)/(x.u_count_w30.fillna(0)+1)
    x = pd.concat([x,pd.DataFrame(derived,index=x.index)],axis=1)
    x = x.drop(columns=["user_id", "item_id"])
    return pd.DataFrame({c:x[c] if c in CATEGORIES else pd.to_numeric(x[c], errors="coerce").fillna(-1).astype(np.float32)
                         for c in x.columns})


def manual_columns(columns):
    return [c for c in columns if not any(c.endswith(f"_w{w}") for w in AUTO_WINDOWS)]
