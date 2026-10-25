"""Shared, explicit temporal protocol. No random row sampling."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CUTOFFS = ["2021-07-12", "2021-07-19", "2021-07-26", "2021-08-02"]
VAL_CUTOFF = "2021-08-09"
TEST_CUTOFF = "2021-08-16"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def load_sample(n_users=50000, seed=42):
    raw = ROOT / "data/raw"
    cache = ROOT / f"data/sample_{n_users}_{seed}.parquet"
    items = pd.read_csv(raw / "items.csv")
    for source, target in [("genres", "genre"), ("directors", "director"), ("countries", "country")]:
        items[target] = items[source].fillna("unknown").str.split(",").str[0].str.strip()
    keep = ["item_id", "content_type", "release_year", "genre", "director", "country", "age_rating", "for_kids"]
    items = items[keep]
    users = pd.read_csv(raw / "users.csv")
    if cache.exists():
        events = pd.read_parquet(cache)
    else:
        # First scan reads only IDs. Second scan retains every row of sampled users.
        ids = pd.read_csv(raw / "interactions.csv", usecols=["user_id"]).user_id.unique()
        ids = np.sort(ids)
        chosen = np.random.default_rng(seed).choice(ids, min(n_users, len(ids)), replace=False)
        chunks = []
        for c in pd.read_csv(raw / "interactions.csv", chunksize=500000):
            chunks.append(c[c.user_id.isin(chosen)])
        events = pd.concat(chunks, ignore_index=True)
        if events[["user_id","item_id","last_watch_dt"]].isna().any().any():
            raise ValueError("Missing source identifiers/timestamps: verify interactions.csv completeness")
        events["timestamp"] = pd.to_datetime(events.pop("last_watch_dt"))
        events["watched_pct"] = pd.to_numeric(events.watched_pct, errors="coerce").clip(0, 100)
        events = events.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(cache, index=False)
        save_json(cache.with_suffix(".json"), {
            "requested_users": n_users, "sampled_users": len(chosen), "rows": len(events),
            "sampling_seed": seed, "user_ids_sha256": hashlib.sha256(np.sort(chosen).tobytes()).hexdigest(),
            "min_timestamp": events.timestamp.min(), "max_timestamp": events.timestamp.max(),
            "policy": "Uniform user sample; all released source rows retained for every selected user",
        })
    return events, items, users


def before(events, cutoff):
    return events.loc[events.timestamp < pd.Timestamp(cutoff)].copy()


def candidate_queries(events, cutoff, negatives=100, max_users=1500, seed=42):
    """Conditional ranking: all known-catalog positives + fixed synthetic negatives.

    Excludes historical items. Cold positives counted separately; no target is used
    to define catalog availability, features, or negative sampling probabilities.
    """
    cutoff = pd.Timestamp(cutoff)
    past = before(events, cutoff)
    future = events[(events.timestamp >= cutoff) & (events.timestamp < cutoff + pd.Timedelta(days=7))]
    counts = past.groupby("item_id").size().sort_index()
    catalog = counts.index.to_numpy()
    prob = np.sqrt(counts.to_numpy(dtype=float))
    prob /= prob.sum()
    histories = past.groupby("user_id").item_id.agg(set).to_dict()
    targets = future.groupby("user_id").item_id.agg(set).to_dict()
    eligible = np.array(sorted(set(histories) & set(targets)))
    rng = np.random.default_rng(seed)
    if len(eligible) > max_users:
        eligible = np.sort(rng.choice(eligible, max_users, replace=False))
    catalog_set = set(catalog)
    rows, cold, total, skipped = [], 0, 0, 0
    for u in eligible:
        pos_all = targets[u] - histories[u]
        pos = pos_all & catalog_set
        total += len(pos_all)
        cold += len(pos_all - catalog_set)
        if not pos:
            skipped += 1
            continue
        forbidden = targets[u] | histories[u]
        pool = set()
        for _ in range(20):
            pool.update(rng.choice(catalog, size=negatives * 3, replace=True).tolist())
            pool.update(rng.choice(catalog, size=negatives * 3, replace=True, p=prob).tolist())
            pool -= forbidden
            if len(pool) >= negatives:
                break
        neg = rng.choice(sorted(pool), size=min(negatives, len(pool)), replace=False)
        qid = f"{cutoff.date()}:{u}"
        for item in sorted(pos):
            rows.append((u, item, qid, 1))
        for item in sorted(neg):
            rows.append((u, item, qid, 0))
    frame = pd.DataFrame(rows, columns=["user_id", "item_id", "query_id", "label"])
    frame["cutoff"] = cutoff
    return frame, {"eligible_sampled_users": len(eligible), "ranking_queries": frame.query_id.nunique(),
                   "cold_positives": cold, "all_positives": total, "all_cold_users_skipped": skipped,
                   "negative_count": negatives, "rows": len(frame)}
