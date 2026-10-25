"""Observed-response RM with strictly historical completion aggregates.

This extension uses richer state than the original item-ID RM: previous watch
fractions, full released history, metadata and demographics. It is reported as
an information/architecture extension, never as an isolated architecture gain.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from shared.data import ROOT, load_sample, save_json
from shared.analyze import bootstrap
from task1.features import CATEGORIES


def cumulative_table(events, keys):
    frame = events.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp).astype("datetime64[ns]")
    frame["watch_sum"] = frame.watched_pct.fillna(0)/100
    frame["watch_n"] = frame.watched_pct.notna().astype(int)
    daily = frame.groupby(keys+["timestamp"], as_index=False, dropna=False).agg(
        count=("item_id", "size"), watch_sum=("watch_sum", "sum"), watch_n=("watch_n", "sum"))
    daily = daily.sort_values(keys+["timestamp"])
    daily[["count","watch_sum","watch_n"]] = daily.groupby(keys, dropna=False)[["count","watch_sum","watch_n"]].cumsum()
    return daily.rename(columns={"timestamp": "history_time"}).sort_values("history_time")


def asof_stats(table, queries, keys, shift=0):
    q = queries[keys+["timestamp"]].copy()
    for key in keys:
        q[key] = q[key].astype(table[key].dtype)
    q["timestamp"] = q.timestamp-pd.Timedelta(days=shift)
    q["timestamp"] = q.timestamp.astype("datetime64[ns]")
    q["_row"] = np.arange(len(q))
    joined = pd.merge_asof(q.sort_values("timestamp"), table, left_on="timestamp", right_on="history_time",
        by=keys, direction="backward", allow_exact_matches=False)
    return joined.sort_values("_row").reset_index(drop=True)


class ContextFeatures:
    def __init__(self, events, items, users):
        self.items = items; self.users = users
        self.events = events.merge(items[["item_id","genre"]], on="item_id", how="left", validate="many_to_one")
        self.paths = {"u": ["user_id"], "i": ["item_id"], "ug": ["user_id","genre"]}
        self.tables = {name: cumulative_table(self.events, keys) for name, keys in self.paths.items()}

    def transform(self, queries):
        q = queries.merge(self.items, on="item_id", how="left", validate="many_to_one").merge(
            self.users, on="user_id", how="left", validate="many_to_one")
        q["genre"] = q.genre.fillna("unknown")
        result = q[[c for c in CATEGORIES+["release_year"] if c in q]].copy()
        for name, keys in self.paths.items():
            # Filtering by the first key preserves every historical match and
            # avoids scanning unrelated users/items for tiny on-policy batches.
            table = self.tables[name]
            table = table[table[keys[0]].isin(q[keys[0]])]
            now = asof_stats(table, q, keys)
            recency = (q.timestamp-now.history_time).dt.total_seconds().div(86400).fillna(999)
            result[f"{name}_recency"] = recency.to_numpy()
            for days in [0,7,30]:
                stats = now[["count","watch_sum","watch_n"]].fillna(0).copy()
                if days:
                    previous = asof_stats(table, q, keys, days)
                    stats -= previous[["count","watch_sum","watch_n"]].fillna(0)
                result[f"{name}_count{days}"] = stats["count"].to_numpy()
                result[f"{name}_watch{days}"] = np.where(stats.watch_n > 0, stats.watch_sum/stats.watch_n.clip(lower=1), -1)
        for col in result:
            if col in CATEGORIES: result[col] = result[col].astype("string").fillna("unknown").astype(str)
            else: result[col] = pd.to_numeric(result[col], errors="coerce").fillna(-1).astype(np.float32)
        return result


def examples_frame(part, inverse):
    return pd.DataFrame({"user_id": part["user"], "item_id": [inverse.get(int(token), -1) for token in part["y"]],
                         "timestamp": pd.to_datetime(part["date"]).astype("datetime64[ns]")})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--source", default="artifacts/task2")
    ap.add_argument("--out", default="artifacts/task2_context_reward"); cli = ap.parse_args()
    out = ROOT/cli.out; src = ROOT/cli.source; out.mkdir(exist_ok=True)
    protocol = json.loads((src/"protocol.json").read_text()); args = protocol["args"]
    mapping = {int(k):int(v) for k,v in json.loads((src/"item_mapping.json").read_text()).items()}
    inverse = {v:k for k,v in mapping.items()}; parts = {}
    for split in ["train","valid","test"]:
        with np.load(src/f"examples_{split}.npz", allow_pickle=False) as z: parts[split] = {k:z[k] for k in z.files}
    save_json(out/"study_protocol.json", {"purpose": "Exploratory richer observed-response model",
        "additional_information": "Past watch fractions, full past user history, item metadata and demographics",
        "paths": ["user", "item", "user-first-genre"], "windows": [0,7,30],
        "history_rule": "Strict timestamp < target day, all events from the same day excluded",
        "depths": [4,6,8], "iterations": 600, "learning_rate": .05, "l2": 10, "early_stop": 50,
        "selection": "Minimum validation RMSE; no test selection", "cold_target": "UNK item metadata/stats unknown, retained",
        "warning": "Different available state from original RM; cannot attribute difference solely to CatBoost; observational responses only"})
    start = time.perf_counter(); events, items, users = load_sample(args["users"])
    features = ContextFeatures(events, items, users); x = {}
    for split in parts:
        x[split] = features.transform(examples_frame(parts[split], inverse))
        x[split].to_parquet(out/f"{split}_features.parquet", index=False)
        print("CONTEXT FEATURES", split, x[split].shape, flush=True)
    rows = []; models = {}; chosen = None
    train_observed = np.isfinite(parts["train"]["reward"][:,0])
    valid_observed = np.isfinite(parts["valid"]["reward"][:,0])
    for depth in [4,6,8]:
        t = time.perf_counter(); model = CatBoostRegressor(iterations=600, depth=depth, learning_rate=.05,
            l2_leaf_reg=10, random_seed=42, loss_function="RMSE", thread_count=4, verbose=False, allow_writing_files=False)
        model.fit(x["train"].loc[train_observed], parts["train"]["reward"][train_observed,0],
            cat_features=[c for c in CATEGORIES if c in x["train"]],
            eval_set=(x["valid"].loc[valid_observed], parts["valid"]["reward"][valid_observed,0]), early_stopping_rounds=50)
        p = np.clip(model.predict(x["valid"]),0,1)
        rmse = float(np.sqrt(mean_squared_error(parts["valid"]["reward"][valid_observed,0], p[valid_observed])))
        row = {"depth": depth, "split": "valid", "rmse": rmse, "trees": model.tree_count_, "fit_seconds": time.perf_counter()-t}
        rows.append(row); models[depth] = model
        if chosen is None or rmse < chosen["rmse"]: chosen = row
        print("CONTEXT VALID", row, flush=True)
    save_json(out/"selection.json", {"selected_depth": chosen["depth"], "validation": rows})
    model = models[chosen["depth"]]; model.save_model(str(out/"reward.cbm"))
    for split in ["valid", "test"]:
        pred = np.clip(model.predict(x[split]),0,1); truth = parts[split]["reward"][:,0]
        observed = np.isfinite(truth); y = truth[observed]; p = pred[observed]
        metrics = {"depth": chosen["depth"], "split": split, "selected": True,
            "mae": float(mean_absolute_error(y,p)), "rmse": float(np.sqrt(mean_squared_error(y,p))),
            "r2": float(r2_score(y,p)), "n": len(y)}
        rows.append(metrics)
        pd.DataFrame({"user_id": parts[split]["user"], "truth": truth, "prediction": pred}).to_csv(out/f"predictions_{split}.csv", index=False)
        print("CONTEXT FINAL", metrics, flush=True)
    pd.DataFrame({"feature": model.feature_names_, "importance": model.feature_importances_}).sort_values("importance",ascending=False).to_csv(out/"feature_importance.csv",index=False)
    pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
    save_json(out/"runtime.json", {"total_seconds": time.perf_counter()-start})


if __name__ == "__main__": main()
