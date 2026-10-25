"""Matched validation budgets for classification/ranking and feature families.

All families get the same two depths and two objectives. Test follows frozen
per-family validation choice, with three CatBoost seeds. Exploratory study.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from shared.data import ROOT, save_json
from shared.metrics import ranking_summary, paired_bootstrap
from task1.features import manual_columns, CATEGORIES


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="artifacts/task1_ranking")
    ap.add_argument("--iterations", type=int, default=500); cli = ap.parse_args()
    out = ROOT/cli.out; out.mkdir(exist_ok=True)
    save_json(out/"study_protocol.json", {"purpose": "Exploratory matched model-selection budget after original test",
        "families": ["manual", "graph", "auto_graph", "graph_only"], "objectives": ["Logloss", "YetiRank"],
        "depths": [4,6], "iterations": cli.iterations, "early_stop": 50, "learning_rate": .07, "l2": 5,
        "selection_seed": 42, "evaluation_seeds": [42,43,44],
        "selection": "Best validation NDCG checkpoint and setting per feature family; equal four candidate settings each",
        "test": "Only validation winners, no test-based choice", "candidates": "Original persisted task1 tables"})
    start = time.perf_counter(); x, q = {}, {}
    for split in ["train", "valid", "test"]:
        original = pd.read_parquet(ROOT/f"artifacts/task1/{split}_features.parquet")
        learned = pd.read_parquet(ROOT/f"artifacts/task1_relational/{split}_features.parquet")
        graphcols = [c for c in learned if c not in original.columns]
        x[split] = pd.concat([original, learned[graphcols]], axis=1)
        q[split] = pd.read_parquet(ROOT/f"artifacts/task1/{split}_candidates.parquet")
    base = manual_columns(original.columns)
    groups = {"manual": base, "graph": base+graphcols,
              "auto_graph": list(x["train"].columns), "graph_only": graphcols}
    def pool(split, cols):
        return Pool(x[split][cols], label=q[split].label,
                    group_id=q[split].query_id, cat_features=[c for c in CATEGORIES if c in cols])
    rows = []; choices = {}; models = {}
    def fit(family, objective, depth, seed):
        t = time.perf_counter()
        cls = CatBoostClassifier if objective == "Logloss" else CatBoostRanker
        model = cls(iterations=cli.iterations, depth=depth, learning_rate=.07, l2_leaf_reg=5,
            loss_function=objective, eval_metric="NDCG:top=10;type=Base", random_seed=seed,
            thread_count=4, verbose=False, allow_writing_files=False)
        model.fit(pool("train", groups[family]), eval_set=pool("valid", groups[family]), early_stopping_rounds=50)
        score = model.predict(x["valid"][groups[family]], prediction_type="RawFormulaVal") if objective == "Logloss" else model.predict(x["valid"][groups[family]])
        met, _ = ranking_summary(q["valid"], score)
        row = {"family": family, "objective": objective, "depth": depth, "seed": seed, "split": "valid",
               "features": len(groups[family]), "trees": model.tree_count_, "train_seconds": time.perf_counter()-t, **met}
        rows.append(row); pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
        print("RANKING VALID", row, flush=True)
        return model, row
    for family in groups:
        best = None
        for objective in ["Logloss", "YetiRank"]:
            for depth in [4,6]:
                model, row = fit(family, objective, depth, 42)
                if best is None or row["ndcg@10"] > best["ndcg@10"]:
                    best = row; models[family] = model
        choices[family] = best
    save_json(out/"selection.json", {"families": choices, "criterion": "Validation NDCG@10; all frozen before test"})
    per = {name: [] for name in groups}
    for family in groups:
        setting = choices[family]
        for seed in [42,43,44]:
            model = models[family] if seed == 42 else fit(family, setting["objective"], setting["depth"], seed)[0]
            score = model.predict(x["test"][groups[family]], prediction_type="RawFormulaVal") if setting["objective"] == "Logloss" else model.predict(x["test"][groups[family]])
            met, user = ranking_summary(q["test"], score)
            per[family].append(user)
            user.to_csv(out/f"per_query_{family}_{seed}.csv", index=False)
            rows.append({"family": family, "objective": setting["objective"], "depth": setting["depth"], "seed": seed,
                         "split": "test", "features": len(groups[family]), **met})
            model.save_model(str(out/f"{family}_{seed}.cbm"))
            print("RANKING TEST", rows[-1], flush=True)
    means = {name: pd.concat(parts).groupby("query_id", as_index=False)[["ndcg","recall","map"]].mean() for name, parts in per.items()}
    save_json(out/"paired_test_ci.json", {name: paired_bootstrap(means["manual"], means[name]) for name in groups if name != "manual"})
    pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
    save_json(out/"runtime.json", {"total_seconds": time.perf_counter()-start})


if __name__ == "__main__": main()
