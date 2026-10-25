"""python -m task1.run --users 50000 --queries 1500 --seeds 42 43 44"""
import argparse
import importlib.metadata
import resource
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from shared.data import ROOT, TRAIN_CUTOFFS, VAL_CUTOFF, TEST_CUTOFF, load_sample, candidate_queries, save_json
from shared.metrics import ranking_summary, paired_bootstrap
from task1.features import build_features, manual_columns, specifications, CATEGORIES, AUTO_WINDOWS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=50000)
    ap.add_argument("--queries", type=int, default=1500)
    ap.add_argument("--iterations", type=int, default=350)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="artifacts/task1")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    e, items, users = load_sample(args.users)
    frames, queries, audit = {}, {}, {}
    for split, cutoffs, neg in [("train", TRAIN_CUTOFFS, 30), ("valid", [VAL_CUTOFF], 100), ("test", [TEST_CUTOFF], 100)]:
        xx, qq, audits = [], [], []
        for cutoff in cutoffs:
            q, a = candidate_queries(e, cutoff, neg, args.queries, seed=42)
            t = time.perf_counter()
            x = build_features(e, items, users, q, cutoff)
            a["feature_seconds"] = time.perf_counter()-t
            a["cutoff"] = cutoff
            xx.append(x); qq.append(q); audits.append(a)
            print(split, cutoff, len(q), "features", x.shape[1], flush=True)
        frames[split] = pd.concat(xx, ignore_index=True)
        queries[split] = pd.concat(qq, ignore_index=True)
        frames[split].to_parquet(out / f"{split}_features.parquet", index=False)
        queries[split].to_parquet(out / f"{split}_candidates.parquet", index=False)
        audit[split] = audits
    save_json(out/"protocol.json", {"args": vars(args), "audits": audit,
        "sample_rows": len(e), "sample_users": e.user_id.nunique(), "feature_specs": specifications(),
        "versions": {n:importlib.metadata.version(n) for n in ["numpy", "pandas", "catboost", "scikit-learn"]}})
    base_cols = manual_columns(frames["train"].columns)
    rows, per_test, chosen, importance = [], {}, None, None
    def fit(name, cols, seed):
        t = time.perf_counter()
        model = CatBoostClassifier(iterations=args.iterations, depth=6, learning_rate=.07, l2_leaf_reg=5,
            loss_function="Logloss", random_seed=seed, thread_count=6, verbose=False, allow_writing_files=False)
        model.fit(frames["train"][cols], queries["train"].label,
                  cat_features=[c for c in CATEGORIES if c in cols],
                  eval_set=(frames["valid"][cols], queries["valid"].label), early_stopping_rounds=50)
        train_seconds = time.perf_counter()-t
        # Only validation is inspected during feature/hyperparameter selection.
        p = model.predict_proba(frames["valid"][cols])[:,1]
        met, _ = ranking_summary(queries["valid"], p, p)
        row = {"model": name, "seed": seed, "split": "valid", "features": len(cols),
               "train_seconds": train_seconds, "trees": model.tree_count_, **met}
        rows.append(row)
        print(row, flush=True)
        return model, met
    # Generate all feature blocks, rank by train-fitted CatBoost gain, then choose
    # number of complete window blocks exclusively on validation NDCG.
    first = args.seeds[0]
    base, _ = fit("manual", base_cols, first)
    allcols = list(frames["train"].columns)
    allmodel, allmet = fit("generated_all", allcols, first)
    importance = pd.DataFrame({"feature": allcols, "importance": allmodel.feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(out/"feature_importance.csv", index=False)
    gains = {w: float(importance.loc[importance.feature.str.endswith(f"_w{w}"), "importance"].sum()) for w in AUTO_WINDOWS}
    order = sorted(AUTO_WINDOWS, key=lambda w: -gains[w])
    options = [(allmet["ndcg@10"], "all", allcols, allmodel)]
    for n in [1, 2, 4]:
        selected = order[:n]
        cols = base_cols + [c for c in allcols if any(c.endswith(f"_w{w}") for w in selected)]
        model, met = fit(f"generated_top{n}", cols, first)
        options.append((met["ndcg@10"], f"top{n}", cols, model))
    _, selection_name, chosen, selected_model = max(options, key=lambda a: a[0])
    save_json(out/"selection.json", {"window_gain": gains, "window_order": order,
               "selected": selection_name, "columns": chosen, "criterion": "validation ndcg@10"})
    # Freeze selection before evaluating any test predictions.
    for seed in args.seeds:
        models = {"manual": base if seed==first else fit("manual", base_cols, seed)[0],
                  "auto_selected": selected_model if seed==first else fit("auto_selected", chosen, seed)[0]}
        if seed==first:
            models["generated_all"] = allmodel
            noindirect = [c for c in base_cols if not c.startswith(("ug_", "ud_"))]
            models["manual_no_indirect"] = fit("manual_no_indirect", noindirect, seed)[0]
        for name, model in models.items():
            cols = model.feature_names_
            t = time.perf_counter()
            p = model.predict_proba(frames["test"][cols])[:,1]
            seconds = time.perf_counter()-t
            met, per = ranking_summary(queries["test"], p, p)
            row = {"model": name, "seed": seed, "split": "test", "features": len(cols), "predict_seconds": seconds, **met}
            rows.append(row); per_test[(name,seed)] = per
            per.to_csv(out/f"per_query_{name}_{seed}.csv", index=False)
            if name in ["manual", "auto_selected"]:
                model.save_model(str(out/f"{name}_{seed}.cbm"))
            print(row, flush=True)
    scores = frames["test"].i_count_w7.to_numpy()
    met, _ = ranking_summary(queries["test"], scores)
    rows.append({"model": "popularity_7d", "seed": 42, "split": "test", **met})
    pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
    save_json(out/"paired_test_ci.json", {str(s): paired_bootstrap(per_test[("manual",s)],per_test[("auto_selected",s)]) for s in args.seeds})
    save_json(out/"runtime.json", {"total_seconds": time.perf_counter()-start,
                                  "max_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


if __name__ == "__main__":
    main()
