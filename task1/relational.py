"""Learned collaborative features with a strict per-snapshot graph cutoff.

No target labels or later interactions enter SVD or item-neighbour estimation.
The original candidates and hand features are reused byte-for-byte.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits
from catboost import CatBoostClassifier
from shared.data import ROOT, load_sample, save_json, TRAIN_CUTOFFS, VAL_CUTOFF, TEST_CUTOFF
from shared.metrics import ranking_summary, paired_bootstrap
from task1.features import manual_columns, CATEGORIES


def top_neighbours(matrix, k):
    """Keep strongest positive off-diagonal neighbours, deterministic ID ties."""
    matrix = matrix.tocsr(copy=True)
    matrix.setdiag(0); matrix.eliminate_zeros(); matrix.sort_indices()
    rows, cols, values = [], [], []
    for i in range(matrix.shape[0]):
        lo, hi = matrix.indptr[i:i+2]
        v, j = matrix.data[lo:hi], matrix.indices[lo:hi]
        order = np.lexsort((j, -v))[:k]
        rows.extend([i]*len(order)); cols.extend(j[order]); values.extend(v[order])
    return sparse.csr_matrix((np.asarray(values, dtype=np.float32), (rows, cols)), shape=matrix.shape)


def graph_features(events, queries, cutoff, seed=42, half_life=None):
    past = events.loc[events.timestamp < pd.Timestamp(cutoff)]
    if len(past) == 0:
        raise ValueError("Empty historical graph")
    user_ids = np.sort(past.user_id.unique()); item_ids = np.sort(past.item_id.unique())
    u = pd.Index(user_ids); i = pd.Index(item_ids)
    if half_life is not None and half_life <= 0:
        raise ValueError("Graph half-life must be positive")
    weights = np.ones(len(past), dtype=np.float32)
    if half_life is not None:
        age=(pd.Timestamp(cutoff)-past.timestamp).dt.total_seconds().to_numpy()/86400
        weights=np.exp(-np.log(2)*age/half_life).astype(np.float32)
    a = sparse.csr_matrix((weights,
                          (u.get_indexer(past.user_id), i.get_indexer(past.item_id))),
                         shape=(len(u), len(i)))
    if half_life is None:
        a.data[:] = 1  # Default binary history, exactly the original branch.
    qu, qi = u.get_indexer(queries.user_id), i.get_indexer(queries.item_id)
    if (qu < 0).any() or (qi < 0).any():
        raise ValueError("Candidates must have a warm graph user and item")
    uc = np.asarray(a.sum(1)).ravel(); ic = np.asarray(a.sum(0)).ravel()
    features = {}
    with threadpool_limits(limits=4):
        for name, mat in [("binary", a), ("normalized", sparse.diags(1/np.sqrt(uc)) @ a @ sparse.diags(1/np.sqrt(ic)))]:
            rank = min(64, min(mat.shape)-1)
            svd = TruncatedSVD(n_components=rank, n_iter=5, random_state=seed)
            profile = svd.fit_transform(mat).astype(np.float32)
            item = svd.components_.T.astype(np.float32)
            for dim in [16, 32, 64]:
                d = min(dim, rank)
                left, right = profile[qu, :d], item[qi, :d]
                dot = np.einsum("ij,ij->i", left, right)
                norm = np.linalg.norm(left, axis=1)*np.linalg.norm(right, axis=1)
                features[f"svd_{name}_dot{dim}"] = dot
                features[f"svd_{name}_cos{dim}"] = dot/np.maximum(norm, 1e-12)
        minimum=1 if half_life is None else 1e-12
        co = (a.T @ sparse.diags(1/np.maximum(uc, minimum)) @ a).tocsr()
        diag = np.sqrt(co.diagonal()).clip(1e-12)
        cosine = sparse.diags(1/diag) @ co @ sparse.diags(1/diag)
        # Item->item similarities are estimated only from observed past edges.
        for k in [50, 200]:
            neighbours = top_neighbours(cosine, k)
            scores = a @ neighbours
            total = np.asarray(scores[qu, qi]).ravel()
            features[f"knn_sum{k}"] = total
            features[f"knn_mean{k}"] = total/uc[qu]
    result = pd.DataFrame(features).astype(np.float32)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Nonfinite graph features")
    audit = {"cutoff": cutoff, "max_graph_timestamp": str(past.timestamp.max()),
             "users": len(u), "items": len(i), "edges": int(a.nnz),
             "graph_event_sha256": hashlib.sha256(past[["user_id", "item_id"]].to_numpy().tobytes()).hexdigest(),
             "features": list(result), "labels_used": False}
    if half_life is not None:
        audit["half_life_days"]=half_life
        audit["edge_weights"]="exp(-log(2)*age_days/half_life); legacy binary prefix denotes raw matrix"
    return result, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="artifacts/task1")
    ap.add_argument("--out", default="artifacts/task1_relational")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--fixed-group", choices=["all", "knn", "svd_binary", "svd_normalized"])
    ap.add_argument("--reuse-baseline", action="store_true")
    cli = ap.parse_args(); src = ROOT/cli.source; out = ROOT/cli.out; out.mkdir(parents=True, exist_ok=True)
    source_protocol = json.loads((src/"protocol.json").read_text())
    args = source_protocol["args"]
    cutoffs_by_split = {split: [a["cutoff"] for a in source_protocol["audits"][split]] for split in ["train", "valid", "test"]}
    # Protocol is written before fitting any new models or inspecting their test.
    save_json(out/"study_protocol.json", {"purpose": "Exploratory learned graph features after original test inspection",
        "cutoffs": cutoffs_by_split,
        "source": cli.source, "feature_options": ["svd_binary", "svd_normalized", "knn", "all"],
        "ranks": [16, 32, 64], "knn_sizes": [50, 200], "selection_seed": 42,
        "selection": "Frozen group from earlier relational study; no reselection" if cli.fixed_group else "Highest validation NDCG; unchanged manual baseline is an eligible fallback",
        "fixed_group": cli.fixed_group, "reuse_baseline": cli.reuse_baseline,
        "seeds": cli.seeds, "candidate_policy": "Exact original persisted candidate tables",
        "warning": "Original test has been seen; this is not an untouched confirmatory test"})
    events, _, _ = load_sample(args["users"])
    x, q, audits = {}, {}, {}; start = time.perf_counter()
    for split, cutoffs in cutoffs_by_split.items():
        q[split] = pd.read_parquet(src/f"{split}_candidates.parquet")
        original = pd.read_parquet(src/f"{split}_features.parquet")
        g = []; audits[split] = []
        for cutoff in cutoffs:
            # Query ID starts with cutoff; no slicing by future label values.
            part = q[split][q[split].query_id.str.startswith(cutoff)]
            t = time.perf_counter(); frame, audit = graph_features(events, part, cutoff)
            audit["seconds"] = time.perf_counter()-t
            g.append(frame); audits[split].append(audit)
            print("GRAPH", split, cutoff, len(frame), audit["seconds"], flush=True)
        graph = pd.concat(g, ignore_index=True)
        assert len(graph) == len(original)
        x[split] = pd.concat([original[manual_columns(original.columns)], graph], axis=1)
        x[split].to_parquet(out/f"{split}_features.parquet", index=False)
    save_json(out/"graph_audit.json", audits)
    base = manual_columns(pd.read_parquet(src/"train_features.parquet").columns)
    groups = {"manual": base}
    for name in ["svd_binary", "svd_normalized", "knn"]:
        groups[name] = base+[c for c in x["train"] if c.startswith(name+"_")]
    groups["all"] = list(x["train"])
    rows, first_models, validation = [], {}, {}
    def fit(name, cols, seed):
        t = time.perf_counter()
        reused = cli.reuse_baseline and name == "manual"
        if reused:
            model = CatBoostClassifier().load_model(str(src/f"manual_{seed}.cbm"))
            assert model.feature_names_ == cols
        else:
            model = CatBoostClassifier(iterations=args["iterations"], depth=6, learning_rate=.07,
                l2_leaf_reg=5, loss_function="Logloss", random_seed=seed, thread_count=6,
                verbose=False, allow_writing_files=False)
            model.fit(x["train"][cols], q["train"].label, cat_features=[c for c in CATEGORIES if c in cols],
                eval_set=(x["valid"][cols], q["valid"].label), early_stopping_rounds=50)
        p = model.predict_proba(x["valid"][cols])[:,1]
        met, per = ranking_summary(q["valid"], p, p)
        rows.append({"model": name, "seed": seed, "split": "valid", "features": len(cols),
                     "train_seconds": time.perf_counter()-t, "baseline_reused": reused, "trees": model.tree_count_, **met})
        print("VALID", rows[-1], flush=True)
        return model, met
    for name, cols in groups.items():
        if cli.fixed_group and name not in ["manual", cli.fixed_group]:
            continue
        first_models[name], validation[name] = fit(name, cols, cli.seeds[0])
    selected = cli.fixed_group or max(validation, key=lambda name: validation[name]["ndcg@10"])
    save_json(out/"selection.json", {"selected": selected, "validation": validation,
                                    "columns": groups[selected], "manual_fallback": cli.fixed_group is None})
    # Only now evaluate selected policy and matched manual baseline on test.
    per_models = {"manual": [], "graph_selected": []}
    for seed in cli.seeds:
        for label, name in [("manual", "manual"), ("graph_selected", selected)]:
            model = first_models[name] if seed == cli.seeds[0] else fit(name, groups[name], seed)[0]
            p = model.predict_proba(x["test"][groups[name]])[:,1]
            met, per = ranking_summary(q["test"], p, p)
            per_models[label].append(per)
            per.to_csv(out/f"per_query_{label}_{seed}.csv", index=False)
            rows.append({"model": label, "selected_features": name, "seed": seed, "split": "test",
                         "features": len(groups[name]), **met})
            model.save_model(str(out/f"{label}_{seed}.cbm"))
            print("TEST", rows[-1], flush=True)
    mean_per = {}
    for label, parts in per_models.items():
        mean_per[label] = pd.concat(parts).groupby("query_id", as_index=False)[["ndcg", "recall", "map"]].mean()
    save_json(out/"paired_test_ci.json", paired_bootstrap(mean_per["manual"], mean_per["graph_selected"]))
    pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
    save_json(out/"runtime.json", {"total_seconds": time.perf_counter()-start})


if __name__ == "__main__":
    main()
