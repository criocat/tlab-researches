"""Initialize tied GPT-2 item embeddings from the train-only collaborative graph.

Same architecture, transition data and SFT optimization budget as the original.
Both prescribed initializers are reported; validation chooses the checkpoint and
whether either initializer is preferred to the frozen original random baseline.
"""
import argparse
import json
import time
import hashlib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits
from shared.data import ROOT, save_json, load_sample, VAL_CUTOFF
from task2.model import GPT2Rec
from task2.factorial import load_frozen
from task2.run import set_seed, train_policy, evaluate
from task2.robust_reward import paired


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--source", default="artifacts/task2")
    ap.add_argument("--out", default="artifacts/task2_warmstart"); cli = ap.parse_args()
    src = ROOT/cli.source; out = ROOT/cli.out; out.mkdir(exist_ok=True)
    torch.set_num_threads(6); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args, data, mapping, ref, rms = load_frozen(src, device)
    save_json(out/"study_protocol.json", {"purpose": "Exploratory train-graph initialization of GPT-2",
        "initializers": ["binary", "user_degree_normalized"], "svd_rank": args.dim,
        "factor": "right singular vectors times sqrt(singular values), global RMS scaled to original 0.02",
        "graph_cutoff": VAL_CUTOFF, "graph_seed": 42, "sft_seed": 42,
        "unchanged": ["GPT-2 architecture", "train transition arrays", "epochs", "optimizer", "validation checkpoint criterion"],
        "selection": "Highest validation NDCG among two graph initializers and frozen original SFT",
        "test": "All prespecified initializers reported after selection",
        "warning": "Original test inspected; one SFT seed, not a claim of universal initialization advantage"})
    events, _, _ = load_sample(args.users); past = events[events.timestamp < pd.Timestamp(VAL_CUTOFF)]
    u = pd.Index(np.sort(past.user_id.unique()))
    cols = np.array([mapping[int(i)]-2 for i in past.item_id])
    graph = sparse.csr_matrix((np.ones(len(past), dtype=np.float32), (u.get_indexer(past.user_id), cols)), shape=(len(u),len(mapping)))
    graph.data[:] = 1; degree = np.asarray(graph.sum(1)).ravel()
    counts = np.bincount(cols+2, minlength=len(mapping)+2)
    save_json(out/"graph_audit.json", {"cutoff": VAL_CUTOFF, "max_timestamp": str(past.timestamp.max()),
        "edges": int(graph.nnz), "users": len(u), "items": len(mapping),
        "event_sha256": hashlib.sha256(past[["user_id","item_id"]].to_numpy().tobytes()).hexdigest(),
        "scope": "Training graph only; no validation/test labels or edges"})
    start = time.perf_counter(); validation = {"original_sft": evaluate(ref,data["valid"],rms,ref,counts,args.batch)[0]}
    rows = [{"name": "original_sft", "split": "valid", **validation["original_sft"]}]
    config = ref.config
    for mode in ["binary", "user_degree_normalized"]:
        matrix = graph if mode == "binary" else sparse.diags(1/np.sqrt(degree))@graph
        with threadpool_limits(limits=4):
            svd = TruncatedSVD(n_components=args.dim, n_iter=7, random_state=42).fit(matrix)
        factors = svd.components_.T*np.sqrt(svd.singular_values_)[None,:]
        factors = (factors*(.02/np.sqrt(np.mean(factors**2)))).astype(np.float32)
        set_seed(args.seed); model = GPT2Rec(**config).to(device)
        with torch.no_grad(): model.item.weight[2:].copy_(torch.as_tensor(factors, device=device))
        cell = out/mode; cell.mkdir(exist_ok=True)
        save_json(cell/"initialization.json", {"rms": float(np.sqrt(np.mean(factors**2))),
            "factor_sha256": hashlib.sha256(factors.tobytes()).hexdigest(), "singular_values": svd.singular_values_.tolist()})
        t = time.perf_counter(); train_policy(model, data["train"], data["valid"], args, cell)
        validation[mode] = evaluate(model, data["valid"], rms, ref, counts, args.batch)[0]
        rows.append({"name": mode, "split": "valid", "train_seconds": time.perf_counter()-t, **validation[mode]})
        pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
        print("WARMSTART VALID", rows[-1], flush=True)
    selected = max(validation, key=lambda name: validation[name]["ndcg@10"])
    save_json(out/"selection.json", {"selected": selected, "validation": validation})
    comparisons = {}; baseline = None
    for name in ["original_sft", "binary", "user_degree_normalized"]:
        if name == "original_sft": model = ref
        else:
            model = GPT2Rec(**config).to(device)
            model.load_state_dict(torch.load(out/name/"sft_best.pt",map_location=device,weights_only=True))
        met, per, top = evaluate(model, data["test"], rms, ref, counts, args.batch)
        rows.append({"name": name, "split": "test", **met})
        per.to_csv(out/f"per_user_{name}.csv", index=False); np.save(out/f"top_{name}.npy", top)
        if baseline is None: baseline = per
        else: comparisons[name] = paired(baseline, per)
        print("WARMSTART TEST", rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
    save_json(out/"paired_test_ci.json", comparisons)
    save_json(out/"runtime.json", {"total_seconds": time.perf_counter()-start})


if __name__ == "__main__": main()
