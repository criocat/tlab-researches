"""Exploratory one-step RL using item-centred and ensemble reward functions.

Selection observes validation only. A third RM is a monitoring model, not a
member of the optimized ensemble; it still shares observational training data.
"""
import argparse
import copy
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from shared.data import ROOT, save_json, load_sample, VAL_CUTOFF
from shared.analyze import bootstrap
from task2.model import RewardModel
from task2.factorial import load_frozen
from task2.run import set_seed, rl_finetune, evaluate, train_reward, reward_eval


@torch.no_grad()
def item_priors(models, train_x, vocab, device, n_contexts=256):
    """Average RM over an independent fixed sample of *real* training contexts.

    This is a context-marginal RM prediction, not observed item completion.
    No empty/unrealistic history, validation or test labels are used.
    """
    rng = np.random.default_rng(20261025)
    idx = rng.choice(len(train_x), min(n_contexts, len(train_x)), replace=False)
    contexts = torch.as_tensor(train_x[idx], device=device)
    result = torch.zeros(len(models), vocab, device=device)
    for start in range(0, len(contexts), 16):
        x = contexts[start:start+16]
        for lo in range(2, vocab, 512):
            ids = torch.arange(lo, min(lo+512, vocab), device=device)[None,:].expand(len(x), -1)
            for m, model in enumerate(models):
                result[m, lo:lo+ids.shape[1]] += model(x, ids)[:,:,0].sum(0)
    result /= len(contexts)
    return result, idx


class ControlledReward(nn.Module):
    def __init__(self, models, priors, mode):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer("priors", priors)
        self.mode = mode

    def forward(self, history, candidates):
        outputs = torch.stack([m(history, candidates) for m in self.models])
        scores = outputs[...,0]
        if "residual" in self.mode:
            scores = scores-self.priors[:, candidates]
        score = scores.min(0).values if "lcb" in self.mode else scores.mean(0)
        return torch.stack([score, outputs[...,1].mean(0)], dim=-1)


def paired(base, per):
    j = base.merge(per, on="user_id", suffixes=("_a", "_b"), validate="one_to_one")
    assert len(j) == len(base) == len(per)
    return {"ndcg": bootstrap(j.ndcg_b-j.ndcg_a), "recall": bootstrap(j.hit_b.astype(float)-j.hit_a.astype(float))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--source", default="artifacts/task2")
    ap.add_argument("--out", default="artifacts/task2_robust"); cli = ap.parse_args()
    out = ROOT/cli.out; src = ROOT/cli.source; out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args, data, mapping, ref, rms = load_frozen(src, device)
    configs = [{"name": f"{mode}_beta{beta:g}", "mode": mode, "beta": beta,
                "ce": .2, "horizon": 1, "reward": "watch"}
               for mode, betas in [("mean", [.1]), ("lcb", [.1]), ("residual", [.1, 1.]), ("residual_lcb", [.1, 1.])]
               for beta in betas]
    save_json(out/"study_protocol.json", {"purpose": "Exploratory reward redesign after original test inspection",
        "configs": configs, "fit_seed": 42, "replication_seeds": [43,44], "steps": args.rl_steps,
        "frozen_policy_and_RMs": cli.source,
        "selection": "Max validation original RM predicted_watch among >=99% SFT NDCG and max-item fraction <= SFT+0.02; SFT fallback",
        "residual_definition": "Subtract item-only RM mean over 256 fixed real training contexts, separately per RM",
        "lcb_definition": "Minimum of two predictions; heuristic disagreement penalty, not calibrated confidence bound",
        "ensemble": "Original reward and original auditor; auditor now participates in training and is not independent evaluation",
        "monitor": "Third new RM with seed 73, same data, excluded from reward optimization and selection",
        "test_policy": "Test only after frozen selection; baseline, selected and fixed mean/lcb/residual/residual_lcb beta0.1 diagnostics",
        "warning": "Post-hoc study, same observed data; no causal retention or online watch improvement claims"})
    start = time.perf_counter(); set_seed(73)
    monitor = RewardModel(len(mapping)+2).to(device)
    train_reward(monitor, data["train"], data["valid"], args, out, "monitor", 73)
    monitoring = [rms[0], monitor]
    # Monitor quality is characterized separately; never used to select a policy.
    save_json(out/"monitor_metrics.json", {s: reward_eval(monitor, data[s])[0] for s in ["valid", "test"]})
    priors, context_idx = item_priors(rms, data["train"]["x"], len(mapping)+2, device)
    np.save(out/"item_priors.npy", priors.cpu().numpy())
    save_json(out/"prior_audit.json", {"n_contexts": len(context_idx), "context_indices": context_idx.tolist(),
        "context_sha256": hashlib.sha256(data["train"]["x"][context_idx].tobytes()).hexdigest(),
        "source": "train histories only", "no_heldout_labels": True})
    events, _, _ = load_sample(args.users)
    counts = np.bincount([mapping[int(i)] for i in events[events.timestamp<pd.Timestamp(VAL_CUTOFF)].item_id], minlength=len(mapping)+2)
    rows = []; validation = {}
    validation["sft"] = evaluate(ref, data["valid"], monitoring, ref, counts, args.batch)[0]
    rows.append({"name": "sft", "seed": 42, "split": "valid", **validation["sft"]})
    for config in configs:
        args.seed = 42; set_seed(args.seed+100)
        policy = copy.deepcopy(ref)
        for p in policy.parameters(): p.requires_grad_(True)
        reward = ControlledReward(rms, priors, config["mode"]).to(device).eval()
        t = time.perf_counter(); rl_finetune(policy, ref, reward, data["train"], args, config, out)
        validation[config["name"]] = evaluate(policy, data["valid"], monitoring, ref, counts, args.batch)[0]
        rows.append({"name": config["name"], "seed": 42, "split": "valid", "train_seconds": time.perf_counter()-t, **validation[config["name"]]})
        pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
        print("ROBUST VALID", rows[-1], flush=True)
    baseline = validation["sft"]
    eligible = [n for n, m in validation.items() if m["ndcg@10"] >= .99*baseline["ndcg@10"]
                and m["max_item_user_fraction"] <= baseline["max_item_user_fraction"]+.02]
    selected = max(eligible, key=lambda name: validation[name]["predicted_watch@10"])
    save_json(out/"selection.json", {"selected": selected, "eligible": eligible, "validation": validation,
                                   "criterion": "validation constraints in study_protocol.json; selected before new test"})
    selected_config = next((c for c in configs if c["name"] == selected), None)
    for seed in [43,44]:
        if selected_config is None: break
        args.seed = seed; set_seed(seed+100)
        policy = copy.deepcopy(ref)
        for p in policy.parameters(): p.requires_grad_(True)
        config = dict(selected_config, name=f"selected_seed{seed}")
        reward = ControlledReward(rms, priors, config["mode"]).to(device).eval()
        rl_finetune(policy, ref, reward, data["train"], args, config, out)
        met = evaluate(policy, data["valid"], monitoring, ref, counts, args.batch)[0]
        rows.append({"name": selected, "seed": seed, "split": "valid", **met})
    # All test reads for new policies occur after selection and matched-seed fits.
    selected_per = []; comparisons = {}; base_per = None
    diagnostic = [c["name"] for c in configs if c["beta"] == .1]
    test_names = list(dict.fromkeys(["sft", selected]+diagnostic))
    for name in test_names:
        for seed in ([42,43,44] if name == selected and selected != "sft" else [42]):
            policy = copy.deepcopy(ref)
            if name != "sft":
                file = f"selected_seed{seed}" if seed != 42 else name
                policy.load_state_dict(torch.load(out/f"{file}.pt", map_location=device, weights_only=True))
            met, per, top = evaluate(policy, data["test"], monitoring, ref, counts, args.batch)
            rows.append({"name": name, "seed": seed, "split": "test", **met})
            per.to_csv(out/f"per_user_{name}_{seed}.csv", index=False)
            np.save(out/f"top_{name}_{seed}.npy", top)
            if name == "sft": base_per = per
            else: comparisons[f"{name}_{seed}"] = paired(base_per, per)
            if name == selected: selected_per.append(per)
            print("ROBUST TEST", rows[-1], flush=True)
    if selected_per:
        mean = pd.concat(selected_per).groupby("user_id", as_index=False)[["ndcg", "hit"]].mean()
        # float mean hits retained, not converted back to bool.
        j = base_per.merge(mean, on="user_id", suffixes=("_a", "_b"), validate="one_to_one")
        comparisons["selected_seed_averaged"] = {"ndcg": bootstrap(j.ndcg_b-j.ndcg_a), "recall": bootstrap(j.hit_b-j.hit_a.astype(float))}
    pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
    save_json(out/"paired_test_ci.json", comparisons)
    save_json(out/"runtime.json", {"total_seconds": time.perf_counter()-start, "device": str(device)})


if __name__ == "__main__":
    main()
