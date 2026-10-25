"""Separate inference constraints from RL: suppress objects in the known context.

KION source pairs are unique. Warm targets are absent from previous rows, so
this constraint matches the released-data task, not necessarily repeat listening.
The same <=32-ID support rule applies to both SFT and RL reference/policy.
"""
import argparse
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from shared.data import ROOT, save_json, load_sample, VAL_CUTOFF
from task2.factorial import load_frozen
from task2.model import RewardModel
from task2.run import set_seed, evaluate, rl_finetune, rollout_diagnostics
from task2.robust_reward import ControlledReward, paired
from shared.analyze import bootstrap


class MaskedHistoryPolicy(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model; self.config = model.config

    def forward(self, history):
        logits = self.model(history)
        # PAD/UNK are already forbidden actions; masking them again is harmless.
        return logits.scatter(1, history, -1e9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--source", default="artifacts/task2")
    ap.add_argument("--out", default="artifacts/task2_masked"); cli = ap.parse_args()
    out = ROOT/cli.out; out.mkdir(exist_ok=True); src = ROOT/cli.source
    torch.set_num_threads(4); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args, data, mapping, raw_ref, rms = load_frozen(src, device)
    ref = MaskedHistoryPolicy(raw_ref).to(device).eval()
    for split, part in data.items():
        assert not ((part["y"][:,None] == part["x"]).any(1)&(part["y"] >= 2)).any(), split
    robust = ROOT/"artifacts/task2_robust"
    priors = torch.as_tensor(np.load(robust/"item_priors.npy"), device=device)
    monitor = RewardModel(len(mapping)+2).to(device)
    monitor.load_state_dict(torch.load(robust/"monitor.pt", map_location=device, weights_only=True)); monitor.eval()
    for p in monitor.parameters(): p.requires_grad_(False)
    monitoring = [rms[0], monitor]
    configs = [{"name": f"masked_{mode}_beta{beta:g}", "mode": mode, "beta": beta,
                "ce": .2, "horizon": 1, "reward": "watch"}
        for mode, betas in [("watch", [.1,.5,1.]), ("residual_lcb", [.1,1.])] for beta in betas]
    configs.append({"name": "masked_continued_sft", "mode": "watch", "beta": .1,
                    "ce": .2, "horizon": 1, "reward": "watch", "pg_weight": 0.})
    save_json(out/"study_protocol.json", {"purpose": "Exploratory support constraint and matched constrained RL",
        "mask": "Only current max32 historical/generated IDs, same rule for SFT and RL",
        "configs": configs, "steps": args.rl_steps, "seed": 42, "selected_replications": [43,44],
        "selection": "Highest original RM watch on validation subject to >=99% masked-SFT NDCG and max-item fraction <= masked-SFT+.02; masked-SFT fallback",
        "control": "continued-SFT reported separately, not eligible for RL selection",
        "test": "Raw SFT, masked SFT, selected RL and fixed continued-SFT control only, after selection",
        "limits": "Already inspected test; source aggregate pairs are unique, exclusion need not suit repeat-consumption services; no target used during masking"})
    start = time.perf_counter(); rows = []
    events, _, _ = load_sample(args.users)
    counts = np.bincount([mapping[int(i)] for i in events[events.timestamp<pd.Timestamp(VAL_CUTOFF)].item_id], minlength=len(mapping)+2)
    validation = {"masked_sft": evaluate(ref, data["valid"], monitoring, ref, counts, args.batch)[0]}
    rows.append({"name": "masked_sft", "seed": 42, "split": "valid", **validation["masked_sft"]})
    for config in configs:
        args.seed = 42; set_seed(142); policy = copy.deepcopy(ref)
        for p in policy.parameters(): p.requires_grad_(True)
        reward = rms[0] if config["mode"] == "watch" else ControlledReward(rms, priors, config["mode"]).to(device).eval()
        t = time.perf_counter(); rl_finetune(policy, ref, reward, data["train"], args, config, out)
        met = evaluate(policy, data["valid"], monitoring, ref, counts, args.batch)[0]
        validation[config["name"]] = met
        rows.append({"name": config["name"], "seed": 42, "split": "valid", "train_seconds": time.perf_counter()-t, **met})
        pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
        print("MASKED VALID", rows[-1], flush=True)
    base = validation["masked_sft"]
    eligible = [name for name, m in validation.items() if name != "masked_continued_sft"
        and m["ndcg@10"] >= .99*base["ndcg@10"] and m["max_item_user_fraction"] <= base["max_item_user_fraction"]+.02]
    selected = max(eligible, key=lambda n: validation[n]["predicted_watch@10"])
    save_json(out/"selection.json", {"selected": selected, "eligible": eligible, "validation": validation})
    selected_config = next((c for c in configs if c["name"] == selected), None)
    if selected_config:
        for seed in [43,44]:
            args.seed = seed; set_seed(seed+100); policy = copy.deepcopy(ref)
            for p in policy.parameters(): p.requires_grad_(True)
            config = dict(selected_config, name=f"selected_seed{seed}")
            reward = rms[0] if config["mode"] == "watch" else ControlledReward(rms, priors, config["mode"]).to(device).eval()
            rl_finetune(policy, ref, reward, data["train"], args, config, out)
            met = evaluate(policy, data["valid"], monitoring, ref, counts, args.batch)[0]
            rows.append({"name": selected, "seed": seed, "split": "valid", **met})
    per_models = {}; selected_per = []; comparisons = {}
    names = list(dict.fromkeys(["raw_sft", "masked_sft", "masked_continued_sft", selected]))
    for name in names:
        for seed in ([42,43,44] if name == selected and selected_config else [42]):
            policy = raw_ref if name == "raw_sft" else copy.deepcopy(ref)
            if name not in ["raw_sft", "masked_sft"]:
                file = f"selected_seed{seed}" if name == selected and seed != 42 else name
                policy.load_state_dict(torch.load(out/f"{file}.pt", map_location=device, weights_only=True))
            met, per, top = evaluate(policy, data["test"], monitoring, ref, counts, args.batch)
            # Raw SFT has positive probability for masked actions: KL(raw||masked)
            # is infinite. Never report the finite -1e9 numerical approximation.
            if name == "raw_sft": met.pop("kl_to_reference", None)
            rows.append({"name": name, "seed": seed, "split": "test", **met})
            per.to_csv(out/f"per_user_{name}_{seed}.csv", index=False)
            np.save(out/f"top_{name}_{seed}.npy", top)
            per_models[name] = per
            if name == selected: selected_per.append(per)
            print("MASKED TEST", rows[-1], flush=True)
    comparisons["mask_only_vs_raw"] = paired(per_models["raw_sft"], per_models["masked_sft"])
    comparisons["continued_control_vs_masked_sft"] = paired(per_models["masked_sft"], per_models["masked_continued_sft"])
    mean = pd.concat(selected_per).groupby("user_id", as_index=False)[["ndcg","hit"]].mean()
    for label, baseline in [("selected_vs_masked_sft", per_models["masked_sft"]), ("selected_vs_raw_sft", per_models["raw_sft"])]:
        j = baseline.merge(mean, on="user_id", suffixes=("_a", "_b"), validate="one_to_one")
        comparisons[label] = {"ndcg": bootstrap(j.ndcg_b-j.ndcg_a), "recall": bootstrap(j.hit_b-j.hit_a.astype(float))}
    pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
    save_json(out/"paired_test_ci.json", comparisons)
    save_json(out/"runtime.json", {"total_seconds": time.perf_counter()-start})


if __name__ == "__main__": main()
