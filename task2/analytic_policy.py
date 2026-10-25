"""A one-step KL-optimal reranking control on fixed SFT top-100 support.

For q(a|s) proportional to p_ref(a|s) exp(alpha*r(s,a)), the conditional
one-step objective E_q[r] - KL(q||p_ref)/alpha is solved exactly. This is an
inference-time comparator, not an additional trained RL policy. The reference
is restricted to its own top-100, without injecting targets into candidates.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from shared.data import ROOT, save_json, load_sample, VAL_CUTOFF
from task2.factorial import load_frozen
from task2.model import RewardModel
from task2.run import action_logits, evaluate
from task2.robust_reward import ControlledReward, paired


class CachedPolicy(nn.Module):
    """Deterministic history lookup used only for evaluation of frozen logits."""
    def __init__(self, histories, logits, config):
        super().__init__()
        self.register_parameter("device_marker", nn.Parameter(torch.zeros((), device=logits.device), requires_grad=False))
        self.logits = logits
        self.config = config
        self.rows = {}
        for i, row in enumerate(histories):
            key = row.tobytes()
            if key in self.rows:
                torch.testing.assert_close(logits[i], logits[self.rows[key]], rtol=0, atol=0)
            self.rows[key] = i

    def forward(self, x):
        ids = [self.rows[row.tobytes()] for row in x.detach().cpu().numpy()]
        return self.logits[ids].clone()


@torch.no_grad()
def candidates(ref, rms, priors, data, support=100, batch=64):
    device = next(ref.parameters()).device
    tokens, reference, rewards, mass = [], [], {mode: [] for mode in ["watch", "mean", "lcb", "residual_lcb"]}, []
    for start in range(0, len(data["x"]), batch):
        x = torch.as_tensor(data["x"][start:start+batch], device=device)
        logits = action_logits(ref, x)
        values, ids = logits.topk(support, dim=-1)
        tokens.append(ids); reference.append(values)
        mass.append((values.logsumexp(-1)-logits.logsumexp(-1)).exp())
        rewards["watch"].append(rms[0](x, ids)[...,0])
        for mode in ["mean", "lcb", "residual_lcb"]:
            rewards[mode].append(ControlledReward(rms, priors, mode)(x, ids)[...,0])
    return torch.cat(tokens), torch.cat(reference), {k: torch.cat(v) for k,v in rewards.items()}, torch.cat(mass)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--source", default="artifacts/task2")
    ap.add_argument("--out", default="artifacts/task2_analytic"); cli = ap.parse_args()
    src = ROOT/cli.source; out = ROOT/cli.out; out.mkdir(exist_ok=True)
    torch.set_num_threads(4); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args, data, mapping, ref, rms = load_frozen(src, device)
    robust = ROOT/"artifacts/task2_robust"
    priors = torch.as_tensor(np.load(robust/"item_priors.npy"), device=device)
    monitor = RewardModel(len(mapping)+2).to(device)
    monitor.load_state_dict(torch.load(robust/"monitor.pt", map_location=device, weights_only=True)); monitor.eval()
    for p in monitor.parameters(): p.requires_grad_(False)
    monitoring = [rms[0], monitor]
    grid = [{"mode": mode, "alpha": alpha} for mode in ["watch", "mean", "lcb", "residual_lcb"] for alpha in [.25,.5,1.,2.,4.,8.]]
    save_json(out/"study_protocol.json", {"purpose": "Exploratory inference-time control after original test inspection",
        "grid": grid, "support": 100, "support_policy": "Frozen SFT top100, no target injection",
        "objective": "Exact conditional one-step expected RM reward minus KL/alpha; no CE, no training",
        "selection": "Max validation original predicted_watch with NDCG >=99% SFT and max-item fraction <= SFT+.02; SFT fallback",
        "evaluation": "Original full-catalog top10 metrics, cold targets retained as misses; truncated support cannot admit target",
        "kl_warning": "Reported KL is to full SFT and includes support truncation; conditional KL reported separately",
        "test": "Only original SFT and validation-selected reranker; no test grid selection"})
    events, _, _ = load_sample(args.users)
    counts = np.bincount([mapping[int(i)] for i in events[events.timestamp<pd.Timestamp(VAL_CUTOFF)].item_id], minlength=len(mapping)+2)
    rows = []; start = time.perf_counter(); selection = None; per_test = {}
    for split in ["valid", "test"]:
        ids, values, scores, support_mass = candidates(ref, rms, priors, data[split])
        save_json(out/f"support_{split}.json", {"mean_reference_probability": float(support_mass.mean()),
            "minimum_reference_probability": float(support_mass.min()), "n": len(support_mass)})
        settings = [{"mode": "sft", "alpha": 0.}]+(grid if split == "valid" else [selection] if selection["mode"] != "sft" else [])
        validation = []
        for setting in settings:
            name = setting["mode"]; alpha = setting["alpha"]
            if name == "sft": policy = ref; conditional_kl = 0.
            else:
                adjusted = values+alpha*scores[name]
                logits = torch.full((len(ids), ref.config["vocab"]), -1e9, device=device)
                logits.scatter_(1, ids, adjusted)
                policy = CachedPolicy(data[split]["x"], logits, ref.config)
                logq = adjusted.log_softmax(-1); logref = values.log_softmax(-1)
                conditional_kl = float((logq.exp()*(logq-logref)).sum(1).mean())
            met, per, top = evaluate(policy, data[split], monitoring, ref, counts, args.batch)
            if name != "sft":
                # Outside-support actions have exactly zero probability in the
                # mathematical policy. A -1e9 mask is only numerical machinery;
                # its huge finite NLL must not be reported as a model likelihood.
                met.pop("warm_full_catalog_nll", None)
                warm = data[split]["y"] >= 2
                inside = (ids.cpu().numpy() == data[split]["y"][:,None]).any(1)
                met["zero_support_warm_target_fraction"] = float((~inside[warm]).mean())
            row = {"mode": name, "alpha": alpha, "split": split, "conditional_support_kl": conditional_kl, **met}
            rows.append(row); validation.append(row)
            print("ANALYTIC", row, flush=True)
            if split == "test":
                per.to_csv(out/f"per_user_{name}.csv", index=False)
                np.save(out/f"top_{name}.npy", top); per_test[name] = per
            del policy
        if split == "valid":
            base = validation[0]
            eligible = [r for r in validation if r["ndcg@10"] >= .99*base["ndcg@10"]
                        and r["max_item_user_fraction"] <= base["max_item_user_fraction"]+.02]
            best = max(eligible, key=lambda r: r["predicted_watch@10"])
            selection = {"mode": best["mode"], "alpha": best["alpha"]}
            save_json(out/"selection.json", {"selected": selection, "eligible": [{"mode": r["mode"], "alpha": r["alpha"]} for r in eligible],
                                             "validation": validation})
            pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
    if selection["mode"] != "sft":
        save_json(out/"paired_test_ci.json", paired(per_test["sft"], per_test[selection["mode"]]))
    pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
    save_json(out/"runtime.json", {"total_seconds": time.perf_counter()-start})


if __name__ == "__main__": main()
