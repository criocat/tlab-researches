"""One-step RL with the richer, point-in-time CatBoost response model.

Reward sees the original user's past response statistics at the original day.
Every candidate is scored using that day, without its future observed response.
"""
import argparse
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from catboost import CatBoostRegressor
from shared.data import ROOT, save_json, load_sample, VAL_CUTOFF
from task2.context_reward import ContextFeatures
from task2.factorial import load_frozen
from task2.history_mask import MaskedHistoryPolicy
from task2.run import set_seed, tensor_batch, action_logits, evaluate
from task2.robust_reward import paired
from shared.analyze import bootstrap


class PointInTimeReward:
    def __init__(self, features, model, inverse):
        self.features = features; self.model = model; self.inverse = inverse

    def predict(self, data, indices, tokens):
        indices = np.asarray(indices); tokens = np.asarray(tokens)
        shape = tokens.shape
        if tokens.ndim == 2: indices = np.repeat(indices, tokens.shape[1])
        q = pd.DataFrame({"user_id": data["user"][indices],
            "item_id": [self.inverse[int(token)] for token in tokens.ravel()],
            "timestamp": pd.to_datetime(data["date"][indices]).astype("datetime64[ns]")})
        x = self.features.transform(q)
        return np.clip(self.model.predict(x), 0, 1).reshape(shape)


def finetune(policy, ref, reward, train, args, beta, name, out):
    device = next(policy.parameters()).device; policy.eval()
    opt = torch.optim.AdamW(policy.parameters(), lr=args.rl_lr, weight_decay=.01)
    rng = np.random.default_rng(args.seed+100); logs = []; start = time.perf_counter()
    for step in range(args.rl_steps):
        idx = rng.integers(0, len(train["y"]), size=args.rl_batch)
        original, y, _, _ = tensor_batch(train, idx, device)
        history = original.repeat_interleave(args.group, 0)
        logits = action_logits(policy, history); logp = logits.log_softmax(-1); p = logp.exp()
        with torch.no_grad():
            action = torch.multinomial(p.detach(), 1).squeeze(1)
            r = torch.as_tensor(reward.predict(train, np.repeat(idx, args.group), action.cpu().numpy()), device=device, dtype=logits.dtype)
            logref = action_logits(ref, history).log_softmax(-1)
        returns = r.reshape(args.rl_batch, args.group)
        advantage = returns-(returns.sum(1, keepdim=True)-returns)/(args.group-1)
        chosen = logp.gather(1, action[:,None]).reshape_as(returns)
        pg = -(chosen*advantage.detach()).mean()
        kl = (p*(logp-logref)).sum(1).mean()
        ce = F.cross_entropy(action_logits(policy, original), y)
        loss = pg+beta*kl+.2*ce
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(),1); opt.step()
        if step % 20 == 0 or step+1 == args.rl_steps:
            row = {"step": step+1, "reward": float(r.mean()), "kl": float(kl.detach()), "ce": float(ce.detach()), "seconds": time.perf_counter()-start}
            logs.append(row); print("CONTEXT RL", name, row, flush=True)
    pd.DataFrame(logs).to_csv(out/f"{name}_training.csv", index=False)
    torch.save(policy.state_dict(), out/f"{name}.pt")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="artifacts/task2_context_rl")
    ap.add_argument("--warmstart-selected", action="store_true"); cli = ap.parse_args()
    out = ROOT/cli.out; out.mkdir(exist_ok=True); src = ROOT/"artifacts/task2"
    args, data, mapping, base, rms = load_frozen(src, torch.device("cpu")); torch.set_num_threads(4)
    policy_source = "Original frozen SFT"
    if cli.warmstart_selected:
        chosen = json.loads((ROOT/"artifacts/task2_warmstart/selection.json").read_text())["selected"]
        if chosen != "original_sft":
            checkpoint = ROOT/"artifacts/task2_warmstart"/chosen/"sft_best.pt"
            base.load_state_dict(torch.load(checkpoint, weights_only=True))
            policy_source = str(checkpoint.relative_to(ROOT))
    ref = MaskedHistoryPolicy(base).eval()
    rm = CatBoostRegressor().load_model(str(ROOT/"artifacts/task2_context_reward/reward.cbm"))
    events, items, users = load_sample(args.users)
    reward = PointInTimeReward(ContextFeatures(events, items, users), rm, {v:k for k,v in mapping.items()})
    counts = np.bincount([mapping[int(i)] for i in events[events.timestamp<pd.Timestamp(VAL_CUTOFF)].item_id], minlength=len(mapping)+2)
    save_json(out/"study_protocol.json", {"purpose": "Exploratory RL with richer observed-response reward",
        "reference": policy_source+" with max32-ID history mask", "betas": [.1,.5,1.], "ce": .2, "steps": args.rl_steps,
        "seeds": [42,43,44], "horizon": 1, "reward": "Frozen selected task2_context_reward CatBoost; richer full-history state declared",
        "selection": "Max validation new RM watch, NDCG>=99% masked SFT, Recall>=99% masked SFT, max-item fraction<=masked SFT+.02; SFT fallback",
        "test": "Only frozen selected policy and masked SFT after validation; matched seeds for selected RL",
        "caution": "Observed RM R2 can improve through user-level effects that actions cannot control; no online effect inference"})
    start = time.perf_counter(); rows = []; validation = {}
    def score(policy, split):
        met, per, top = evaluate(policy, data[split], rms, ref, counts, args.batch)
        met["context_rm_watch@10"] = float(reward.predict(data[split], np.arange(len(data[split]["y"])), top).mean())
        return met, per, top
    validation["masked_sft"] = score(ref,"valid")[0]
    rows.append({"name":"masked_sft","seed":42,"split":"valid",**validation["masked_sft"]})
    for beta in [.1,.5,1.]:
        args.seed = 42; set_seed(142); policy = copy.deepcopy(ref)
        for p in policy.parameters(): p.requires_grad_(True)
        name = f"context_beta{beta:g}"; finetune(policy,ref,reward,data["train"],args,beta,name,out)
        validation[name] = score(policy,"valid")[0]
        rows.append({"name":name,"seed":42,"split":"valid",**validation[name]})
        pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
        print("CONTEXT RL VALID",rows[-1],flush=True)
    baseline = validation["masked_sft"]
    eligible = [n for n,m in validation.items() if m["ndcg@10"]>=.99*baseline["ndcg@10"]
        and m["recall@10"]>=.99*baseline["recall@10"] and m["max_item_user_fraction"]<=baseline["max_item_user_fraction"]+.02]
    selected = max(eligible,key=lambda n:validation[n]["context_rm_watch@10"])
    save_json(out/"selection.json",{"selected":selected,"eligible":eligible,"validation":validation})
    if selected != "masked_sft":
        beta = float(selected.removeprefix("context_beta"))
        for seed in [43,44]:
            args.seed = seed; set_seed(seed+100); policy = copy.deepcopy(ref)
            for p in policy.parameters(): p.requires_grad_(True)
            finetune(policy,ref,reward,data["train"],args,beta,f"selected_seed{seed}",out)
    baseline_per = None; selected_parts = []
    for name in list(dict.fromkeys(["masked_sft",selected])):
        for seed in ([42,43,44] if name != "masked_sft" else [42]):
            policy = copy.deepcopy(ref)
            if name != "masked_sft": policy.load_state_dict(torch.load(out/f'{name if seed==42 else "selected_seed"+str(seed)}.pt',weights_only=True))
            met,per,top = score(policy,"test")
            rows.append({"name":name,"seed":seed,"split":"test",**met})
            per.to_csv(out/f"per_user_{name}_{seed}.csv",index=False); np.save(out/f"top_{name}_{seed}.npy",top)
            if name == "masked_sft": baseline_per = per
            if name == selected: selected_parts.append(per)
            print("CONTEXT RL TEST",rows[-1],flush=True)
    avg = pd.concat(selected_parts).groupby("user_id",as_index=False)[["ndcg","hit"]].mean()
    j=baseline_per.merge(avg,on="user_id",suffixes=("_a","_b"),validate="one_to_one")
    save_json(out/"paired_test_ci.json",{"ndcg":bootstrap(j.ndcg_b-j.ndcg_a),"recall":bootstrap(j.hit_b-j.hit_a.astype(float))})
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    save_json(out/"runtime.json",{"total_seconds":time.perf_counter()-start})


if __name__ == "__main__": main()
