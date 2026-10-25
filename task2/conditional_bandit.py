"""Matched Monte Carlo/exact one-step reward gradients on frozen SFT top100.

The objective is conditional expected learned reward on a fixed reference
support, plus full masked-policy KL and CE. This is a bounded bandit surrogate,
not a full-catalog expected-reward guarantee or observed counterfactual reward.
"""
import argparse
import copy
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from catboost import CatBoostRegressor
from shared.data import ROOT,save_json,load_sample,VAL_CUTOFF
from shared.analyze import bootstrap
from task2.factorial import load_frozen
from task2.history_mask import MaskedHistoryPolicy
from task2.context_reward import ContextFeatures
from task2.context_rl import PointInTimeReward
from task2.run import set_seed,action_logits,evaluate


def conditional_objective(log_prob,rewards,method,group=4,samples=None,probabilities=None):
    """Negative exact reward or unbiased leave-one-out score-function gradient."""
    probabilities=log_prob.exp() if probabilities is None else probabilities
    if method=="exact":return -(probabilities*rewards).sum(1).mean()
    if samples is None:
        with torch.no_grad():samples=torch.multinomial(probabilities.detach(),group,replacement=True)
    if samples.shape[1]<2:raise ValueError("Leave-one-out requires at least two samples")
    values=rewards.gather(1,samples)
    loo=(values.sum(1,keepdim=True)-values)/(samples.shape[1]-1)
    return -(log_prob.gather(1,samples)*(values-loo).detach()).mean()


@torch.no_grad()
def cache_contexts(ref,teacher,train,indices):
    flat=indices.ravel();support=[];reference=[];mass=[]
    for start in range(0,len(flat),64):
        h=torch.as_tensor(train["x"][flat[start:start+64]])
        lp=action_logits(ref,h).log_softmax(-1);top=lp.topk(100,dim=-1).indices
        support.append(top);reference.append(lp);mass.append(lp.gather(1,top).logsumexp(-1).exp())
    support=torch.cat(support);reference=torch.cat(reference)
    rewards=torch.as_tensor(teacher.predict(train,flat,support.numpy()),dtype=torch.float32)
    return support,reference,rewards,float(torch.cat(mass).mean())


def train(policy,ref,data,args,indices,support,reference,rewards,method,out,on_step=None):
    opt=torch.optim.AdamW(policy.parameters(),lr=3e-5,weight_decay=.01);policy.eval();logs=[];start=time.perf_counter()
    for step,idx in enumerate(indices):
        lo=step*args.rl_batch;hi=lo+args.rl_batch
        x=torch.as_tensor(data["x"][idx]);y=torch.as_tensor(data["y"][idx]);ids=support[lo:hi];r=rewards[lo:hi]
        logits=action_logits(policy,x);lp=logits.log_softmax(-1);p=lp.exp()
        conditional=lp.gather(1,ids).log_softmax(-1);probs=conditional.exp()
        pg=logits.sum()*0 if method=="continued_sft" else conditional_objective(conditional,r,method,args.group,probabilities=probs)
        kl=(p*(lp-reference[lo:hi])).sum(1).mean();ce=F.cross_entropy(logits,y)
        loss=pg+.5*kl+.2*ce
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1);opt.step()
        if on_step is not None:on_step(step+1,policy)
        if step%20==0 or step+1==len(indices):
            row={"step":step+1,"conditional_expected_reward":float((probs*r).sum(1).mean().detach()),
                "kl":float(kl.detach()),"ce":float(ce.detach()),"seconds":time.perf_counter()-start}
            logs.append(row);print("CONDITIONAL BANDIT",method,args.seed,row,flush=True)
    pd.DataFrame(logs).to_csv(out/f"{method}_seed{args.seed}_training.csv",index=False)
    torch.save(policy.state_dict(),out/f"{method}_seed{args.seed}.pt")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="artifacts/task2_conditional_bandit")
    ap.add_argument("--full-reference",action="store_true");cli=ap.parse_args()
    out=ROOT/cli.out;out.mkdir(exist_ok=True);torch.set_num_threads(4)
    args,data,mapping,base,rms=load_frozen(ROOT/"artifacts/task2",torch.device("cpu"))
    chosen=json.loads((ROOT/"artifacts/task2_warmstart/selection.json").read_text())["selected"]
    if chosen!="original_sft":base.load_state_dict(torch.load(ROOT/f"artifacts/task2_warmstart/{chosen}/sft_best.pt",weights_only=True))
    reference_source=f"artifacts/task2_warmstart/{chosen}/sft_best.pt" if chosen!="original_sft" else "artifacts/task2/sft_best.pt"
    if cli.full_reference:
        variant=json.loads((ROOT/"artifacts/task2_full_data/selection.json").read_text())["selected"]["masked"]
        name=variant.removesuffix("_masked")
        reference_source={"capped_random":"artifacts/task2/sft_best.pt","capped_graph":"artifacts/task2_warmstart/binary/sft_best.pt"}.get(name,f"artifacts/task2_full_data/{name}/sft_best.pt")
        base.load_state_dict(torch.load(ROOT/reference_source,weights_only=True))
    ref=MaskedHistoryPolicy(base).eval()
    model=CatBoostRegressor().load_model(str(ROOT/"artifacts/task2_context_reward/reward.cbm"))
    events,items,users=load_sample(args.users)
    teacher=PointInTimeReward(ContextFeatures(events,items,users),model,{v:k for k,v in mapping.items()})
    counts=np.bincount([mapping[int(i)] for i in events[events.timestamp<pd.Timestamp(VAL_CUTOFF)].item_id],minlength=len(mapping)+2)
    save_json(out/"study_protocol.json",{"purpose":"Exploratory matched reward-gradient comparison after earlier test inspection",
        "reference":reference_source+" with max32 history mask","methods":["monte_carlo","exact"],
        "fine_tuning_train_contexts":len(data["train"]["y"]),
        "objective":"Conditional expected Context RM reward on fixed reference top100 +.5 full-policyKL +.2CE",
        "lr":3e-5,"steps":args.rl_steps,"batch":args.rl_batch,"group":args.group,"seeds":[42,43,44],
        "selection":"Max mean validation Context RM watch across three RL sampling/minibatch seeds, NDCG/Recall>=99% reference, max-item fraction<=ref+.02; SFT fallback",
        "paired_test_ndcg_margin":.005,"margin_note":"Exploratory absolute loss margin defined before new results; conditional user bootstrap, not online guarantee",
        "compute":"Common precomputed train context/reward table for both methods; exact100 actions vs MC4 per context; neural updates/batches/lr/KL/CE matched",
        "test":"All two prespecified methods reported after frozen validation selection, no test winner choice",
        "limits":"Conditional-support surrogate; output evaluated on full catalog; richer observational RM not causal simulator; SFT/RM fixed across seeds"})
    start=time.perf_counter();rows=[];cache_audit=[]
    def score(policy,split):
        met,per,top=evaluate(policy,data[split],rms,ref,counts,args.batch)
        met["context_rm_watch@10"]=float(teacher.predict(data[split],np.arange(len(data[split]["y"])),top).mean())
        return met,per,top
    baseline=score(ref,"valid")[0];rows.append({"method":"reference","seed":42,"split":"valid",**baseline})
    for seed in [42,43,44]:
        args.seed=seed;idx=np.random.default_rng(seed+100).integers(0,len(data["train"]["y"]),size=(args.rl_steps,args.rl_batch))
        t=time.perf_counter();support,reference,rewards,mass=cache_contexts(ref,teacher,data["train"],idx)
        cache_audit.append({"seed":seed,"contexts":idx.size,"reward_candidates":int(rewards.numel()),
            "indices_sha256":hashlib.sha256(idx.tobytes()).hexdigest(),"support_sha256":hashlib.sha256(support.numpy().tobytes()).hexdigest(),
            "reference_support_mass":mass,"cache_seconds":time.perf_counter()-t})
        save_json(out/"cache_audit.json",cache_audit)
        for method in ["monte_carlo","exact"]:
            set_seed(seed+100);policy=copy.deepcopy(ref)
            for param in policy.parameters():param.requires_grad_(True)
            t=time.perf_counter();train(policy,ref,data["train"],args,idx,support,reference,rewards,method,out)
            met,per,top=score(policy,"valid")
            rows.append({"method":method,"seed":seed,"split":"valid","train_seconds":time.perf_counter()-t,**met})
            pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
            print("CONDITIONAL VALID",rows[-1],flush=True)
        del support,reference,rewards
    valid=pd.DataFrame(rows).groupby("method").mean(numeric_only=True).to_dict(orient="index")
    eligible=[name for name,m in valid.items() if m["ndcg@10"]>=.99*baseline["ndcg@10"] and m["recall@10"]>=.99*baseline["recall@10"]
        and m["max_item_user_fraction"]<=baseline["max_item_user_fraction"]+.02]
    selected=max(eligible,key=lambda name:valid[name]["context_rm_watch@10"])
    save_json(out/"selection.json",{"selected":selected,"eligible":eligible,"validation_means":valid})
    per_methods={};ref_per=None
    for method in ["reference","monte_carlo","exact"]:
        per_methods[method]=[]
        for seed in ([42] if method=="reference" else [42,43,44]):
            policy=copy.deepcopy(ref)
            if method!="reference":policy.load_state_dict(torch.load(out/f"{method}_seed{seed}.pt",weights_only=True))
            met,per,top=score(policy,"test");rows.append({"method":method,"seed":seed,"split":"test",**met})
            per.to_csv(out/f"per_user_{method}_{seed}.csv",index=False);np.save(out/f"top_{method}_{seed}.npy",top)
            per_methods[method].append(per);print("CONDITIONAL TEST",rows[-1],flush=True)
            if method=="reference":ref_per=per
    effects={}
    for method in ["monte_carlo","exact"]:
        avg=pd.concat(per_methods[method]).groupby("user_id",as_index=False)[["ndcg","hit"]].mean()
        merged=ref_per.merge(avg,on="user_id",suffixes=("_a","_b"),validate="one_to_one")
        effects[method]={"ndcg":bootstrap(merged.ndcg_b-merged.ndcg_a),"recall":bootstrap(merged.hit_b-merged.hit_a.astype(float))}
        effects[method]["ndcg_ci_within_prespecified_loss_margin"]=effects[method]["ndcg"]["ci95"][0]>-.005
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False);save_json(out/"paired_test_ci.json",effects)
    save_json(out/"runtime.json",{"total_seconds":time.perf_counter()-start})


if __name__=="__main__":main()
