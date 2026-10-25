"""Exploratory five-item slate diagnostics on three frozen policies.

Compare one-shot top-5, greedy autoregressive decoding, and autoregressive
decoding with generated-item exclusion. No target is used during decoding.
"""
import argparse
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
from shared.data import ROOT,save_json
from shared.analyze import bootstrap
from task2.factorial import load_frozen
from task2.model import append_tokens
from task2.run import action_logits


def slate_metrics(slate,targets,users):
    matches=slate==targets[:,None];hit=matches.any(1)
    first=matches.argmax(1)+1
    ndcg=np.where(hit,1/np.log2(first+1),0.)
    unique=np.array([len(set(row)) for row in slate])
    per=pd.DataFrame({"user_id":users,"hit":hit,"ndcg":ndcg,"unique_items":unique})
    counts=np.bincount(slate.ravel());freq=counts[counts>0]/slate.size
    user_presence={}
    for row in slate:
        for item in set(row):user_presence[item]=user_presence.get(item,0)+1
    metrics={"hit@5":float(hit.mean()),"ndcg@5":float(ndcg.mean()),
             "mean_unique_items":float(unique.mean()),"duplicate_slot_fraction":float(1-unique.mean()/slate.shape[1]),
             "distinct_items":int((counts>0).sum()),"max_item_user_fraction":max(user_presence.values())/len(slate),
             "exposure_entropy":float(-(freq*np.log(freq)).sum())}
    return metrics,per


@torch.no_grad()
def decode(model,data,rms,mode,batch=128):
    model.eval();device=next(model.parameters()).device
    slates=[];step_rows=[]
    for start in range(0,len(data["y"]),batch):
        original=torch.as_tensor(data["x"][start:start+batch],device=device,dtype=torch.long)
        history=original.clone(); generated=[]
        fixed=action_logits(model,original).topk(5,dim=1).indices if mode=="one_shot" else None
        for step in range(5):
            if fixed is not None:action=fixed[:,step]
            else:
                logits=action_logits(model,history)
                if mode=="no_repeat" and generated:
                    logits.scatter_(1,torch.stack(generated,1),-1e9)
                action=logits.argmax(1)
            a=rms[0](history,action)[:,0];f=rms[0](original,action)[:,0];b=rms[1](history,action)[:,0]
            repeat=torch.stack(generated,1).eq(action[:,None]).any(1) if generated else torch.zeros(len(action),device=device,dtype=torch.bool)
            step_rows.extend({"step":step+1,"free_reward":float(u),"frozen_reward":float(v),
                "auditor_reward":float(w),"repeat":bool(r)} for u,v,w,r in zip(a.cpu(),f.cpu(),b.cpu(),repeat.cpu()))
            generated.append(action)
            if fixed is None:history=append_tokens(history,action)
        slates.append(torch.stack(generated,1).cpu().numpy())
    return np.concatenate(slates),pd.DataFrame(step_rows).groupby("step",as_index=False).mean()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source",default="artifacts/task2")
    ap.add_argument("--out",default="artifacts/task2_decoding");args=ap.parse_args()
    src=ROOT/args.source;out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(6);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training,data,mapping,ref,rms=load_frozen(src,device)
    save_json(out/"study_protocol.json",{"purpose":"Post-hoc decoding diagnostic after observing original collapse",
        "policies":["sft","rl_watch_unconstrained","rl_balanced_5step"],
        "modes":["one_shot","autoregressive","no_repeat"],"users":len(data["test"]["y"]),
        "length":5,"history_exclusion":False,"generated_item_exclusion":"only no_repeat mode",
        "target_relevance":"First occurrence of the single held-out next target; duplicates never count twice",
        "warning":"Generated items are hypothetical; no observed multi-step response or retention claim"})
    rows=[];steps=[];per_modes={};start=time.perf_counter()
    for name in ["sft","rl_watch_unconstrained","rl_balanced_5step"]:
        model=copy.deepcopy(ref)
        if name!="sft":model.load_state_dict(torch.load(src/f"{name}.pt",map_location=device,weights_only=True))
        for mode in ["one_shot","autoregressive","no_repeat"]:
            slate,trace=decode(model,data["test"],rms,mode,training.batch)
            if mode!="autoregressive":assert all(len(set(r))==5 for r in slate)
            metrics,per=slate_metrics(slate,data["test"]["y"],data["test"]["user"])
            rows.append({"policy":name,"mode":mode,**metrics});trace["policy"]=name;trace["mode"]=mode;steps.append(trace)
            per.to_csv(out/f"per_user_{name}_{mode}.csv",index=False);np.save(out/f"slate_{name}_{mode}.npy",slate)
            per_modes[(name,mode)]=per
            print("DECODING",rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.concat(steps,ignore_index=True).to_csv(out/"step_metrics.csv",index=False)
    ci={}
    for name in ["sft","rl_watch_unconstrained","rl_balanced_5step"]:
        a=per_modes[(name,"autoregressive")];b=per_modes[(name,"no_repeat")]
        assert a.user_id.equals(b.user_id)
        ci[name]={"no_repeat_minus_autoregressive_ndcg":bootstrap((b.ndcg-a.ndcg).to_numpy()),
                  "no_repeat_minus_autoregressive_hit":bootstrap((b.hit.astype(float)-a.hit.astype(float)).to_numpy())}
    save_json(out/"paired_ci.json",ci);save_json(out/"runtime.json",{"total_seconds":time.perf_counter()-start})
    print("DECODING COMPLETE",flush=True)


if __name__=="__main__":main()
