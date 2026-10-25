"""Prespecified continued-SFT/KL control, run after task2.run.

Reuses frozen checkpoints and exact examples. Not eligible for the original
five-variant validation selection; isolates policy-gradient from extra CE steps.
"""
import argparse
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
from shared.data import ROOT,VAL_CUTOFF,load_sample,save_json
from task2.model import GPT2Rec,RewardModel
from task2.run import evaluate,rl_finetune,rollout_diagnostics,set_seed

CONFIG={"name":"continued_sft_control","reward":"watch","beta":.1,"ce":.2,"horizon":1,"pg_weight":0.}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="artifacts/task2")
    cli=ap.parse_args(); out=ROOT/cli.out
    args=argparse.Namespace(**json.loads((out/"protocol.json").read_text())["args"])
    set_seed(args.seed+100); torch.set_num_threads(6)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data={}
    for split in ["train","valid","test"]:
        with np.load(out/f"examples_{split}.npz",allow_pickle=False) as z:
            data[split]={k:z[k] for k in z.files}
    mapping={int(k):int(v) for k,v in json.loads((out/"item_mapping.json").read_text()).items()}
    reference=GPT2Rec(**json.loads((out/"model_config.json").read_text())).to(device)
    reference.load_state_dict(torch.load(out/"sft_best.pt",map_location=device,weights_only=True)); reference.eval()
    for p in reference.parameters():p.requires_grad_(False)
    rms=[]
    for name in ["reward","auditor"]:
        rm=RewardModel(len(mapping)+2).to(device)
        rm.load_state_dict(torch.load(out/f"{name}.pt",map_location=device,weights_only=True));rm.eval()
        for p in rm.parameters():p.requires_grad_(False)
        rms.append(rm)
    events,items,_=load_sample(args.users)
    trained=events[events.timestamp<pd.Timestamp(VAL_CUTOFF)]
    counts=np.bincount([mapping[int(i)] for i in trained.item_id],minlength=len(mapping)+2)
    policy=copy.deepcopy(reference)
    for p in policy.parameters():p.requires_grad_(True)
    start=time.perf_counter(); set_seed(args.seed+100)
    rl_finetune(policy,reference,rms[0],data["train"],args,CONFIG,out)
    valid,_,_=evaluate(policy,data["valid"],rms,reference,counts,args.batch)
    test,per,top=evaluate(policy,data["test"],rms,reference,counts,args.batch)
    original=np.load(out/"top10_sft.npy",allow_pickle=False)
    test["overlap_with_sft@10"]=float(np.mean([len(set(a)&set(b))/10 for a,b in zip(top,original)]))
    meta=items.set_index("item_id"); inverse={v:k for k,v in mapping.items()}
    genres=np.array([str(meta.loc[inverse[t],"genre"]) if t in inverse else "unknown" for t in range(len(mapping)+2)])
    types=np.array([str(meta.loc[inverse[t],"content_type"]) if t in inverse else "unknown" for t in range(len(mapping)+2)])
    test["first_genre_diversity@10"]=float(np.mean([len(set(genres[r]))/10 for r in top]))
    test["film_share@10"]=float((types[top]=="film").mean())
    name=CONFIG["name"]
    per.to_csv(out/f"per_user_{name}.csv",index=False); np.save(out/f"top10_{name}.npy",top)
    metrics=pd.read_csv(out/"metrics.csv")
    metrics=metrics[metrics.model!=name]
    row={"model":name,"selected_on_validation":False,**test}
    pd.concat([metrics,pd.DataFrame([row])],ignore_index=True).to_csv(out/"metrics.csv",index=False)
    roll=json.loads((out/"rollout_diagnostics.json").read_text())
    roll[name]=rollout_diagnostics(policy,data["test"],rms,reference,args,mapping)
    save_json(out/"rollout_diagnostics.json",roll)
    metadata=json.loads((out/"recommendation_metadata.json").read_text())
    metadata[name]={"first_genre":{str(k):float(v) for k,v in pd.Series(genres[top].ravel()).value_counts(normalize=True).items()},
                    "content_type":{str(k):float(v) for k,v in pd.Series(types[top].ravel()).value_counts(normalize=True).items()}}
    save_json(out/"recommendation_metadata.json",metadata)
    groups=pd.cut(per.context_length,bins=[0,1,5,15,100000],labels=["1","2-5","6-15","16+"])
    strata=pd.read_csv(out/"history_length_metrics.csv")
    added=[{"model":name,"context_length":str(group),"users":len(g),"ndcg@10":float(g.ndcg.mean()),"recall@10":float(g.hit.mean())} for group,g in per.groupby(groups,observed=True)]
    pd.concat([strata[strata.model!=name],pd.DataFrame(added)],ignore_index=True).to_csv(out/"history_length_metrics.csv",index=False)
    save_json(out/"control_protocol.json",{"config":CONFIG,"valid_metrics":valid,"test_metrics":test,
        "seconds":time.perf_counter()-start,"selection_eligible":False,
        "purpose":"Same CE/KL/batches as one-step constrained RL, zero policy-gradient coefficient"})
    print("CONTROL",row,flush=True)


if __name__=="__main__":main()
