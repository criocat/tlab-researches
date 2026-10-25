"""Post-hoc 2x2 RL experiment: horizon x repeat penalty, fixed KL/CE.

Three matched RL sampling/minibatch seeds; SFT and RM weights stay fixed.
All four cells are reported; no new model is chosen by test performance.
"""
import argparse
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
from shared.data import ROOT,save_json,load_sample,VAL_CUTOFF
from task2.model import GPT2Rec,RewardModel
from task2.run import set_seed,rl_finetune,evaluate,rollout_diagnostics


def load_frozen(src,device):
    args=argparse.Namespace(**json.loads((src/"protocol.json").read_text())["args"])
    data={}
    for split in ["train","valid","test"]:
        with np.load(src/f"examples_{split}.npz",allow_pickle=False) as z:data[split]={k:z[k] for k in z.files}
    mapping={int(k):int(v) for k,v in json.loads((src/"item_mapping.json").read_text()).items()}
    ref=GPT2Rec(**json.loads((src/"model_config.json").read_text())).to(device)
    ref.load_state_dict(torch.load(src/"sft_best.pt",map_location=device,weights_only=True));ref.eval()
    models=[ref]
    for name in ["reward","auditor"]:
        rm=RewardModel(len(mapping)+2).to(device)
        rm.load_state_dict(torch.load(src/f"{name}.pt",map_location=device,weights_only=True));rm.eval();models.append(rm)
    for model in models:
        for p in model.parameters():p.requires_grad_(False)
    return args,data,mapping,ref,models[1:]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source",default="artifacts/task2")
    ap.add_argument("--out",default="artifacts/task2_factorial");cli=ap.parse_args()
    src=ROOT/cli.source;out=ROOT/cli.out;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(6);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args,data,mapping,ref,rms=load_frozen(src,device)
    save_json(out/"study_protocol.json",{"purpose":"Exploratory factorial analysis after original test inspection",
        "horizons":[1,5],"repeat_penalties":[0,.2],"rl_seeds":[42,43,44],"beta":.1,"ce":.2,
        "steps":args.rl_steps,"base_sft_and_reward":"frozen original CPU checkpoints",
        "no_new_selection":True,"warning":"Seeds vary RL sampling/minibatches, not SFT/RM initialization"})
    events,_,_=load_sample(args.users)
    counts=np.bincount([mapping[int(i)] for i in events[events.timestamp<pd.Timestamp(VAL_CUTOFF)].item_id],minlength=len(mapping)+2)
    rows=[];start=time.perf_counter()
    for seed in [42,43,44]:
        args.seed=seed
        for horizon in [1,5]:
            for penalty in [False,True]:
                name=f"h{horizon}_repeat{int(penalty)}_seed{seed}";cell=out/name;cell.mkdir(exist_ok=True)
                config={"name":name,"reward":"balanced" if penalty else "watch","horizon":horizon,"beta":.1,"ce":.2}
                set_seed(seed+100);policy=copy.deepcopy(ref)
                for p in policy.parameters():p.requires_grad_(True)
                t=time.perf_counter();rl_finetune(policy,ref,rms[0],data["train"],args,config,cell)
                train_seconds=time.perf_counter()-t
                for split in ["valid","test"]:
                    metrics,per,top=evaluate(policy,data[split],rms,ref,counts,args.batch)
                    per.to_csv(cell/f"per_user_{split}.csv",index=False)
                    rows.append({"name":name,"seed":seed,"horizon":horizon,"penalty":.2 if penalty else 0.,
                                 "split":split,"train_seconds":train_seconds,**metrics})
                roll=rollout_diagnostics(policy,data["test"],rms,ref,args,mapping)
                save_json(cell/"rollout.json",roll)
                pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
                print("FACTORIAL",name,rows[-1],flush=True)
                del policy
    save_json(out/"runtime.json",{"total_seconds":time.perf_counter()-start})
    print("FACTORIAL COMPLETE",flush=True)


if __name__=="__main__":main()
