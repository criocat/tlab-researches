"""Validation-selected stopping time with matched continued-SFT controls.

All paths are fixed before new results; the original test has already been
inspected. This is exploratory, not a new untouched confirmation experiment.
"""
import copy
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from shared.data import ROOT,load_sample,save_json,VAL_CUTOFF
from shared.analyze import bootstrap
from task2.factorial import load_frozen
from task2.history_mask import MaskedHistoryPolicy
from task2.context_reward import ContextFeatures
from task2.context_rl import PointInTimeReward
from task2.conditional_bandit import cache_contexts,train
from task2.run import set_seed,evaluate


def digest(array):return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    out=ROOT/"artifacts/task2_validation_path";out.mkdir(exist_ok=True)
    checkpoints=out/"checkpoints";checkpoints.mkdir(exist_ok=True);torch.set_num_threads(4)
    args,data,mapping,base,rms=load_frozen(ROOT/"artifacts/task2",torch.device("cpu"))
    original_args_steps=args.rl_steps;args.rl_steps=400
    full=json.loads((ROOT/"artifacts/task2_full_data/selection.json").read_text())["selected"]["masked"]
    assert full=="full_graph_masked","Revisit the recorded reference if upstream selection changes"
    base.load_state_dict(torch.load(ROOT/"artifacts/task2_full_data/full_graph/sft_best.pt",weights_only=True))
    ref=MaskedHistoryPolicy(base).eval();steps=[20,50,100,150,200,400]
    protocol={"purpose":"Exploratory stopping-time selection under the original strict validation guard",
        "reference":"Frozen full_graph_masked CPU SFT, one SFT/RM seed",
        "methods":["monte_carlo","exact"],"rl_seeds":[42,43,44],"stopping_steps":steps,
        "control":"Continued SFT with identical batches/optimizer/CE/KL and zero reward gradient; not eligible for RL selection",
        "lr":3e-5,"kl":.5,"ce":.2,"group":4,"batch":args.rl_batch,
        "support":"Same frozen reference top100; identical teacher cache across all three methods",
        "selection":"Max mean validation Context RM watch over method/step, NDCG and Recall>=99% reference, max-item<=reference+.02; reference fallback",
        "control_step":"Same selected stopping time, or prespecified200 if reference fallback",
        "test":"Reference, selected RL path and matched control only; no test path selection",
        "prefix_check":"At200, both RL methods must be parameter-identical to original fixed200 experiment at every seed",
        "limits":"Original test already inspected; validation reused for extra12-setting search; conditional user CIs do not cover selection uncertainty or SFT/RM training variability"}
    save_json(out/"study_protocol.json",protocol);start=time.perf_counter()
    events,items,users=load_sample(args.users)
    tree=CatBoostRegressor().load_model(str(ROOT/"artifacts/task2_context_reward/reward.cbm"))
    teacher=PointInTimeReward(ContextFeatures(events,items,users),tree,{v:k for k,v in mapping.items()})
    counts=np.bincount([mapping[int(i)] for i in events[events.timestamp<pd.Timestamp(VAL_CUTOFF)].item_id],minlength=len(mapping)+2)
    def score(policy,split):
        met,per,top=evaluate(policy,data[split],reference=ref,counts=counts,batch=args.batch)
        met["context_rm_watch@10"]=float(teacher.predict(data[split],np.arange(len(data[split]["y"])),top).mean())
        return met,per,top
    baseline,_,_=score(ref,"valid")
    rows=[{"method":"reference","seed":42,"step":0,"split":"valid",**baseline}]
    old_cache=json.loads((ROOT/"artifacts/task2_conditional_full_reference/cache_audit.json").read_text())
    prefix=[];caches=[]
    for seed in [42,43,44]:
        args.seed=seed
        indices=np.random.default_rng(seed+100).integers(0,len(data["train"]["y"]),size=(args.rl_steps,args.rl_batch))
        support,reference,rewards,mass=cache_contexts(ref,teacher,data["train"],indices)
        previous=next(r for r in old_cache if r["seed"]==seed)
        assert digest(indices[:original_args_steps])==previous["indices_sha256"]
        assert digest(support[:original_args_steps*args.rl_batch].numpy())==previous["support_sha256"]
        caches.append({"seed":seed,"contexts":int(indices.size),"support_mass":mass,
            "indices_sha256":digest(indices),"support_sha256":digest(support.numpy()),"original200_prefix_identical":True})
        save_json(out/"cache_audit.json",caches)
        for method in ["monte_carlo","exact","continued_sft"]:
            set_seed(seed+100);policy=copy.deepcopy(ref)
            for p in policy.parameters():p.requires_grad_(True)
            def checkpoint(step,model):
                if step not in steps:return
                torch.save(model.state_dict(),checkpoints/f"{method}_seed{seed}_step{step}.pt")
                if step==200 and method!="continued_sft":
                    old=torch.load(ROOT/f"artifacts/task2_conditional_full_reference/{method}_seed{seed}.pt",weights_only=True)
                    identical=all(torch.equal(old[k],v) for k,v in model.state_dict().items());assert identical
                    prefix.append({"method":method,"seed":seed,"bitwise_parameter_identity":identical})
                    save_json(out/"prefix_identity.json",prefix)
                met,per,top=score(model,"valid")
                per.to_csv(out/f"per_user_valid_{method}_{seed}_step{step}.csv",index=False)
                rows.append({"method":method,"seed":seed,"step":step,"split":"valid",**met})
                pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
                print("PATH VALID",method,seed,step,met,flush=True)
            train(policy,ref,data["train"],args,indices,support,reference,rewards,method,out,on_step=checkpoint)
        del support,reference,rewards
    valid=pd.DataFrame(rows).groupby(["method","step"],as_index=False).mean(numeric_only=True)
    valid.to_csv(out/"validation_means.csv",index=False)
    allowed=valid[(valid.method!="continued_sft")&(valid["ndcg@10"]>=.99*baseline["ndcg@10"])
        &(valid["recall@10"]>=.99*baseline["recall@10"])
        &(valid.max_item_user_fraction<=baseline["max_item_user_fraction"]+.02)]
    winner=allowed.loc[allowed["context_rm_watch@10"].idxmax()]
    selected={"method":str(winner.method),"step":int(winner.step)}
    control_step=selected["step"] if selected["step"] else 200
    save_json(out/"selection.json",{"selected":selected,"eligible":allowed[["method","step"]].to_dict("records"),
        "control_step":control_step,"criterion":protocol["selection"]})
    selection_hash=hashlib.sha256((out/"selection.json").read_bytes()).hexdigest()
    names=[("reference",0)]
    if selected["method"]!="reference":names.append((selected["method"],selected["step"]))
    names.append(("continued_sft",control_step));per_methods={}
    for method,step in names:
        fits=[]
        for seed in ([42] if method=="reference" else [42,43,44]):
            policy=copy.deepcopy(ref)
            if method!="reference":policy.load_state_dict(torch.load(checkpoints/f"{method}_seed{seed}_step{step}.pt",weights_only=True))
            met,per,top=score(policy,"test")
            per.to_csv(out/f"per_user_test_{method}_{seed}.csv",index=False);np.save(out/f"top_test_{method}_{seed}.npy",top)
            rows.append({"method":method,"seed":seed,"step":step,"split":"test",**met});fits.append(per)
        per_methods[method]=pd.concat(fits).groupby("user_id",as_index=False)[["ndcg","hit"]].mean()
    effects={}
    for name in per_methods:
        if name=="reference":continue
        j=per_methods["reference"].merge(per_methods[name],on="user_id",suffixes=("_a","_b"),validate="one_to_one")
        effects[name]={"ndcg":bootstrap(j.ndcg_b-j.ndcg_a),"recall":bootstrap(j.hit_b-j.hit_a)}
    if selected["method"]!="reference":
        j=per_methods["continued_sft"].merge(per_methods[selected["method"]],on="user_id",suffixes=("_a","_b"),validate="one_to_one")
        effects["selected_rl_minus_matched_control"]={"ndcg":bootstrap(j.ndcg_b-j.ndcg_a),"recall":bootstrap(j.hit_b-j.hit_a)}
    assert selection_hash==hashlib.sha256((out/"selection.json").read_bytes()).hexdigest()
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    save_json(out/"paired_test_ci.json",effects)
    save_json(out/"runtime.json",{"total_seconds":time.perf_counter()-start,"selection_sha256_before_test":selection_hash,
        "prefix_model_checks":len(prefix),"all_original200_parameters_bitwise_identical":all(r["bitwise_parameter_identity"] for r in prefix)})
    print("PATH SELECTED",selected,"CONTROL STEP",control_step,flush=True)
    print(pd.DataFrame(rows).query("split == 'test'").groupby("method").mean(numeric_only=True).to_string(),flush=True)


if __name__=="__main__":main()
