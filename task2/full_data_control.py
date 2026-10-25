"""Control SFT transition subsampling and the extra graph pretraining stage.

Original validation/test arrays are loaded unchanged. make_examples with a
larger cap changes RNG consumption, so its newly generated eval arrays MUST
be discarded. Only its complete training split is used here.
"""
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits
from shared.data import ROOT,load_sample,save_json,VAL_CUTOFF
from task2.data import make_examples
from task2.factorial import load_frozen
from task2.model import GPT2Rec
from task2.history_mask import MaskedHistoryPolicy
from task2.run import set_seed,train_policy,evaluate
from task2.robust_reward import paired


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="artifacts/task2_full_data");cli=ap.parse_args()
    out=ROOT/cli.out;out.mkdir(exist_ok=True);torch.set_num_threads(6)
    args,data,mapping,reference,rms=load_frozen(ROOT/"artifacts/task2",torch.device("cpu"))
    save_json(out/"study_protocol.json",{"purpose":"Exploratory control for supervised transition cap and graph initializer information",
        "methods":["capped_random","capped_graph","full_random","full_graph"],"new_sft_seed":42,"epochs":args.epochs,
        "full_training":"Every eligible prior-day transition, unchanged histories and vocabulary",
        "heldout":"Exact original npz arrays; discard make_examples eval arrays with changed RNG state",
        "within_full_comparison":"Same full supervised examples, epochs, optimizer and architecture; graph has additional all-train-edge unsupervised SVD stage",
        "across_caps_limit":"Same epochs but more SGD updates with full data; not an isolated data-only effect",
        "selection":"Per-run checkpoint by raw validation NDCG, then best raw/masked variant by validation only, before new test scoring",
        "test":"All prespecified models and inference masks reported; no test winner selection",
        "limits":"Previously inspected KION test; one SFT seed; singleton/first-day edges also enter SVD even without next-day training targets"})
    start=time.perf_counter();events,_,_=load_sample(args.users)
    generated,new_mapping,counts,audit=make_examples(events,args.max_len,max_train=len(events)+1,max_eval=args.max_eval,seed=42)
    assert new_mapping==mapping
    full=generated["train"];old=data["train"]
    index=pd.MultiIndex.from_arrays([full["user"],full["date"],full["y"]]);old_index=pd.MultiIndex.from_arrays([old["user"],old["date"],old["y"]])
    assert index.is_unique;positions=index.get_indexer(old_index);assert (positions>=0).all()
    for key in old:np.testing.assert_array_equal(full[key][positions],old[key])
    np.savez_compressed(out/"examples_train_full.npz",**full)
    save_json(out/"data_audit.json",{"original_train":len(old["y"]),"full_train":len(full["y"]),
        "old_train_is_exact_subset":True,"validation_sha256":hashlib.sha256(data["valid"]["x"].tobytes()+data["valid"]["y"].tobytes()).hexdigest(),
        "test_sha256":hashlib.sha256(data["test"]["x"].tobytes()+data["test"]["y"].tobytes()).hexdigest(),
        "unchanged_eval_users":{s:len(data[s]["y"]) for s in ["valid","test"]},"raw_example_audit":audit})
    past=events[events.timestamp<pd.Timestamp(VAL_CUTOFF)];u=pd.Index(np.sort(past.user_id.unique()))
    columns=np.array([mapping[int(i)]-2 for i in past.item_id])
    graph=sparse.csr_matrix((np.ones(len(past),dtype=np.float32),(u.get_indexer(past.user_id),columns)),shape=(len(u),len(mapping)))
    graph.data[:]=1
    with threadpool_limits(limits=4):svd=TruncatedSVD(n_components=args.dim,n_iter=7,random_state=42).fit(graph)
    factors=svd.components_.T*np.sqrt(svd.singular_values_)[None,:];factors=(factors*(.02/np.sqrt(np.mean(factors**2)))).astype(np.float32)
    factor_hash=hashlib.sha256(factors.tobytes()).hexdigest()
    assert factor_hash==json.loads((ROOT/"artifacts/task2_warmstart/binary/initialization.json").read_text())["factor_sha256"]
    save_json(out/"initializer_audit.json",{"same_as_capped_graph_hash":factor_hash,"edges":int(graph.nnz),
        "max_graph_timestamp":str(past.timestamp.max()),"cutoff":VAL_CUTOFF})
    models={"capped_random":reference}
    capped=GPT2Rec(**reference.config);capped.load_state_dict(torch.load(ROOT/"artifacts/task2_warmstart/binary/sft_best.pt",weights_only=True));models["capped_graph"]=capped
    rows=[];validation={}
    for name in ["capped_random","capped_graph","full_random","full_graph"]:
        if name.startswith("full_"):
            cell=out/name;cell.mkdir(exist_ok=True);set_seed(42);model=GPT2Rec(**reference.config)
            if name=="full_graph":
                with torch.no_grad():model.item.weight[2:].copy_(torch.as_tensor(factors))
            t=time.perf_counter();train_policy(model,full,data["valid"],args,cell)
            models[name]=model;print("FULL DATA TRAIN",name,time.perf_counter()-t,flush=True)
        for masked in [False,True]:
            variant=name+("_masked" if masked else "_raw");policy=MaskedHistoryPolicy(models[name]) if masked else models[name]
            met,per,top=evaluate(policy,data["valid"],rms,reference,counts,args.batch)
            rows.append({"model":name,"masked":masked,"variant":variant,"split":"valid",**met});validation[variant]=met
            print("FULL DATA VALID",rows[-1],flush=True)
        pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    selected={mode:max((n for n in validation if n.endswith("_"+mode)),key=lambda n:validation[n]["ndcg@10"]) for mode in ["raw","masked"]}
    save_json(out/"selection.json",{"selected":selected,"validation":validation,"criterion":"Validation only, original heldout arrays"})
    baseline=None;comparisons={}
    for name,model in models.items():
        for masked in [False,True]:
            variant=name+("_masked" if masked else "_raw");policy=MaskedHistoryPolicy(model) if masked else model
            met,per,top=evaluate(policy,data["test"],rms,reference,counts,args.batch)
            rows.append({"model":name,"masked":masked,"variant":variant,"split":"test",**met})
            per.to_csv(out/f"per_user_{variant}.csv",index=False);np.save(out/f"top_{variant}.npy",top)
            if variant=="capped_random_raw":baseline=per
            else:comparisons[variant]=paired(baseline,per)
            print("FULL DATA TEST",rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False);save_json(out/"paired_test_ci.json",comparisons)
    save_json(out/"runtime.json",{"total_seconds":time.perf_counter()-start,"full_train":len(full["y"]),"epochs":args.epochs})


if __name__=="__main__":main()
