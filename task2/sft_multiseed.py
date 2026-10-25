"""Standalone CPU/CUDA SFT initializer replication with fixed data and 3 seeds.

Graph-row permutation preserves the initializer's distribution but destroys
item alignment (both popularity and collaborative structure, not only semantics).
"""
import argparse
import hashlib
import importlib.metadata
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits
from shared.data import ROOT,load_sample,save_json,VAL_CUTOFF
from shared.analyze import bootstrap
from task2.data import make_examples
from task2.model import GPT2Rec
from task2.history_mask import MaskedHistoryPolicy
from task2.run import set_seed,train_policy,evaluate


def data_hash(part):
    h=hashlib.sha256()
    for key in ["x","y","user"]:h.update(np.asarray(part[key],dtype=np.int64).tobytes())
    h.update(np.asarray(part["date"],dtype="U10").tobytes())
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="artifacts/task2_sft_multiseed")
    ap.add_argument("--users",type=int,default=50000);ap.add_argument("--full-train",action="store_true");cli=ap.parse_args()
    out=ROOT/cli.out;out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args=argparse.Namespace(users=cli.users,max_len=32,max_train=100000,max_eval=2000,epochs=8,batch=128,lr=.001,seed=42)
    save_json(out/"study_protocol.json",{"purpose":"Independent-device multi-initialization and seed replication",
        "methods":["random","graph","permuted_graph"],"sft_seeds":[42,43,44],"data_seed":42,"epochs":8,
        "transitions":"All eligible" if cli.full_train else 100000,"graph":"Binary full train graph,64dim SVD seed42,V*sqrt(S),RMS.02",
        "permutation_seed":20261031,"control_limit":"Row permutation destroys item association including popularity, not solely semantics",
        "selection":"Per-run raw validation checkpoint, then mean validation initializer choice; all test variants reported after selection",
        "information_limit":"Graph uses all train edges including first-day/singleton rows absent as next-day targets; extra unsupervised stage",
        "device":str(device),"device_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "versions":{n:importlib.metadata.version(n) for n in ["torch","numpy","pandas","scipy","scikit-learn"]},
        "warning":"Original test inspected; additional replication, not fresh external confirmation"})
    start=time.perf_counter();events,_,_=load_sample(cli.users)
    data,mapping,counts,audit=make_examples(events,32,100000,2000,42)
    expected_path=ROOT/"task2/evaluation_manifest.json"
    expected=json.loads(expected_path.read_text())
    assert {s:data_hash(part) for s,part in data.items()}==expected["array_hashes"],"Input arrays differ from original CPU protocol"
    if cli.full_train:
        complete,new_mapping,_,_=make_examples(events,32,len(events)+1,2000,42)
        assert new_mapping==mapping
        data["train"]=complete["train"]  # Preserve the original heldout arrays.
        audit["retained_examples"]["train"]=len(data["train"]["y"])
    hashes={s:data_hash(part) for s,part in data.items()}
    assert hashes==expected["full_array_hashes" if cli.full_train else "array_hashes"]
    save_json(out/"data_audit.json",{"array_hashes":hashes,"original_eval_protocol_match":True,"counts":audit,
        "training_protocol":"All eligible" if cli.full_train else "100k cap"})
    for split,part in data.items():np.savez_compressed(out/f"examples_{split}.npz",**part)
    past=events[events.timestamp<pd.Timestamp(VAL_CUTOFF)];u=pd.Index(np.sort(past.user_id.unique()));columns=np.array([mapping[int(i)]-2 for i in past.item_id])
    a=sparse.csr_matrix((np.ones(len(past),dtype=np.float32),(u.get_indexer(past.user_id),columns)),shape=(len(u),len(mapping)))
    a.data[:]=1
    with threadpool_limits(limits=4):svd=TruncatedSVD(n_components=64,n_iter=7,random_state=42).fit(a)
    factors=svd.components_.T*np.sqrt(svd.singular_values_)[None,:];factors=(factors*(.02/np.sqrt(np.mean(factors**2)))).astype(np.float32)
    permutation=np.random.default_rng(20261031).permutation(len(factors))
    save_json(out/"initializer_audit.json",{"factor_sha256":hashlib.sha256(factors.tobytes()).hexdigest(),
        "permutation_sha256":hashlib.sha256(permutation.tobytes()).hexdigest(),"cutoff":VAL_CUTOFF,"max_graph_timestamp":str(past.timestamp.max()),"edges":int(a.nnz)})
    rows=[]
    for seed in [42,43,44]:
        args.seed=seed
        for method in ["random","graph","permuted_graph"]:
            set_seed(seed);model=GPT2Rec(len(mapping)+2).to(device)
            if method!="random":
                initial=factors if method=="graph" else factors[permutation]
                with torch.no_grad():model.item.weight[2:].copy_(torch.as_tensor(initial,device=device))
            cell=out/f"{method}_seed{seed}";cell.mkdir(exist_ok=True)
            t=time.perf_counter();train_policy(model,data["train"],data["valid"],args,cell)
            for masked in [False,True]:
                policy=MaskedHistoryPolicy(model) if masked else model
                met,per,top=evaluate(policy,data["valid"],batch=args.batch,counts=counts)
                rows.append({"method":method,"seed":seed,"masked":masked,"split":"valid","fit_seconds":time.perf_counter()-t,**met})
            pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    valid=pd.DataFrame(rows);means=valid[~valid.masked].groupby("method")["ndcg@10"].mean()
    save_json(out/"selection.json",{"selected":str(means.idxmax()),"raw_validation_means":means.to_dict(),"criterion":"Mean validation raw NDCG across three SFT seeds"})
    per_methods={}
    for seed in [42,43,44]:
        for method in ["random","graph","permuted_graph"]:
            model=GPT2Rec(len(mapping)+2).to(device)
            model.load_state_dict(torch.load(out/f"{method}_seed{seed}/sft_best.pt",map_location=device,weights_only=True))
            for masked in [False,True]:
                policy=MaskedHistoryPolicy(model) if masked else model
                met,per,top=evaluate(policy,data["test"],batch=args.batch,counts=counts)
                rows.append({"method":method,"seed":seed,"masked":masked,"split":"test",**met})
                per.to_csv(out/f"per_user_{method}_seed{seed}_mask{int(masked)}.csv",index=False)
                per_methods.setdefault((method,masked),[]).append(per)
                print("MULTISEED TEST",rows[-1],flush=True)
    effects={}
    for masked in [False,True]:
        avg={method:pd.concat(per_methods[(method,masked)]).groupby("user_id",as_index=False)[["ndcg","hit"]].mean() for method in ["random","graph","permuted_graph"]}
        for method in ["graph","permuted_graph"]:
            j=avg["random"].merge(avg[method],on="user_id",suffixes=("_a","_b"),validate="one_to_one")
            effects[f"{method}_mask{int(masked)}"]={"ndcg":bootstrap(j.ndcg_b-j.ndcg_a),"recall":bootstrap(j.hit_b-j.hit_a)}
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False);save_json(out/"paired_test_ci.json",effects)
    save_json(out/"runtime.json",{"total_seconds":time.perf_counter()-start,"device":str(device)})


if __name__=="__main__":main()
