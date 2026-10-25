"""Full-catalog rolling-origin replication using an earlier correct time split.

The population classifiers are refitted on the existing earlier snapshots.
Popularity coefficients are selected on that period's earlier validation only.
Previously enumerated full-catalog users are excluded from new evaluation.
"""
import argparse
import json
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoost
from shared.data import ROOT,load_sample,save_json
from shared.metrics import ranking_summary,paired_bootstrap
from task1.population_study import superset_history
from task1.features import build_features
from task1.relational import graph_features


def full_queries(events,cutoff,allowed,cap,seed):
    cutoff=pd.Timestamp(cutoff);past=events[events.timestamp<cutoff]
    future=events[(events.timestamp>=cutoff)&(events.timestamp<cutoff+pd.Timedelta(days=7))]
    catalog=np.sort(past.item_id.unique());known=set(catalog)
    histories=past[past.user_id.isin(allowed)].groupby("user_id").item_id.agg(set).to_dict()
    targets=future[future.user_id.isin(allowed)].groupby("user_id").item_id.agg(set).to_dict()
    eligible=np.array(sorted(u for u in set(histories)&set(targets) if (targets[u]-histories[u])&known))
    selected=np.sort(np.random.default_rng(seed).choice(eligible,min(cap,len(eligible)),replace=False))
    frames=[]
    for user in selected:
        ids=catalog[~np.isin(catalog,list(histories[user]))]
        frames.append(pd.DataFrame({"user_id":int(user),"item_id":ids,"query_id":f"{cutoff.date()}:{user}",
            "label":np.isin(ids,list(targets[user])).astype(np.int8)}))
    q=pd.concat(frames,ignore_index=True);q["cutoff"]=cutoff
    return q,{"users":len(selected),"selected_user_ids":selected.tolist(),"eligible_warm_active_users":len(eligible),
        "catalog":len(catalog),"rows":len(q),"cutoff":str(cutoff.date()),"max_past_timestamp":str(past.timestamp.max()),
        "rule":"Full past catalog minus user history; active users with known targets; no target injection"}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--date",default="2021-08-09")
    ap.add_argument("--out",default="artifacts/task1_rolling_catalog");cli=ap.parse_args()
    out=ROOT/cli.out;out.mkdir(exist_ok=True);models=out/"population_models"
    source=ROOT/f"artifacts/task1_temporal/test_{cli.date}"
    boundaries=json.loads((source/"time_boundaries.json").read_text());valid_date=boundaries["valid_cutoff"]
    save_json(out/"study_protocol.json",{"purpose":"Exploratory full-catalog temporal replication with new evaluation users",
        "test_date":cli.date,"valid_date":valid_date,"train_cutoffs":boundaries["train_cutoffs"],
        "population":200000,"supervised_source":str(source.relative_to(ROOT)),"graph_group":"Frozen all16",
        "classifier":"Original depth6,lr.07,L2=5,350trees,validation earlystop; seeds42/43/44",
        "alpha_grid":[0,.1,.25,.5,1.],"selection":"Per-family full-catalog validation NDCG, seed42; no future-period coefficients",
        "validation_cap":128,"test_cap":512,"query_seeds":{"valid":20261028,"test":20261029},
        "excluded":"Original50k supervised population, prior128/512full-test users and prior124calibration users; new validation/test disjoint",
        "limits":"Same studied KION, retrospective method choice, overlapping historical graph population; active/warm evaluation, not pristine independent dataset"})
    start=time.perf_counter()
    if not (models/"runtime.json").exists():
        with (out/"population_training.log").open("w") as log:
            subprocess.run([sys.executable,"-m","task1.population_study","--source",str(source.relative_to(ROOT)),
                "--out",str(models.relative_to(ROOT))],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    original,items,users=load_sample(50000);events=superset_history(original,200000,out)
    excluded=set(original.user_id)
    for folder in ["task1_full_catalog","task1_full_catalog_replication"]:
        excluded.update(json.loads((ROOT/f"artifacts/{folder}/candidate_audit.json").read_text())["selected_user_ids"])
    excluded.update(json.loads((ROOT/"artifacts/task1_catalog_calibration/validation_audit.json").read_text())["selected_user_ids"])
    allowed=set(events.user_id)-excluded;rows=[];choices={};all_per={}
    for split,date,cap,seed in [("valid",valid_date,128,20261028),("test",cli.date,512,20261029)]:
        q,audit=full_queries(events,date,allowed,cap,seed)
        if split=="valid":allowed-=set(audit["selected_user_ids"])
        save_json(out/f"{split}_audit.json",audit)
        manual=build_features(events,items,users,q,date,include_auto=False)
        learned,graph_audit=graph_features(events,q,date);x=pd.concat([manual,learned],axis=1)
        save_json(out/f"{split}_graph_audit.json",graph_audit)
        q.to_parquet(out/f"{split}_candidates.parquet",index=False);x.to_parquet(out/f"{split}_features.parquet",index=False)
        popularity=np.log1p(x.i_count_w0.astype(np.float64).to_numpy())
        for name in ["manual","graph"]:
            all_per[name]=[]
            for model_seed in ([42] if split=="valid" else [42,43,44]):
                model=CatBoost().load_model(str(models/f"{name}_{model_seed}.cbm"));raw=model.predict(x[model.feature_names_])
                if split=="valid":
                    options=[]
                    for alpha in [0,.1,.25,.5,1.]:
                        met,per=ranking_summary(q,raw+alpha*popularity)
                        row={"model":name,"seed":model_seed,"alpha":alpha,"split":split,**met};rows.append(row);options.append(row)
                    choices[name]=max(options,key=lambda r:r["ndcg@10"])
                else:
                    for mode,alpha in [("raw",0.),("calibrated",choices[name]["alpha"])]:
                        met,per=ranking_summary(q,raw+alpha*popularity)
                        rows.append({"model":name,"seed":model_seed,"alpha":alpha,"split":split,"mode":mode,**met})
                        per.to_csv(out/f"per_query_{mode}_{name}_{model_seed}.csv",index=False)
                        if mode=="calibrated":all_per[name].append(per)
                        print("ROLLING CATALOG",date,rows[-1],flush=True)
        if split=="valid":save_json(out/"selection.json",{"selected":choices,"criterion":"Earlier full-catalog validation NDCG, frozen before test"})
        pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
        del x,q,manual,learned
    avg={n:pd.concat(frames).groupby("query_id",as_index=False)[["ndcg","recall","map"]].mean() for n,frames in all_per.items()}
    save_json(out/"paired_calibrated_ci.json",paired_bootstrap(avg["manual"],avg["graph"]))
    save_json(out/"runtime.json",{"total_seconds":time.perf_counter()-start})


if __name__=="__main__":main()
