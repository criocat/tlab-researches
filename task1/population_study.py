"""Increase historical graph population while keeping supervised queries fixed.

Both manual and graph features see the exact same larger history. All histories
of the original 50k users are kept and 150k uniformly sampled users are added.
No extra labelled training pairs or test candidates are introduced.
"""
import argparse
import hashlib
import json
import resource
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from shared.data import ROOT, load_sample, save_json
from shared.metrics import ranking_summary, paired_bootstrap
from task1.features import build_features, CATEGORIES
from task1.relational import graph_features


def superset_history(original, n_users, out):
    path=ROOT/"data/raw/interactions.csv"
    ids=pd.read_csv(path,usecols=["user_id"]).user_id.unique()
    original_ids=np.sort(original.user_id.unique())
    pool=np.setdiff1d(ids,original_ids)
    added=np.random.default_rng(20261025).choice(pool,n_users-len(original_ids),replace=False)
    selected=np.sort(np.concatenate([original_ids,added]))
    chunks=[];counts=[]
    for chunk in pd.read_csv(path,chunksize=500000):
        part=chunk[chunk.user_id.isin(selected)]
        chunks.append(part);counts.append(part.groupby("user_id").size())
    events=pd.concat(chunks,ignore_index=True)
    events["timestamp"]=pd.to_datetime(events.pop("last_watch_dt"))
    events["watched_pct"]=pd.to_numeric(events.watched_pct,errors="coerce").clip(0,100)
    events=events.sort_values(["timestamp","user_id","item_id"]).reset_index(drop=True)
    source_counts=pd.concat(counts).groupby(level=0).sum().sort_index()
    retained=events.groupby("user_id").size().sort_index()
    pd.testing.assert_series_equal(source_counts,retained)
    old_counts=original.groupby("user_id").size().sort_index()
    pd.testing.assert_series_equal(retained.reindex(old_counts.index),old_counts)
    assert set(original_ids).issubset(set(selected)) and len(retained)==n_users
    save_json(out/"sampling_audit.json",{"users":n_users,"source_rows":len(events),"original_users":len(original_ids),
        "added_users":len(added),"all_source_rows_of_selected_users_retained":True,"original_histories_preserved":True,
        "sampling":"Uniform additional users without replacement outside original uniform50k; no row sampling",
        "selection_seed":20261025,"user_ids_sha256":hashlib.sha256(selected.tobytes()).hexdigest()})
    return events


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--users",type=int,default=200000)
    ap.add_argument("--source",default="artifacts/task1")
    ap.add_argument("--out",default="artifacts/task1_population");cli=ap.parse_args()
    out=ROOT/cli.out;out.mkdir(exist_ok=True);src=ROOT/cli.source
    assert cli.users>50000
    source=json.loads((src/"protocol.json").read_text());cutoffs={k:[r["cutoff"] for r in v] for k,v in source["audits"].items()}
    save_json(out/"study_protocol.json",{"purpose":"Exploratory graph-population scaling after original test inspection",
        "historical_users":cli.users,"supervised_candidates":cli.source+" persisted50k tables, identical rows/labels/candidates",
        "information_fairness":"Both manual and graph use identical larger point-in-time historical population",
        "feature_group":"Frozen all16 graph features, no feature/hyperparameter reselection",
        "seeds":[42,43,44],"cutoffs":cutoffs,"model":"Original Logloss,depth6,lr.07,L2=5,max350,earlystop50onvalidationLogloss",
        "limits":"Additional observational users are not a new held-out sample; original candidate catalog remains fixed"})
    start=time.perf_counter();original,items,users=load_sample(50000)
    events=superset_history(original,cli.users,out);q={};x={};audits={}
    for split in cutoffs:
        q[split]=pd.read_parquet(src/f"{split}_candidates.parquet");frames=[];audits[split]=[]
        for cutoff in cutoffs[split]:
            part=q[split][q[split].cutoff==cutoff]
            t=time.perf_counter();manual=build_features(events,items,users,part,cutoff,include_auto=False)
            learned,audit=graph_features(events,part,cutoff)
            audit["builder_seconds"]=time.perf_counter()-t;audits[split].append(audit)
            frames.append(pd.concat([manual,learned],axis=1))
            print("POPULATION FEATURES",split,cutoff,audit["builder_seconds"],flush=True)
        x[split]=pd.concat(frames,ignore_index=True);x[split].to_parquet(out/f"{split}_features.parquet",index=False)
    save_json(out/"graph_audit.json",audits)
    cols=list(manual.columns);groups={"manual":cols,"graph":list(x["train"])};rows=[];per={n:[] for n in groups}
    for seed in [42,43,44]:
        for name,columns in groups.items():
            t=time.perf_counter();model=CatBoostClassifier(iterations=350,depth=6,learning_rate=.07,l2_leaf_reg=5,
                random_seed=seed,loss_function="Logloss",thread_count=4,verbose=False,allow_writing_files=False)
            model.fit(x["train"][columns],q["train"].label,cat_features=[c for c in CATEGORIES if c in columns],
                eval_set=(x["valid"][columns],q["valid"].label),early_stopping_rounds=50)
            for split in ["valid","test"]:
                prob=model.predict_proba(x[split][columns])[:,1];met,user=ranking_summary(q[split],prob,prob)
                rows.append({"model":name,"seed":seed,"split":split,"population":cli.users,"features":len(columns),
                    "fit_seconds":time.perf_counter()-t,**met})
                if split=="test":
                    per[name].append(user);user.to_csv(out/f"per_query_{name}_{seed}.csv",index=False)
                print("POPULATION",rows[-1],flush=True)
            model.save_model(str(out/f"{name}_{seed}.cbm"))
            pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    means={n:pd.concat(parts).groupby("query_id",as_index=False)[["ndcg","recall","map"]].mean() for n,parts in per.items()}
    save_json(out/"paired_test_ci.json",paired_bootstrap(means["manual"],means["graph"]))
    save_json(out/"runtime.json",{"total_seconds":time.perf_counter()-start,"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


if __name__=="__main__":main()
