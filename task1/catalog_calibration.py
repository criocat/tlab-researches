"""Validation-only popularity score calibration for a full-catalog protocol.

Motivated by the failed transfer from sampled candidates to all known items.
This is an exploratory one-parameter heuristic, not an exact propensity fix.
Both feature families receive the same validation grid; test never selects alpha.
"""
import argparse
import time
import json
import numpy as np
import pandas as pd
from catboost import CatBoost
from shared.data import ROOT, load_sample, save_json, VAL_CUTOFF
from shared.metrics import ranking_summary, paired_bootstrap
from task1.population_study import superset_history
from task1.features import build_features
from task1.relational import graph_features


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="artifacts/task1_catalog_calibration");cli=ap.parse_args()
    out=ROOT/cli.out;out.mkdir(exist_ok=True);src=ROOT/"artifacts/task1_population"
    save_json(out/"study_protocol.json",{"purpose":"Exploratory adaptation of frozen scores to full-catalog candidate distribution",
        "alpha_grid":[0,.1,.25,.5,1.],"score":"Frozen raw classifier score + alpha*log1p(past item event count)",
        "selection":"Best full-catalog validation NDCG on one frozen seed42, separately per manual/graph, equal grid",
        "validation":"128 uniformly sampled added-population users, week09Aug, full past catalog excluding own history",
        "test":"Previously enumerated frozen128-user full-catalog test; all choices frozen before new scoring",
        "model_fits":0,"limitations":"Heuristic rather than exact negative-sampling correction; full-catalog test already inspected, exploratory; no causal/online claim"})
    start=time.perf_counter();original,items,users=load_sample(50000);events=superset_history(original,200000,out)
    cutoff=pd.Timestamp(VAL_CUTOFF);past=events[events.timestamp<cutoff]
    future=events[(events.timestamp>=cutoff)&(events.timestamp<cutoff+pd.Timedelta(days=7))]
    eligible=np.array(sorted((set(past.user_id)&set(future.user_id))-set(original.user_id)))
    selected=np.sort(np.random.default_rng(20261026).choice(eligible,128,replace=False))
    catalog=np.sort(past.item_id.unique());hist=past[past.user_id.isin(selected)].groupby("user_id").item_id.agg(set)
    targets=future[future.user_id.isin(selected)].groupby("user_id").item_id.agg(set);frames=[];skipped=0
    for user in selected:
        ids=catalog[~np.isin(catalog,list(hist[user]))];positive=set(targets[user])&set(ids)
        if not positive:skipped+=1;continue
        frames.append(pd.DataFrame({"user_id":int(user),"item_id":ids,"query_id":f"{VAL_CUTOFF}:{user}",
                                   "label":np.isin(ids,list(positive)).astype(np.int8)}))
    qv=pd.concat(frames,ignore_index=True);qv["cutoff"]=cutoff
    manual=build_features(events,items,users,qv,VAL_CUTOFF,include_auto=False);learned,audit=graph_features(events,qv,VAL_CUTOFF)
    xv=pd.concat([manual,learned],axis=1);xv.to_parquet(out/"valid_features.parquet",index=False);qv.to_parquet(out/"valid_candidates.parquet",index=False)
    save_json(out/"validation_audit.json",{"queries":qv.query_id.nunique(),"selected_user_ids":sorted(map(int,qv.user_id.unique())),
        "rows":len(qv),"all_cold_skipped":skipped,"graph":audit})
    popularity=np.log1p(qv.item_id.map(past.groupby("item_id").size()).to_numpy())
    rows=[];choices={}
    for name in ["manual","graph"]:
        model=CatBoost().load_model(str(src/f"{name}_42.cbm"));raw=model.predict(xv[model.feature_names_]);best=None
        for alpha in [0,.1,.25,.5,1.]:
            met,per=ranking_summary(qv,raw+alpha*popularity)
            row={"model":name,"seed":42,"alpha":alpha,"split":"valid",**met};rows.append(row)
            if best is None or met["ndcg@10"]>best["ndcg@10"]:best=row
            print("CATALOG CALIBRATION VALID",row,flush=True)
        choices[name]=best
    save_json(out/"selection.json",{"selected":choices,"criterion":"Full-catalog validation NDCG; test not read during alpha selection"})
    # Release validation matrices before reading the frozen test matrices.
    del xv,manual,learned,qv
    qt=pd.read_parquet(ROOT/"artifacts/task1_full_catalog/test_candidates.parquet")
    xt=pd.read_parquet(ROOT/"artifacts/task1_full_catalog/test_features.parquet")
    test_cutoff=pd.Timestamp("2021-08-16");counts=events[events.timestamp<test_cutoff].groupby("item_id").size()
    popularity=np.log1p(qt.item_id.map(counts).to_numpy());test_per={n:[] for n in choices}
    for name in choices:
        for seed in [42,43,44]:
            model=CatBoost().load_model(str(src/f"{name}_{seed}.cbm"));raw=model.predict(xt[model.feature_names_])
            met,per=ranking_summary(qt,raw+choices[name]["alpha"]*popularity)
            per.to_csv(out/f"per_query_{name}_{seed}.csv",index=False);test_per[name].append(per)
            rows.append({"model":name,"seed":seed,"alpha":choices[name]["alpha"],"split":"test",**met})
            print("CATALOG CALIBRATION TEST",rows[-1],flush=True)
    avg={n:pd.concat(frames).groupby("query_id",as_index=False)[["ndcg","recall","map"]].mean() for n,frames in test_per.items()}
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False);save_json(out/"paired_test_ci.json",paired_bootstrap(avg["manual"],avg["graph"]))
    save_json(out/"runtime.json",{"total_seconds":time.perf_counter()-start})


if __name__=="__main__":main()
