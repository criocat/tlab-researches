"""Frozen-model all-known-catalog ranking on a uniform fixed query subset.

Candidate membership is determined by past catalog minus past user history.
Labels are attached only after enumeration; no positive injection or new fit.
"""
import argparse
import json
import time
import resource
import numpy as np
import pandas as pd
from catboost import CatBoost
from shared.data import ROOT, load_sample, save_json, TEST_CUTOFF
from shared.metrics import ranking_summary, paired_bootstrap
from task1.population_study import superset_history
from task1.features import build_features
from task1.relational import graph_features


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--queries",type=int,default=128)
    ap.add_argument("--sampling-seed",type=int,default=20261025)
    ap.add_argument("--exclude-prior",action="store_true")
    ap.add_argument("--calibration",default=None)
    ap.add_argument("--out",default="artifacts/task1_full_catalog");cli=ap.parse_args()
    out=ROOT/cli.out;out.mkdir(exist_ok=True);src=ROOT/"artifacts/task1_population"
    save_json(out/"study_protocol.json",{"purpose":"Exploratory full-known-catalog diagnostic with frozen models",
        "users":cli.queries,"user_sampling_seed":cli.sampling_seed,"source_cohort":"Additional users outside original50k",
        "exclude_prior_full_and_calibration_users":cli.exclude_prior,"calibration":cli.calibration,
        "candidate_rule":"Every item observed before cutoff, minus entire prior history of each user; no target injection",
        "positive_rule":"Observed next-week interactions on available catalog; same targets as earlier cohort",
        "models":"Frozen population-study manual/graph models42/43/44; no fit or new hyperparameter selection",
        "matched_sampled_comparator":"Same selected users and existing up-to100-negative candidate evaluation",
        "limitations":"Small random active/warm subset, same already studied dataset/week; not online or unconditioned population quality"})
    start=time.perf_counter();original,items,users=load_sample(50000)
    events=superset_history(original,200000,out)
    oldq=pd.read_parquet(ROOT/"artifacts/task1_cohort/test_candidates.parquet")
    eligible=np.sort(oldq.user_id.unique())
    if cli.exclude_prior:
        previous=json.loads((ROOT/"artifacts/task1_full_catalog/candidate_audit.json").read_text())["selected_user_ids"]
        validation=pd.read_parquet(ROOT/"artifacts/task1_catalog_calibration/valid_candidates.parquet",columns=["user_id"]).user_id.unique()
        eligible=np.setdiff1d(eligible,np.union1d(previous,validation))
    selected=np.sort(np.random.default_rng(cli.sampling_seed).choice(eligible,min(cli.queries,len(eligible)),replace=False))
    cutoff=pd.Timestamp(TEST_CUTOFF);past=events[events.timestamp<cutoff]
    catalog=np.sort(past.item_id.unique());hist=past[past.user_id.isin(selected)].groupby("user_id").item_id.agg(set)
    positives=oldq[(oldq.label==1)&oldq.user_id.isin(selected)].groupby("user_id").item_id.agg(set)
    frames=[]
    for user in selected:
        ids=catalog[~np.isin(catalog,list(hist[user]))]
        frames.append(pd.DataFrame({"user_id":int(user),"item_id":ids,"query_id":f"{TEST_CUTOFF}:{user}",
            "label":np.isin(ids,list(positives[user])).astype(np.int8)}))
    q=pd.concat(frames,ignore_index=True);q["cutoff"]=cutoff
    assert len(q)==sum(len(catalog)-len(hist[user]) for user in selected)
    assert q.groupby("user_id").label.sum().to_dict()=={int(u):len(positives[u]) for u in selected}
    assert not set(selected)&set(original.user_id)
    manual=build_features(events,items,users,q,TEST_CUTOFF,include_auto=False)
    graph,audit=graph_features(events,q,TEST_CUTOFF);x=pd.concat([manual,graph],axis=1)
    x.to_parquet(out/"test_features.parquet",index=False);q.to_parquet(out/"test_candidates.parquet",index=False)
    save_json(out/"candidate_audit.json",{"users":len(selected),"selected_user_ids":selected.tolist(),"catalog":len(catalog),
        "rows":len(q),"positives":int(q.label.sum()),"all_catalog_minus_histories":True,"injected_positives":False,
        "max_catalog_timestamp":str(past.timestamp.max()),"cutoff":TEST_CUTOFF})
    save_json(out/"graph_audit.json",audit)
    rows=[];per={"manual":[],"graph":[]};calibrated_per={"manual":[],"graph":[]}
    calibration=json.loads((ROOT/cli.calibration).read_text())["selected"] if cli.calibration else None
    log_pop=np.log1p(q.item_id.map(past.groupby("item_id").size()).to_numpy()) if calibration else None
    for name in per:
        for seed in [42,43,44]:
            model=CatBoost().load_model(str(src/f"{name}_{seed}.cbm"))
            scores=model.predict(x[model.feature_names_]);met,user=ranking_summary(q,scores)
            per[name].append(user);user.to_csv(out/f"per_query_{name}_{seed}.csv",index=False)
            rows.append({"model":name,"seed":seed,"protocol":"full_catalog","queries":len(selected),"candidate_rows":len(q),**met})
            if calibration:
                calibrated,cal_per=ranking_summary(q,scores+calibration[name]["alpha"]*log_pop)
                calibrated_per[name].append(cal_per)
                cal_per.to_csv(out/f"per_query_calibrated_{name}_{seed}.csv",index=False)
                rows.append({"model":name,"seed":seed,"protocol":"calibrated_full","alpha":calibration[name]["alpha"],
                    "queries":len(selected),"candidate_rows":len(q),**calibrated})
            prior=pd.read_csv(ROOT/f"artifacts/task1_cohort/per_query_{name}_{seed}.csv")
            query_ids={f"{TEST_CUTOFF}:{u}" for u in selected};prior=prior[prior.query_id.isin(query_ids)]
            assert len(prior)==len(selected)
            rows.append({"model":name,"seed":seed,"protocol":"sampled_100","queries":len(selected),
                "candidate_rows":int(oldq.user_id.isin(selected).sum()),
                "ndcg@10":prior.ndcg.mean(),"recall@10":prior.recall.mean(),"map@10":prior["map"].mean()})
            print("FULLCATALOG",rows[-2:],flush=True)
    avg={n:pd.concat(frames).groupby("query_id",as_index=False)[["ndcg","recall","map"]].mean() for n,frames in per.items()}
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    save_json(out/"paired_test_ci.json",paired_bootstrap(avg["manual"],avg["graph"]))
    if calibration:
        avg={n:pd.concat(frames).groupby("query_id",as_index=False)[["ndcg","recall","map"]].mean() for n,frames in calibrated_per.items()}
        save_json(out/"paired_calibrated_ci.json",paired_bootstrap(avg["manual"],avg["graph"]))
    save_json(out/"runtime.json",{"total_seconds":time.perf_counter()-start,"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


if __name__=="__main__":main()
