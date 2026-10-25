"""Frozen-model evaluation on additional users without supervised query overlap.

Past graph edges of these users are allowed (warm/transductive evaluation).
Their labelled training/validation queries never fit or select the classifiers.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
from catboost import CatBoost
from shared.data import ROOT, load_sample, save_json, candidate_queries, TEST_CUTOFF
from shared.metrics import ranking_summary, paired_bootstrap
from task1.population_study import superset_history
from task1.features import build_features
from task1.relational import graph_features


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--queries",type=int,default=5000)
    ap.add_argument("--out",default="artifacts/task1_cohort");cli=ap.parse_args()
    out=ROOT/cli.out;out.mkdir(exist_ok=True);src=ROOT/"artifacts/task1_population"
    save_json(out/"study_protocol.json",{"purpose":"Exploratory frozen-model transfer to a separate evaluation user cohort",
        "population":200000,"query_cap":cli.queries,"query_seed":42,"cutoff":TEST_CUTOFF,
        "cohort":"Only added150k users outside original50k; no supervised training/validation query overlap",
        "allowed_information":"Their historical edges are in the graph and history aggregates, strictly before cutoff",
        "models":"Frozen population-study manual/graph checkpoints seeds42/43/44; no new fit or selection",
        "candidates":"Original generator, complete200k historical catalog, up to100negatives, all available positives",
        "limits":"Same test week and already studied dataset; warm transductive users, not cold-start or pristine preregistered confirmation"})
    start=time.perf_counter();original,items,users=load_sample(50000)
    events=superset_history(original,200000,out);old=set(original.user_id)
    cutoff=pd.Timestamp(TEST_CUTOFF)
    # Eligibility-only view: retain every past edge for the catalog, but expose
    # future target rows only for the new evaluation cohort. Feature history
    # always uses the full events table above; no user history was resampled.
    eligibility_view=events[(events.timestamp<cutoff)|(~events.user_id.isin(old))]
    q,audit=candidate_queries(eligibility_view,TEST_CUTOFF,100,cli.queries,seed=42)
    assert not set(q.user_id)&old
    supervised=pd.read_parquet(ROOT/"artifacts/task1/train_candidates.parquet")
    assert not set(q.user_id)&set(supervised.user_id)
    audit["no_original_user_overlap"]=True;audit["no_supervised_query_overlap"]=True
    save_json(out/"candidate_audit.json",audit)
    manual=build_features(events,items,users,q,TEST_CUTOFF,include_auto=False)
    graph,graph_audit=graph_features(events,q,TEST_CUTOFF)
    x=pd.concat([manual,graph],axis=1);x.to_parquet(out/"test_features.parquet",index=False)
    q.to_parquet(out/"test_candidates.parquet",index=False);save_json(out/"graph_audit.json",graph_audit)
    rows=[];per={"manual":[],"graph":[]}
    for name in per:
        for seed in [42,43,44]:
            model=CatBoost().load_model(str(src/f"{name}_{seed}.cbm"))
            scores=model.predict(x[model.feature_names_]);met,user=ranking_summary(q,scores)
            per[name].append(user);user.to_csv(out/f"per_query_{name}_{seed}.csv",index=False)
            rows.append({"model":name,"seed":seed,"queries":q.query_id.nunique(),**met})
            print("COHORT",rows[-1],flush=True)
    avg={n:pd.concat(frames).groupby("query_id",as_index=False)[["ndcg","recall","map"]].mean() for n,frames in per.items()}
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    save_json(out/"paired_test_ci.json",paired_bootstrap(avg["manual"],avg["graph"]))
    save_json(out/"runtime.json",{"total_seconds":time.perf_counter()-start})


if __name__=="__main__":main()
