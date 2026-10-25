"""Matched validation search: graph time scale vs manual regularization.

Each family gets 4 settings x 5 score-popularity coefficients. The final query
cohort is disjoint from every previously enumerated full-catalog cohort.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from shared.data import ROOT,load_sample,save_json
from shared.metrics import ranking_summary,paired_bootstrap
from task1.population_study import superset_history
from task1.relational import graph_features
from task1.features import build_features,CATEGORIES
from task1.rolling_catalog import full_queries


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="artifacts/task1_decayed_graph");cli=ap.parse_args()
    out=ROOT/cli.out;out.mkdir(exist_ok=True);pop=ROOT/"artifacts/task1_population";original_src=ROOT/"artifacts/task1"
    save_json(out/"study_protocol.json",{"purpose":"Exploratory matched validation search for graph decay and manual regularization",
        "graph_half_lives":[None,10,21,60],"manual_l2":[1,5,20,100],"alpha_grid":[0,.1,.25,.5,1.],
        "budget":"20 full-catalog validation settings per family; reused old infinity/L2=5 anchors",
        "classifier":"Logloss,depth6,lr.07,350trees,earlystop50on original conditional validation Logloss; selection by full-catalog validation NDCG",
        "population":200000,"evaluation_seeds":[42,43,44],"selection_seed":42,
        "validation":"Existing124 full-catalog users09Aug; no test read during parameter selection",
        "test":"512 warm active users16Aug sampled with seed20261030, excluding original50k and every earlier full-catalog/rolling test or validation cohort",
        "candidate_rule":"Full past catalog minus real history; no injected target",
        "limits":"Already studied KION/test week and meta-method choice; new evaluation users, not pristine external confirmation"})
    start=time.perf_counter();original,items,users=load_sample(50000);events=superset_history(original,200000,out)
    q={s:pd.read_parquet(original_src/f"{s}_candidates.parquet") for s in ["train","valid"]}
    p={s:pd.read_parquet(pop/f"{s}_features.parquet") for s in ["train","valid"]}
    graphcols=[c for c in p["train"] if c.startswith(("svd_","knn_"))];basecols=[c for c in p["train"] if c not in graphcols]
    manual={s:p[s][basecols] for s in p}
    qv=pd.read_parquet(ROOT/"artifacts/task1_catalog_calibration/valid_candidates.parquet")
    xv=pd.read_parquet(ROOT/"artifacts/task1_catalog_calibration/valid_features.parquet")
    valpop=np.log1p(xv.i_count_w0.astype(np.float64).to_numpy())
    source=json.loads((original_src/"protocol.json").read_text());cutoffs={s:[r["cutoff"] for r in source["audits"][s]] for s in ["train","valid"]}
    rows=[];choices={};models={};weighted_cache={}
    def train(family,setting,seed,frames,columns):
        model=CatBoostClassifier(iterations=350,depth=6,learning_rate=.07,l2_leaf_reg=setting if family=="manual" else 5,
            loss_function="Logloss",random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False)
        model.fit(frames["train"][columns],q["train"].label,cat_features=[c for c in CATEGORIES if c in columns],
            eval_set=(frames["valid"][columns],q["valid"].label),early_stopping_rounds=50)
        return model
    for family,settings in [("manual",[1,5,20,100]),("graph",[None,10,21,60])]:
        best=None
        for setting in settings:
            if family=="manual":
                frames=manual;validation=xv[basecols];columns=basecols
                model=CatBoostClassifier().load_model(str(pop/"manual_42.cbm")) if setting==5 else train(family,setting,42,frames,columns)
            elif setting is None:
                frames=p;validation=xv;columns=list(p["train"])
                model=CatBoostClassifier().load_model(str(pop/"graph_42.cbm"))
            else:
                frames={};audits={}
                for split in ["train","valid"]:
                    chunks=[];audits[split]=[]
                    for cutoff in cutoffs[split]:
                        part=q[split][q[split].cutoff==cutoff]
                        features,audit=graph_features(events,part,cutoff,half_life=setting)
                        chunks.append(features);audits[split].append(audit)
                    frames[split]=pd.concat([manual[split],pd.concat(chunks,ignore_index=True)],axis=1)
                feat,audit=graph_features(events,qv,"2021-08-09",half_life=setting)
                audits["full_validation"]=audit;save_json(out/f"graph_audit_h{setting}.json",audits)
                validation=pd.concat([xv[basecols],feat],axis=1);columns=list(frames["train"])
                model=train(family,setting,42,frames,columns)
            raw=model.predict(validation[columns],prediction_type="RawFormulaVal")
            for alpha in [0,.1,.25,.5,1.]:
                met,per=ranking_summary(qv,raw+alpha*valpop)
                row={"family":family,"setting":"infinity" if setting is None else setting,"alpha":alpha,"split":"valid",**met};rows.append(row)
                if best is None or met["ndcg@10"]>best["ndcg@10"]:
                    best=row;models[family]=model
                    if family=="graph":weighted_cache={"frames":frames,"columns":columns,"half_life":setting}
            print("DECAY VALID",family,setting,"best",best,flush=True)
        choices[family]=best
    save_json(out/"selection.json",{"selected":choices,"criterion":"Same20 full-catalog validation settings per family; frozen before new cohort evaluation"})
    excluded=set(original.user_id)
    for folder in ["task1_full_catalog","task1_full_catalog_replication"]:
        excluded.update(json.loads((ROOT/f"artifacts/{folder}/candidate_audit.json").read_text())["selected_user_ids"])
    excluded.update(json.loads((ROOT/"artifacts/task1_catalog_calibration/validation_audit.json").read_text())["selected_user_ids"])
    for split in ["valid","test"]:excluded.update(json.loads((ROOT/f"artifacts/task1_rolling_catalog/{split}_audit.json").read_text())["selected_user_ids"])
    qt,audit=full_queries(events,"2021-08-16",set(events.user_id)-excluded,512,20261030);save_json(out/"test_audit.json",audit)
    mt=build_features(events,items,users,qt,"2021-08-16",include_auto=False)
    gt,ga=graph_features(events,qt,"2021-08-16",half_life=weighted_cache["half_life"]);save_json(out/"test_graph_audit.json",ga)
    xt=pd.concat([mt,gt],axis=1);qt.to_parquet(out/"test_candidates.parquet",index=False);xt.to_parquet(out/"test_features.parquet",index=False)
    popularity=np.log1p(xt.i_count_w0.astype(np.float64).to_numpy());per_families={n:[] for n in choices}
    for family in choices:
        for seed in [42,43,44]:
            if seed==42:model=models[family]
            elif family=="manual":model=train(family,int(choices[family]["setting"]),seed,manual,basecols)
            else:model=train(family,weighted_cache["half_life"],seed,weighted_cache["frames"],weighted_cache["columns"])
            columns=model.feature_names_;raw=model.predict(xt[columns],prediction_type="RawFormulaVal")
            met,per=ranking_summary(qt,raw+choices[family]["alpha"]*popularity)
            per_families[family].append(per);per.to_csv(out/f"per_query_{family}_{seed}.csv",index=False)
            model.save_model(str(out/f"{family}_{seed}.cbm"));rows.append({"family":family,"seed":seed,"split":"test",**met})
            print("DECAY TEST",rows[-1],flush=True)
    avg={n:pd.concat(frames).groupby("query_id",as_index=False)[["ndcg","recall","map"]].mean() for n,frames in per_families.items()}
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False);save_json(out/"paired_test_ci.json",paired_bootstrap(avg["manual"],avg["graph"]))
    save_json(out/"runtime.json",{"total_seconds":time.perf_counter()-start})


if __name__=="__main__":main()
