"""A stronger censored three-day recorded-activity proxy, with state ablations.

Predictive return probability is not the causal effect of recommending an item.
All features and censoring masks are the existing strict temporal protocol.
"""
import argparse
import time
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss
from shared.data import ROOT,save_json
from shared.analyze import bootstrap
from task1.features import CATEGORIES
from task2.factorial import load_frozen
from task2.run import reward_eval


def metrics(y,p):
    return {"auc":float(roc_auc_score(y,p)),"brier":float(brier_score_loss(y,p)),"logloss":float(log_loss(y,p)),"n":len(y)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="artifacts/task2_return_reward");cli=ap.parse_args()
    out=ROOT/cli.out;out.mkdir(exist_ok=True);torch.set_num_threads(4)
    args,data,mapping,ref,rms=load_frozen(ROOT/"artifacts/task2",torch.device("cpu"))
    x={s:pd.read_parquet(ROOT/f"artifacts/task2_context_reward/{s}_features.parquet") for s in data}
    user=[c for c in x["train"] if c.startswith("u_") or c in ["age","income","sex","kids_flg"]]
    item=[c for c in x["train"] if c.startswith("i_") or c in ["content_type","genre","director","country","age_rating","for_kids","release_year"]]
    groups={"user_only":user,"item_only":item,"full":list(x["train"])}
    save_json(out/"study_protocol.json",{"purpose":"Exploratory richer recorded-activity return model and information ablation",
        "target":"Next recorded active day within three days, original censoring masks; not causal retention",
        "features":"Existing strict past-only ContextFeatures, includes watch fractions and static metadata beyond original item-ID RM",
        "families":groups,"depths":[4,6],"iterations":600,"lr":.05,"l2":10,"seed":42,
        "selection":"Lowest validation Brier per family; training earlystop50 on validation Logloss",
        "test":"All prespecified families plus original neural return head and train-prevalence constant; no test selection",
        "limits":"Already studied test, observational consumed-item data, no exposure propensities; state predictability does not imply action effects"})
    start=time.perf_counter();masks={s:data[s]["return_mask"].astype(bool) for s in data}
    rows=[];models={};choices={}
    for name,cols in groups.items():
        best=None
        for depth in [4,6]:
            model=CatBoostClassifier(iterations=600,depth=depth,learning_rate=.05,l2_leaf_reg=10,random_seed=42,
                loss_function="Logloss",thread_count=4,verbose=False,allow_writing_files=False)
            model.fit(x["train"].loc[masks["train"],cols],data["train"]["reward"][masks["train"],1],
                cat_features=[c for c in CATEGORIES if c in cols],
                eval_set=(x["valid"].loc[masks["valid"],cols],data["valid"]["reward"][masks["valid"],1]),early_stopping_rounds=50)
            pred=model.predict_proba(x["valid"][cols])[:,1];met=metrics(data["valid"]["reward"][masks["valid"],1],pred[masks["valid"]])
            row={"model":name,"depth":depth,"split":"valid",**met};rows.append(row)
            if best is None or met["brier"]<best["brier"]:best=row;models[name]=model
            print("RETURN RM VALID",row,flush=True)
        choices[name]=best
    save_json(out/"selection.json",{"selected":choices,"criterion":"Validation Brier, all before new test scoring"})
    truth=data["test"]["reward"][:,1];observed=masks["test"];predictions={}
    original=reward_eval(rms[0],data["test"])[1][:,1];predictions["original_neural"]=original
    prevalence=float(data["train"]["reward"][masks["train"],1].mean())
    predictions["train_prevalence"]=np.full(len(truth),prevalence)
    for name,model in models.items():
        model.save_model(str(out/f"{name}.cbm"));predictions[name]=model.predict_proba(x["test"][model.feature_names_])[:,1]
    effects={}
    for name,pred in predictions.items():
        row={"model":name,"split":"test",**metrics(truth[observed],pred[observed])};rows.append(row)
        pd.DataFrame({"user_id":data["test"]["user"],"observed":observed,"truth":truth,"prediction":pred}).to_csv(out/f"predictions_{name}.csv",index=False)
        if name not in ["original_neural","train_prevalence"]:
            delta=(original[observed]-truth[observed])**2-(pred[observed]-truth[observed])**2
            effects[name]=bootstrap(delta)
        print("RETURN RM TEST",row,flush=True)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False);save_json(out/"paired_brier_improvement.json",effects)
    save_json(out/"runtime.json",{"total_seconds":time.perf_counter()-start,"train_prevalence":prevalence,
        "observed_rows":{s:int(m.sum()) for s,m in masks.items()}})


if __name__=="__main__":main()
