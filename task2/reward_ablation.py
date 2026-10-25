"""Separate observed-response predictability from candidate-dependent variation.

User-only rewards are constant across actions and cancel in REINFORCE's group
advantage, even if they predict observed responses well. Diagnostic, not causality.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from shared.data import ROOT, save_json, load_sample
from task1.features import CATEGORIES
from task2.factorial import load_frozen
from task2.run import action_logits
from task2.context_reward import ContextFeatures


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="artifacts/task2_reward_ablation");cli=ap.parse_args()
    out=ROOT/cli.out;out.mkdir(exist_ok=True);src=ROOT/"artifacts/task2_context_reward"
    torch.set_num_threads(4);args,data,mapping,ref,rms=load_frozen(ROOT/"artifacts/task2",torch.device("cpu"))
    x={s:pd.read_parquet(src/f"{s}_features.parquet") for s in ["train","valid","test"]}
    user_cols=[c for c in x["train"] if c.startswith("u_") or c in ["age","income","sex","kids_flg"]]
    item_cols=[c for c in x["train"] if c.startswith("i_") or c in ["content_type","genre","director","country","age_rating","for_kids","release_year"]]
    save_json(out/"study_protocol.json",{"purpose":"Exploratory reward information and action-variation ablation",
        "families":{"user_only":user_cols,"item_only":item_cols,"full":list(x["train"])},
        "depths":[4,6],"iterations":600,"lr":.05,"l2":10,"selection":"Minimum validation RMSE per family; full anchor reuses prior depth4/6 results",
        "seeds":42,"observed_evaluation":"Same original train/validation/test examples; strict past-only features",
        "variation":"256 fixed random test contexts, original frozen SFT top100, uniform actions per context",
        "limits":"Predicted variation is not causal controllability; candidate sets also vary between states; dataset/test already studied"})
    start=time.perf_counter();rows=[];models={};validation={}
    observed={s:np.isfinite(data[s]["reward"][:,0]) for s in data}
    for name,cols in [("user_only",user_cols),("item_only",item_cols)]:
        best=None
        for depth in [4,6]:
            t=time.perf_counter();model=CatBoostRegressor(iterations=600,depth=depth,learning_rate=.05,l2_leaf_reg=10,
                random_seed=42,loss_function="RMSE",thread_count=4,verbose=False,allow_writing_files=False)
            model.fit(x["train"].loc[observed["train"],cols],data["train"]["reward"][observed["train"],0],
                cat_features=[c for c in CATEGORIES if c in cols],
                eval_set=(x["valid"].loc[observed["valid"],cols],data["valid"]["reward"][observed["valid"],0]),early_stopping_rounds=50)
            pred=np.clip(model.predict(x["valid"][cols]),0,1);mask=observed["valid"]
            rmse=float(np.sqrt(mean_squared_error(data["valid"]["reward"][mask,0],pred[mask])))
            row={"family":name,"depth":depth,"split":"valid","rmse":rmse,"fit_seconds":time.perf_counter()-t}
            rows.append(row);print("RM ABLATION VALID",row,flush=True)
            if best is None or rmse<best["rmse"]:best=row;models[name]=model
        validation[name]=best
    models["full"]=CatBoostRegressor().load_model(str(src/"reward.cbm"))
    save_json(out/"selection.json",{"validation":validation,"full_anchor":"Frozen previously selected depth6 model"})
    for name,model in models.items():
        model.save_model(str(out/f"{name}.cbm"))
        pred=np.clip(model.predict(x["test"][model.feature_names_]),0,1);mask=observed["test"];truth=data["test"]["reward"][mask,0]
        row={"family":name,"split":"test","mae":float(mean_absolute_error(truth,pred[mask])),
             "rmse":float(np.sqrt(mean_squared_error(truth,pred[mask]))),"r2":float(r2_score(truth,pred[mask]))}
        rows.append(row);print("RM ABLATION TEST",row,flush=True)
        pd.DataFrame({"user_id":data["test"]["user"],"truth":data["test"]["reward"][:,0],"prediction":pred}).to_csv(out/f"predictions_{name}.csv",index=False)
    rng=np.random.default_rng(20261025);indices=np.sort(rng.choice(len(data["test"]["y"]),256,replace=False))
    histories=torch.as_tensor(data["test"]["x"][indices]);tokens=[];old_scores=[]
    with torch.no_grad():
        for lo in range(0,len(indices),64):
            h=histories[lo:lo+64];ids=action_logits(ref,h).topk(100,dim=-1).indices
            tokens.append(ids.numpy());old_scores.append(rms[0](h,ids)[...,0].numpy())
    tokens=np.concatenate(tokens);scores={"original_item_id_rm":np.concatenate(old_scores)}
    inverse={v:k for k,v in mapping.items()};repeat=np.repeat(indices,100)
    queries=pd.DataFrame({"user_id":data["test"]["user"][repeat],"item_id":[inverse[int(i)] for i in tokens.ravel()],
        "timestamp":pd.to_datetime(data["test"]["date"][repeat]).astype("datetime64[ns]")})
    events,items,users=load_sample(50000);features=ContextFeatures(events,items,users).transform(queries)
    for name,model in models.items():scores[name]=np.clip(model.predict(features[model.feature_names_]),0,1).reshape(len(indices),100)
    summary=[];per=[]
    for name,array in scores.items():
        within=array.var(1).mean();between=array.mean(1).var();total=array.var()
        assert np.isclose(total,within+between,atol=1e-10)
        if name=="user_only":assert np.max(np.ptp(array,axis=1))<1e-12
        summary.append({"model":name,"total_variance":total,"within_context_action_variance":within,
                        "between_context_mean_variance":between,"within_fraction":within/total if total else 0.})
        for row,idx in enumerate(indices):per.append({"model":name,"user_id":int(data["test"]["user"][idx]),
            "mean":float(array[row].mean()),"action_sd":float(array[row].std()),"action_range":float(np.ptp(array[row]))})
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(summary).to_csv(out/"variance.csv",index=False);pd.DataFrame(per).to_csv(out/"per_context.csv",index=False)
    save_json(out/"runtime.json",{"total_seconds":time.perf_counter()-start})


if __name__=="__main__":main()
