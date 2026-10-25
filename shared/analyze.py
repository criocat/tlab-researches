"""Post-training diagnostics from frozen artifacts; no model/hyperparameter fit."""
import argparse
from collections import Counter
import json
import numpy as np
import pandas as pd
from shared.data import ROOT,save_json
from shared.metrics import paired_bootstrap


def bootstrap(delta,seed=42):
    rng=np.random.default_rng(seed)
    samples=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)]
    return {"mean":float(np.mean(delta)),"ci95":np.quantile(samples,[.025,.975]).tolist(),"users":len(delta)}


def promptfe(base):
    proposals=json.loads((base/"proposals.json").read_text())
    budget={m:{"attempts":sum(r["method"]==m for r in proposals),
        "valid":sum(r["method"]==m and r["valid"] for r in proposals),
        "attempt_seconds":sum(r["seconds"] for r in proposals if r["method"]==m)} for m in ["random","llm"]}
    analysis={"budget":budget,"llm_minus_random":paired_bootstrap(
        pd.read_csv(base/"per_query_random.csv"),pd.read_csv(base/"per_query_llm.csv")),
        "invalid_reasons":dict(Counter(r["error"] for r in proposals if not r["valid"])),
        "timing_limitation":("CPU search partially overlapped the original RPN worker; not an isolated hardware benchmark"
            if (base/"RECOVERY.json").exists() else "Measured attempt time includes generation and fitting; full run also includes initialization and final subset fits")}
    save_json(base/"analysis.json",analysis)


def task1(base):
    metrics=pd.read_csv(base/"metrics.csv"); test=metrics[metrics.split=="test"]
    seeds=sorted(test.loc[test.model=="manual","seed"].astype(int).unique())
    deltas=[]
    for seed in seeds:
        a=pd.read_csv(base/f"per_query_manual_{seed}.csv")
        b=pd.read_csv(base/f"per_query_auto_selected_{seed}.csv")
        pair=a.merge(b,on="query_id",suffixes=("_a","_b"),validate="one_to_one").sort_values("query_id")
        deltas.append((pair.ndcg_b-pair.ndcg_a).to_numpy())
    seed_average=np.stack(deltas).mean(0)
    importance=pd.read_csv(base/"feature_importance.csv")
    protocol=json.loads((base/"protocol.json").read_text())
    raw={"train":protocol["audits"]["train"],"valid":protocol["audits"]["valid"],"test":protocol["audits"]["test"]}
    first=seeds[0]
    rows=test[test.seed==first].set_index("model")
    analysis={"seed_averaged_paired_ndcg_delta":bootstrap(seed_average),
        "ci_interpretation":"Average each user's metric across training seeds, then bootstrap users; not an ensemble-ranking metric or an independent-time-window CI",
        "mean_test_metrics":test.groupby("model")[["ndcg@10","recall@10","map@10"]].mean().to_dict(orient="index"),
        "manual_vs_no_indirect_same_seed":float(rows.loc["manual","ndcg@10"]-rows.loc["manual_no_indirect","ndcg@10"]),
        "direct_user_item_importance_sum":float(importance.loc[importance.feature.str.startswith("ui_"),"importance"].sum()),
        "top_features":importance.head(15).to_dict(orient="records"),
        "feature_seconds_all_snapshots":float(sum(a["feature_seconds"] for split in raw.values() for a in split)),
        "all_search_fit_seconds":float(metrics[metrics.split=="valid"].train_seconds.sum()),
        "sampled_ranking_audit":raw,
        "timing_limitation":"Manual-only feature materialization was not separately timed; both methods used the shared generated matrix"}
    save_json(base/"analysis.json",analysis)
    if (base.parent/"task1_promptfe/metrics.csv").exists():promptfe(base.parent/"task1_promptfe")
    print(json.dumps({k:v for k,v in analysis.items() if k not in ["top_features","sampled_ranking_audit"]},indent=2),flush=True)


def task2(base):
    met=pd.read_csv(base/"metrics.csv").set_index("model")
    select=json.loads((base/"selection.json").read_text())
    selected=select["selected"]; sft=met.loc["sft"]; target=met.loc[selected]
    quality=json.loads((base/"reward_quality.json").read_text())
    train=np.load(base/"examples_train.npz",allow_pickle=False)
    train_mask=train["return_mask"].astype(bool)
    prior=float(train["reward"][train_mask,1].mean())
    for split in ["valid","test"]:
        with np.load(base/f"examples_{split}.npz",allow_pickle=False) as d:
            mask=d["return_mask"].astype(bool); y=d["reward"][mask,1]
        quality[split]["return_prior_baseline"]={"return3d_train_prior":prior,
            "return3d_observed_prevalence":float(y.mean()),"return3d_brier":float(((y-prior)**2).mean()),
            "return3d_n":len(y),"return3d_auc":.5}
    save_json(base/"reward_quality.json",quality)
    primary=json.loads((base/"selected_test_ci.json").read_text()) if (base/"selected_test_ci.json").exists() else None
    analysis={"selected_on_validation":selected,
        "selected_test_ndcg_relative_change":float(target["ndcg@10"]/sft["ndcg@10"]-1),
        "selected_test_watch_proxy_relative_change":float(target["predicted_watch@10"]/sft["predicted_watch@10"]-1),
        "selected_paired_ci":primary,
        "unconstrained_ndcg_relative_change":float(met.loc["rl_watch_unconstrained","ndcg@10"]/sft["ndcg@10"]-1),
        "unconstrained_watch_proxy_relative_change":float(met.loc["rl_watch_unconstrained","predicted_watch@10"]/sft["predicted_watch@10"]-1),
        "unconstrained_max_item_user_fraction":float(met.loc["rl_watch_unconstrained","max_item_user_fraction"]),
        "sft_max_item_user_fraction":float(sft["max_item_user_fraction"]),
        "cold_target_fraction":float(sft["cold_target_fraction"]),
        "limitations":["RM explains little watch-fraction variance on held-out logged targets",
            "Independent initialization is not independent ground truth",
            "Generated context is hypothetical consumption, not a user transition",
            "Three-day return is a censored correlation proxy; no logged propensities or online treatment outcomes"]}
    # Reconstruct held-out predictions from frozen checkpoints for an auditable
    # reward-error/support analysis. No further fitting or checkpoint selection.
    import torch
    from task2.model import RewardModel
    from task2.run import reward_eval
    config=json.loads((base/"model_config.json").read_text())
    with np.load(base/"examples_test.npz",allow_pickle=False) as z:
        test_data={k:z[k] for k in z.files}
    observed=np.isfinite(train["reward"][:,0])
    count=np.bincount(train["y"][observed],minlength=config["vocab"])
    sums=np.bincount(train["y"][observed],weights=train["reward"][observed,0],minlength=config["vocab"])
    mean=float(np.nanmean(train["reward"][:,0])); item=(sums+20*mean)/(count+20)
    predicted={}
    for name in ["reward","auditor"]:
        model=RewardModel(config["vocab"])
        model.load_state_dict(torch.load(base/f"{name}.pt",map_location="cpu",weights_only=True))
        _,predicted[name]=reward_eval(model,test_data)
    truth=test_data["reward"][:,0]; valid=np.isfinite(truth)
    main=predicted["reward"][:,0]; error=np.abs(main-truth)
    assert abs(float(error[valid].mean())-quality["test"]["reward"]["watch_mae"])<1e-5
    item_pred=item[test_data["y"]]
    analysis["reward_mae_gain_over_item_mean"]=bootstrap((np.abs(item_pred-truth)-error)[valid])
    prediction_table=pd.DataFrame({"user_id":test_data["user"],"target_token":test_data["y"],
        "watch_true":truth,"watch_reward_pred":main,"watch_auditor_pred":predicted["auditor"][:,0],
        "watch_item_mean_pred":item_pred,"reward_train_support":count[test_data["y"]],
        "cold_target":test_data["y"]==1})
    prediction_table.to_csv(base/"reward_test_predictions.csv",index=False)
    analysis["recommendation_reward_support"]={}
    for name in met.index:
        top=np.load(base/f"top10_{name}.npy",allow_pickle=False)
        support=count[top]
        analysis["recommendation_reward_support"][name]={"mean":float(support.mean()),
            "median":float(np.median(support)),"zero_fraction":float((support==0).mean()),
            "at_most_five_fraction":float((support<=5).mean())}
    if "continued_sft_control" in met.index:
        a=pd.read_csv(base/"per_user_continued_sft_control.csv")
        b=pd.read_csv(base/"per_user_rl_watch_kl01.csv")
        p=a.merge(b,on="user_id",suffixes=("_a","_b"),validate="one_to_one")
        analysis["one_step_rl_vs_continued_sft_ndcg"]=bootstrap((p.ndcg_b-p.ndcg_a).to_numpy())
        analysis["one_step_rl_vs_continued_sft_watch_proxy_delta"]=float(met.loc["rl_watch_kl01","predicted_watch@10"]-met.loc["continued_sft_control","predicted_watch@10"])
    roll=json.loads((base/"rollout_diagnostics.json").read_text())
    analysis["fifth_hypothetical_step"]={name:obj["steps"][-1] for name,obj in roll.items()}
    save_json(base/"analysis.json",analysis)
    print(json.dumps({k:v for k,v in analysis.items() if k!="fifth_hypothetical_step"},indent=2),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--task",choices=["task1","task2"],required=True);ap.add_argument("--root",default="artifacts")
    args=ap.parse_args();{"task1":task1,"task2":task2}[args.task](ROOT/args.root/args.task)
