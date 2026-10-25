"""Summarize both immutable GPU seed replications without pooling CPU results."""
import json
import numpy as np
import pandas as pd
from shared.data import ROOT,save_json
from shared.analyze import bootstrap


REPLICATIONS={"capped":"kaggle_sft_multiseed","full":"kaggle_sft_full_multiseed"}


def crossed_bootstrap(differences, repetitions=4000):
    """Paired two-way resampling of the same seeds and users in both methods.

    This is a sensitivity calculation with only three observed training seeds,
    not a reliable estimate of the population of all training runs.
    """
    rng=np.random.default_rng(20261101);nseed,nuser=differences.shape
    values=np.empty(repetitions)
    for i in range(repetitions):
        seeds=rng.integers(nseed,size=nseed);users=rng.integers(nuser,size=nuser)
        values[i]=differences[seeds][:,users].mean()
    return {"mean":float(differences.mean()),"ci95":np.quantile(values,[.025,.975]).tolist(),
            "seed_count":nseed,"users":nuser,"repetitions":repetitions,
            "scope":"Paired crossed seed/user bootstrap sensitivity; three observed seeds only"}


def main():
    out=ROOT/"artifacts/task2_sft_replication";out.mkdir(exist_ok=True)
    summaries=[];contrasts={};provenance={}
    for cap,folder in REPLICATIONS.items():
        base=ROOT/"artifacts"/folder/"tlab/artifacts/task2_sft_multiseed"
        metrics=pd.read_csv(base/"metrics.csv");test=metrics[metrics.split=="test"]
        for (method,masked),rows in test.groupby(["method","masked"]):
            entry={"training":cap,"method":method,"masked":bool(masked),"seeds":len(rows)}
            for name in ["ndcg@10","recall@10","coverage@10","max_item_user_fraction"]:
                entry[name]=rows[name].mean();entry[name+"_seed_sd"]=rows[name].std(ddof=1)
            summaries.append(entry)
        per={}
        for method in ["random","graph","permuted_graph"]:
            for masked in [0,1]:
                per[(method,masked)]=[pd.read_csv(base/f"per_user_{method}_seed{s}_mask{masked}.csv").sort_values("user_id") for s in [42,43,44]]
        for masked in [0,1]:
            for left,right in [("random","graph"),("random","permuted_graph"),("permuted_graph","graph")]:
                key=f"{cap}/{right}_minus_{left}/mask{masked}";contrasts[key]={}
                for column in ["ndcg","hit"]:
                    delta=np.stack([b[column].to_numpy(dtype=float)-a[column].to_numpy(dtype=float) for a,b in zip(per[(left,masked)],per[(right,masked)])])
                    contrasts[key][column]={"user_only":bootstrap(delta.mean(axis=0)),
                                            "seed_and_user_sensitivity":crossed_bootstrap(delta),
                                            "paired_seed_deltas":delta.mean(axis=1).tolist()}
        provenance[cap]={"folder":str(base.relative_to(ROOT)),
            "protocol":json.loads((base/"study_protocol.json").read_text()),
            "runtime":json.loads((base/"runtime.json").read_text()),
            "selection":json.loads((base/"selection.json").read_text())}
    pd.DataFrame(summaries).to_csv(out/"summary.csv",index=False)
    save_json(out/"contrasts.json",contrasts);save_json(out/"provenance.json",provenance)
    save_json(out/"interpretation.json",{
        "scope":"18 GPU SFT fits: two train caps, three initializers and three paired seeds; same original heldout arrays",
        "result":"Graph improves raw NDCG in each observed paired seed; its additional masked gain is uncertain",
        "control":"Permutation preserves initializer weight distribution but destroys item alignment including popularity",
        "limits":"Original test already inspected; CPU and GPU versions/device differ and are reported separately. Three seeds and user-only CIs do not establish training-wide certainty."})
    print(pd.DataFrame(summaries)[["training","method","masked","ndcg@10","ndcg@10_seed_sd","recall@10"]].to_string(index=False))


if __name__=="__main__":main()
