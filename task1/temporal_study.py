"""Retrospective rolling-origin replication with the frozen feature grammar.

No result from this study chooses a new primary model. The original held-out
results were already inspected; these are exploratory temporal diagnostics.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
from shared.data import ROOT,save_json
from shared.analyze import task1 as analyze_task1
import task1.run as pipeline


DATES=["2021-07-26","2021-08-02","2021-08-09"]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="artifacts/task1_temporal")
    args=ap.parse_args(); out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    spec={"purpose":"Exploratory rolling-origin robustness; no new primary selection",
        "test_cutoffs":DATES,"seeds":[42,43,44],"users":50000,"queries":1500,
        "frozen":"Original grammar, model parameters, selection grid and candidate rules",
        "warning":"Earlier weeks and overlapping users are not independent pristine holdouts"}
    save_json(out/"study_protocol.json",spec)
    summary=[]
    for date in DATES:
        t=pd.Timestamp(date); cell=out/f"test_{date}"
        train=[str((t-pd.Timedelta(days=d)).date()) for d in [35,28,21,14]]
        valid=str((t-pd.Timedelta(days=7)).date())
        save_json(cell/"time_boundaries.json",{"train_cutoffs":train,"valid_cutoff":valid,"test_cutoff":date})
        if not (cell/"metrics.csv").exists():
            pipeline.TRAIN_CUTOFFS=train;pipeline.VAL_CUTOFF=valid;pipeline.TEST_CUTOFF=date
            sys.argv=["task1.run","--users","50000","--queries","1500","--seeds","42","43","44","--out",str(cell.relative_to(ROOT))]
            pipeline.main()
        analyze_task1(cell)
        import json
        analysis=json.loads((cell/"analysis.json").read_text())
        sel=json.loads((cell/"selection.json").read_text())
        summary.append({"test_cutoff":date,"valid_cutoff":valid,"selected":sel["selected"],
            "features":len(sel["columns"]),"manual_ndcg":analysis["mean_test_metrics"]["manual"]["ndcg@10"],
            "auto_ndcg":analysis["mean_test_metrics"]["auto_selected"]["ndcg@10"],
            **analysis["seed_averaged_paired_ndcg_delta"]})
        pd.DataFrame(summary).to_csv(out/"summary.csv",index=False)
    print("TEMPORAL COMPLETE",summary,flush=True)


if __name__=="__main__":main()
