"""Run expression-based feature search with a temporal KION environment.

Uses a local stack evaluator, fresh local LLM proposals, validation-only feedback,
a same-budget random-search control, and a frozen final test evaluation.
Replay mode uses saved proposals and needs no LLM downloads.
"""
import argparse
import fcntl
import random
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from shared.data import ROOT,save_json
from shared.metrics import ranking_summary,paired_bootstrap
from task1.features import CATEGORIES,manual_columns
from task1.promptfe_engine import FrameEnvironment,LocalProposer,extract_expression,extract_infix_expression


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--feature-dir",default="artifacts/task1")
    ap.add_argument("--out",default="artifacts/task1_promptfe")
    ap.add_argument("--proposals",type=int,default=12)
    ap.add_argument("--iterations",type=int,default=350)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--model",default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--revision",default=None)
    ap.add_argument("--replay",default=None)
    ap.add_argument("--syntax",choices=["rpn","infix"],default="rpn")
    args=ap.parse_args(); np.random.seed(args.seed);random.seed(args.seed)
    if args.model=="Qwen/Qwen2.5-Coder-1.5B-Instruct" and args.revision is None:
        args.revision="2e1fd397ee46e1388853d2af2c993145b0f1098a"
    start=time.perf_counter(); src=ROOT/args.feature_dir; out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    run_lock=(out/".run.lock").open("a")
    try:
        fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"Another feature-search worker is using {out}; choose a separate --out") from None
    # Test features/labels are not loaded until candidate search/selection ends.
    frames={s:pd.read_parquet(src/f"{s}_features.parquet") for s in ["train","valid"]}
    q={s:pd.read_parquet(src/f"{s}_candidates.parquet") for s in ["train","valid"]}
    cols=manual_columns(frames["train"].columns)
    frames={s:v[cols] for s,v in frames.items()}
    def fit(expressions,env):
        x={s:v.copy() for s,v in frames.items()}
        for j,expression in enumerate(expressions):
            for s in x: x[s][f"constructed_{j}"]=env.generate(expression,s).to_numpy()
        model=CatBoostClassifier(iterations=args.iterations,depth=6,learning_rate=.07,l2_leaf_reg=5,
            random_seed=args.seed,thread_count=6,verbose=False,allow_writing_files=False)
        model.fit(x["train"],q["train"].label,cat_features=[c for c in CATEGORIES if c in cols],
                  eval_set=(x["valid"],q["valid"].label),early_stopping_rounds=50)
        p=model.predict_proba(x["valid"])[:,1]
        met,_=ranking_summary(q["valid"],p,p)
        return model,met
    numeric=[c for c in cols if c not in CATEGORIES]
    # Train-only feature importance chooses the compact prompt vocabulary.
    initial_env=FrameEnvironment(frames,numeric)
    baseline,baseline_metrics=fit([],initial_env)
    gain=dict(zip(baseline.feature_names_,baseline.feature_importances_))
    promptcols=sorted(numeric,key=lambda c:-gain[c])[:24]
    env=FrameEnvironment(frames,promptcols)
    if args.replay:
        import json
        replay=json.loads((ROOT/args.replay).read_text())
        responses=iter([x["response"] for x in replay if x.get("method")=="llm"])
        random_responses=iter([x["expression"] for x in replay if x.get("method")=="random"])
        proposer=lambda prompt:next(responses)
        provenance={"backend":"replay","source":args.replay}
    else:
        proposer=LocalProposer(args.model,args.revision,args.seed)
        provenance=proposer.provenance
    save_json(out/"protocol.json",{"args":vars(args),"expression_engine":"task1.expression_engine:stack-v1",
        "seeds":{"numpy":args.seed,"python":args.seed,"catboost":args.seed,"local_torch":args.seed},
        "prompt_columns":promptcols,"llm":provenance,
        "settings":["KION point-in-time aggregates","chronological holdout",
                   "CatBoost ranking metric","local LLM","test hidden until selection"],
        "random_sampling":"Bounded depth2 generator; replay uses saved random expressions"})
    logs=[]; chosen={}; models={"manual":baseline}; validation={"manual":baseline_metrics}
    for method in ["random","llm"]:
        results={}; attempts=0
        # Same accepted-candidate budget; invalid/duplicate LLM attempts are logged
        # and bounded, not silently substituted with random proposals.
        while len(results)<args.proposals and attempts<args.proposals*4:
            attempts+=1; prompt=None; response=None; expression=None
            t=time.perf_counter()
            try:
                if method=="random": expression=next(random_responses) if args.replay else env.random_expression()
                else:
                    prefix="col_" if args.syntax=="infix" else "col-"
                    features="\n".join(f"{prefix}{i}: {c}; training range [{frames['train'][c].min():.4g}, {frames['train'][c].max():.4g}]" for i,c in enumerate(promptcols))
                    feedback="\n".join(f"{e}: delta NDCG@10={s-baseline_metrics['ndcg@10']:.6f}" for e,s in sorted(results.items(),key=lambda x:-x[1])[:8])
                    syntax=("Return exactly ONE ordinary mathematical expression, at most five operators. "
                        "Use col_0 through col_23 as variables. Use parentheses, +, -, *, / and the listed functions. "
                        "Do not use constants, assignments, arrays, imports, attributes or other functions. "
                        "No code fences or explanation. Example: col_0 / (col_1 + col_2)\n"
                        if args.syntax=="infix" else
                        "Return exactly ONE comma-separated reverse Polish expression with col-ID operands and at most five operators. "
                        "No code, no markdown, no target column. Example: col-0,col-1,/\n")
                    prompt=("Task: rank movies that a user watches in the next 7 days. Input columns are strictly past KION aggregates. "
                        "Counts summarize past activity; shares and recencies describe user preferences. Improve CatBoost NDCG@10.\n"
                        +features+"\nOperators: log, sqrt_abs, min_max, reciprocal (unary); +, -, *, /, mod_column (binary). "
                        +syntax+"Previously evaluated expressions in RPN (do not repeat):\n"+feedback)
                    response=proposer(prompt)
                    expression=(extract_infix_expression(response,len(promptcols)) if args.syntax=="infix" else extract_expression(response))
                env.parse(expression)
                if expression in results: raise ValueError("Duplicate expression")
                trainfeat=env.generate(expression,"train")
                if trainfeat.nunique()<2: raise ValueError("Constant generated feature")
                model,met=fit([expression],env)
                results[expression]=met["ndcg@10"]
                row={"method":method,"attempt":attempts,"expression":expression,"valid":True,
                     "valid_ndcg":met["ndcg@10"],"delta":met["ndcg@10"]-baseline_metrics["ndcg@10"]}
            except Exception as exc:
                row={"method":method,"attempt":attempts,"expression":expression,"valid":False,"error":str(exc)}
                if isinstance(exc,StopIteration):
                    break
            row.update({"seconds":time.perf_counter()-t,"prompt":prompt,"response":response})
            logs.append(row); save_json(out/"proposals.json",logs)
            print("FEATURE SEARCH",{k:v for k,v in row.items() if k not in ["prompt","response"]},flush=True)
        ranked=sorted(results,key=lambda e:-results[e])
        options=[(baseline_metrics["ndcg@10"],[],baseline,baseline_metrics)]
        for n in [1,3,5]:
            if n>len(ranked): continue
            exprs=ranked[:n]; model,met=fit(exprs,env)
            options.append((met["ndcg@10"],exprs,model,met))
        _,chosen[method],models[method],validation[method]=max(options,key=lambda v:v[0])
    save_json(out/"selection.json",{"selected_expressions":chosen,"validation":validation,"criterion":"validation ndcg@10; baseline fallback allowed"})
    # Model + selected expressions now frozen. Load the single final test split.
    test=pd.read_parquet(src/"test_features.parquet")[cols]
    testq=pd.read_parquet(src/"test_candidates.parquet")
    env.frames["test"]=test[promptcols].reset_index(drop=True)
    rows=[]; pers={}
    for name,model in models.items():
        x=test.copy()
        for j,expression in enumerate(chosen.get(name,[])):
            x[f"constructed_{j}"]=env.generate(expression,"test").to_numpy()
        p=model.predict_proba(x)[:,1]; met,per=ranking_summary(testq,p,p)
        rows.append({"method":name,"selected_features":len(chosen.get(name,[])),**met}); pers[name]=per
        model.save_model(str(out/f"{name}.cbm")); per.to_csv(out/f"per_query_{name}.csv",index=False)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    save_json(out/"paired_test_ci.json",{n:paired_bootstrap(pers["manual"],pers[n]) for n in ["random","llm"]})
    save_json(out/"runtime.json",{"total_seconds":time.perf_counter()-start})
    print(rows,flush=True)


if __name__=="__main__": main()
