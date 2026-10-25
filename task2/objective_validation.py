"""Frozen validation replication of the outcome-weighted target diagnostic.

The secondary metric was devised after inspecting the original test. Validation
was used for earlier checkpoint selection: this is temporal consistency analysis,
not a new holdout, new selection rule or off-policy evaluation.
"""
import copy
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from shared.data import ROOT,save_json
from shared.improvement_analysis import markdown
from task2.factorial import load_frozen
from task2.history_mask import MaskedHistoryPolicy
from task2.run import evaluate
from task2.sft_multiseed import data_hash
from task2.relevance_objectives import paired_ratio_ci


def main():
    torch.set_num_threads(4);out=ROOT/"artifacts/task2_objective_validation";out.mkdir(exist_ok=True)
    args,data,mapping,original,rms=load_frozen(ROOT/"artifacts/task2",torch.device("cpu"))
    part=data["valid"]
    expected=json.loads((ROOT/"task2/evaluation_manifest.json").read_text())["array_hashes"]["valid"]
    assert data_hash(part)==expected
    source={
        "original_sft":(["artifacts/task2/sft_best.pt"],False),
        "original_watch_rl":(["artifacts/task2/rl_watch_unconstrained.pt"],False),
        "original_balanced_rl":(["artifacts/task2/rl_balanced_5step.pt"],False),
        "original_continued_sft":(["artifacts/task2/continued_sft_control.pt"],False),
        "masked_sft":(["artifacts/task2/sft_best.pt"],True),
        "capped_graph_masked":(["artifacts/task2_warmstart/binary/sft_best.pt"],True),
        "full_graph_masked":(["artifacts/task2_full_data/full_graph/sft_best.pt"],True),
        "full_monte_carlo":([f"artifacts/task2_conditional_full_reference/monte_carlo_seed{s}.pt" for s in [42,43,44]],True),
        "full_exact":([f"artifacts/task2_conditional_full_reference/exact_seed{s}.pt" for s in [42,43,44]],True),
    }
    earlier=json.loads((ROOT/"artifacts/task2_relevance_objectives/study_protocol.json").read_text())
    assert list(source)==earlier["policies"]
    save_json(out/"study_protocol.json",{
        "purpose":"Temporal consistency check of the same post-hoc outcome-weighted target metric on frozen validation",
        "split":"Original validation09Aug–15Aug; exact original arrays",
        "policies":list(source),"comparisons":earlier["comparisons"],
        "metric":earlier["weighted_metrics"],"groups":earlier["groups"],
        "selection":"None; no training or replacement of any earlier selection",
        "limits":"Validation used for model selection; secondary metric defined after original test. Neither a fresh holdout nor causal value.",
        "validation_array_hash":expected})
    start=time.perf_counter();weights=part["reward"][:,0].astype(float)
    assert np.isfinite(weights).all() and (weights>=0).all() and (weights<=1).all()
    anchors=pd.DataFrame({"user_id":part["user"],"target":part["y"],"watch_fraction":weights})
    anchors.to_csv(out/"target_weights.csv",index=False)
    input_sources={};hashes={};validation_rows=[];per={}
    for name,(paths,masked) in source.items():
        fits=[];input_sources[name]=[]
        for i,path in enumerate(paths):
            model=copy.deepcopy(original);state=torch.load(ROOT/path,map_location="cpu",weights_only=True)
            if any(k.startswith("model.") for k in state):
                assert masked;model=MaskedHistoryPolicy(model);model.load_state_dict(state)
            else:
                model.load_state_dict(state)
                if masked:model=MaskedHistoryPolicy(model)
            metric,frame,top=evaluate(model,part,batch=args.batch)
            seed=42+i;target=out/f"per_user_{name}_{seed}.csv";frame.to_csv(target,index=False)
            input_sources[name].append(str(target.relative_to(ROOT)))
            hashes[path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            validation_rows.append({"policy":name,"seed":seed,**metric})
            fits.append(frame[["user_id","ndcg","hit"]].assign(hit=lambda f:f.hit.astype(float)))
        average=pd.concat(fits).groupby("user_id",as_index=False)[["ndcg","hit"]].mean()
        per[name]=anchors.merge(average,on="user_id",validate="one_to_one",sort=False)
        print("VALIDATION DIAGNOSTIC",name,per[name].ndcg.mean(),flush=True)
    rows=[];effects={}
    groups={"all":np.ones(len(weights),dtype=bool),"watch=0":weights==0,
            "0<watch<0.5":(weights>0)&(weights<.5),"watch>=0.5":weights>=.5}
    for name,frame in per.items():
        for group,mask in groups.items():
            rows.append({"policy":name,"group":group,"users":int(mask.sum()),
                "ndcg@10":frame.loc[mask,"ndcg"].mean(),"recall@10":frame.loc[mask,"hit"].mean()})
        rows.append({"policy":name,"group":"observed_target_watch_weighted","users":len(weights),
            "ndcg@10":float(np.average(frame.ndcg,weights=weights)),
            "recall@10":float(np.average(frame.hit,weights=weights))})
        delta=frame.ndcg-per[earlier["comparisons"][name]].ndcg
        effects[name]={"ndcg":{"unweighted":paired_ratio_ci(delta.to_numpy(),np.ones(len(weights))),
            "target_watch_weighted":paired_ratio_ci(delta.to_numpy(),weights)}}
    metrics=pd.DataFrame(rows);metrics.to_csv(out/"metrics.csv",index=False)
    validation=pd.DataFrame(validation_rows);validation.to_csv(out/"checkpoint_metrics.csv",index=False)
    # Independently recorded earlier validation metrics must be reproduced.
    primary=json.loads((ROOT/"artifacts/task2/selection.json").read_text())["valid_metrics"]
    selected={"original_sft":primary["sft"],"original_watch_rl":primary["rl_watch_unconstrained"],
              "original_balanced_rl":primary["rl_balanced_5step"]}
    full=pd.read_csv(ROOT/"artifacts/task2_full_data/metrics.csv")
    selected["full_graph_masked"]=full[(full.split=="valid")&(full.variant=="full_graph_masked")].iloc[0]
    bandit=pd.read_csv(ROOT/"artifacts/task2_conditional_full_reference/metrics.csv")
    for name,method in [("full_monte_carlo","monte_carlo"),("full_exact","exact")]:
        selected[name]=bandit[(bandit.split=="valid")&(bandit.method==method)].mean(numeric_only=True)
    for name,row in selected.items():
        cell=validation[validation.policy==name]
        for key in ["ndcg@10","recall@10","max_item_user_fraction"]:
            assert np.isclose(cell[key].mean(),row[key],atol=1e-12),name+" / "+key
    save_json(out/"paired_effects.json",effects);save_json(out/"input_sources.json",input_sources)
    hashes["artifacts/task2/examples_valid.npz"]=hashlib.sha256((ROOT/"artifacts/task2/examples_valid.npz").read_bytes()).hexdigest()
    save_json(out/"source_hashes.json",hashes)
    save_json(out/"audit.json",{"counts":{g:int(m.sum()) for g,m in groups.items()},
        "previous_validation_metrics_reproduced":list(selected),"validation_array_hash":expected,
        "checkpoint_count":len(validation),"selection_changed":False})
    save_json(out/"runtime.json",{"seconds":time.perf_counter()-start})
    comparison=[];test=json.loads((ROOT/"artifacts/task2_relevance_objectives/paired_effects.json").read_text())
    for name in ["original_watch_rl","original_balanced_rl","original_continued_sft","full_monte_carlo","full_exact"]:
        for split,values in [("validation",effects),("test",test)]:
            effect=values[name]["ndcg"]["target_watch_weighted"]
            comparison.append({"policy":name,"split":split,"delta":effect["mean"],
                "CI low":effect["ci95"][0],"CI high":effect["ci95"][1]})
    pd.DataFrame(comparison).to_csv(out/"period_comparison.csv",index=False)
    text="# Временная проверка взвешенной next-target метрики\n\n"
    text+="Те же девять фиксированных политик оценены на исходной validation 9–15 августа. Модели не переобучаются, выбор не меняется. Это дополнительный период для диагностики; validation уже использовалась для отбора checkpoint, поэтому её нельзя считать новой независимой выборкой. Вторичная метрика сформулирована после исходного test.\n\n"
    text+="Как и [на test](OBJECTIVE_DIAGNOSTIC.md), считаем `sum(w_i * ndcg_i) / sum(w_i)`, где `w_i` — фактическая доля просмотра записанного next-target. Взвешиваются пользователи; это не reward новой выдачи и не off-policy оценка.\n\n"
    text+="## Изменение к своему reference на двух периодах\n\n"+markdown(pd.DataFrame(comparison))+"\n\n"
    text+="Все интервалы — paired user bootstrap с пересэмплированием weights и denominator. У MC/exact сначала усредняются пользовательские метрики трёх RL seed; SFT/RM фиксированы. Ни дополнительный период, ни вторичная метрика не отменяют исходный строгий validation fallback.\n\n"
    text+="## Полные validation-результаты\n\n"+markdown(metrics)+"\n\n"
    text+="Модели заново предсказали метрики исходной validation: сохранённые раньше результаты контрольных SFT/RL вариантов совпали. Все checkpoint SHA и новые per-user таблицы приложены. Воспроизведение: `python -m task2.objective_validation` после исходных этапов и `task2.relevance_objectives`.\n"
    (ROOT/"task2/OBJECTIVE_VALIDATION.md").write_text(text)
    print(pd.DataFrame(comparison).to_string(index=False))


if __name__=="__main__":main()
