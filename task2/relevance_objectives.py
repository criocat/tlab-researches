"""Post-hoc diagnostic: next-target relevance vs observed target watch fraction.

This evaluates retrieval of the *logged next target*. It is neither off-policy
value nor an observed response to the policy's recommended slate. No fitting,
hyperparameter search, policy choice or changes to the primary metrics occur.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from shared.data import ROOT,save_json
from shared.improvement_analysis import markdown


def paired_ratio_ci(delta,weights,repetitions=2000):
    rng=np.random.default_rng(20261102);values=[]
    for _ in range(repetitions):
        idx=rng.integers(len(weights),size=len(weights));denom=weights[idx].sum()
        if denom>0:values.append(np.dot(delta[idx],weights[idx])/denom)
    return {"mean":float(np.dot(delta,weights)/weights.sum()),
            "ci95":np.quantile(values,[.025,.975]).tolist(),"users":len(weights),
            "scope":"Paired user bootstrap; training seeds, models and observed-target weights fixed"}


def main():
    out=ROOT/"artifacts/task2_relevance_objectives";out.mkdir(exist_ok=True)
    raw=ROOT/"artifacts/task2";bandit=ROOT/"artifacts/task2_conditional_full_reference"
    sources={
        "original_sft":[raw/"per_user_sft.csv"],
        "original_watch_rl":[raw/"per_user_rl_watch_unconstrained.csv"],
        "original_balanced_rl":[raw/"per_user_rl_balanced_5step.csv"],
        "original_continued_sft":[raw/"per_user_continued_sft_control.csv"],
        "masked_sft":[ROOT/"artifacts/task2_masked/per_user_masked_sft_42.csv"],
        "capped_graph_masked":[ROOT/"artifacts/task2_joint/per_user_masked_sft_42.csv"],
        "full_graph_masked":[ROOT/"artifacts/task2_full_data/per_user_full_graph_masked.csv"],
        "full_monte_carlo":[bandit/f"per_user_monte_carlo_{s}.csv" for s in [42,43,44]],
        "full_exact":[bandit/f"per_user_exact_{s}.csv" for s in [42,43,44]],
    }
    baselines={name:("full_graph_masked" if name.startswith("full_") and name!="full_graph_masked" else "original_sft") for name in sources}
    save_json(out/"study_protocol.json",{
        "purpose":"Post-hoc diagnostic of mismatch between next-target and completion objectives",
        "policies":list(sources),"comparisons":baselines,
        "groups":["watch=0","0<watch<0.5","watch>=0.5"],
        "weighted_metrics":"sum(observed_target_watch * per_user_target_metric) / sum(observed_target_watch)",
        "metrics":["ndcg@10","recall@10","target_watch_weighted_ndcg@10","target_watch_weighted_recall@10"],
        "selection":"None; all fixed policies reported, original validation selections unchanged",
        "limits":"Already inspected test; outcome-weighted logged-target retrieval, not off-policy value or observed slate reward. No statistical claim of a causal subgroup effect."})
    arrays=np.load(raw/"examples_test.npz");weights=arrays["reward"][:,0].astype(float)
    assert np.isfinite(weights).all() and (weights>=0).all() and (weights<=1).all()
    anchors=pd.DataFrame({"user_id":arrays["user"],"target":arrays["y"],"watch_fraction":weights})
    anchors.to_csv(out/"target_weights.csv",index=False)
    per={};hashes={}
    for name,paths in sources.items():
        fits=[]
        for path in paths:
            frame=pd.read_csv(path)
            np.testing.assert_array_equal(frame.user_id,anchors.user_id)
            np.testing.assert_array_equal(frame.target,anchors.target)
            fits.append(frame[["user_id","ndcg","hit"]].assign(hit=lambda f:f.hit.astype(float)))
            hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        average=pd.concat(fits).groupby("user_id",as_index=False)[["ndcg","hit"]].mean()
        per[name]=anchors.merge(average,on="user_id",validate="one_to_one",sort=False)
    groups={"all":np.ones(len(weights),dtype=bool),"watch=0":weights==0,
            "0<watch<0.5":(weights>0)&(weights<.5),"watch>=0.5":weights>=.5}
    rows=[];effects={}
    for name,frame in per.items():
        for group,mask in groups.items():
            rows.append({"policy":name,"group":group,"users":int(mask.sum()),
                         "ndcg@10":frame.loc[mask,"ndcg"].mean(),"recall@10":frame.loc[mask,"hit"].mean()})
        rows.append({"policy":name,"group":"observed_target_watch_weighted","users":len(weights),
            "ndcg@10":float(np.dot(weights,frame.ndcg)/weights.sum()),
            "recall@10":float(np.dot(weights,frame.hit)/weights.sum())})
        baseline=per[baselines[name]];effects[name]={}
        for metric in ["ndcg","hit"]:
            delta=(frame[metric]-baseline[metric]).to_numpy()
            effects[name][metric]={"unweighted":paired_ratio_ci(delta,np.ones(len(weights))),
                                  "target_watch_weighted":paired_ratio_ci(delta,weights)}
    metrics=pd.DataFrame(rows);metrics.to_csv(out/"metrics.csv",index=False)
    save_json(out/"paired_effects.json",effects)
    hashes[str((raw/"examples_test.npz").relative_to(ROOT))]=hashlib.sha256((raw/"examples_test.npz").read_bytes()).hexdigest()
    save_json(out/"source_hashes.json",hashes)
    save_json(out/"input_sources.json",{name:[str(p.relative_to(ROOT)) for p in paths] for name,paths in sources.items()})
    save_json(out/"audit.json",{"users":len(weights),"target_weights_observed":int(np.isfinite(weights).sum()),
        "counts":{g:int(mask.sum()) for g,mask in groups.items()},"user_target_identity_verified":True,
        "watch_mass":float(weights.sum()),"selection_changed":False})
    table=[]
    for name in sources:
        cell=metrics[metrics.policy==name].set_index("group")
        table.append({"policy":name,"ordinary NDCG":cell.loc["all","ndcg@10"],
            "watch-weighted mean NDCG":cell.loc["observed_target_watch_weighted","ndcg@10"],
            "ordinary Recall":cell.loc["all","recall@10"],
            "watch-weighted Recall":cell.loc["observed_target_watch_weighted","recall@10"]})
    text="# Разные цели: следующий объект и доля просмотра\n\n"
    text+="Это дополнительная диагностика на уже изученном test. Модели, параметры и validation-выбор сохранены. Публикуются все девять заранее перечисленных политик; веса не подбираются.\n\n"
    text+="Обычная next-target оценка одинаково учитывает любое записанное взаимодействие. В KION 260 из 2000 test-targets имеют нулевую долю просмотра, 1003 — от нуля до 0,5, ещё 737 — не менее 0,5. Поэтому максимизация completion-proxy и извлечение любого следующего target задают разные предпочтения.\n\n"
    text+="Для анализа считаем **среднее per-user target NDCG, взвешенное фактической долей просмотра target**: `sum(w_i * ndcg_i) / sum(w_i)`. Аналогично считаем weighted Recall. Это веса между пользователями, а не переопределённый graded NDCG внутри запроса. Они используют наблюдаемый outcome записанной цели; это не reward нового рекомендованного списка и не off-policy оценка ценности.\n\n"
    text+=markdown(pd.DataFrame(table))+"\n\n"
    text+="## Парные изменения к соответствующему reference\n\n"
    comparisons=[]
    for name in ["original_watch_rl","original_balanced_rl","original_continued_sft","full_monte_carlo","full_exact"]:
        for metric in ["unweighted","target_watch_weighted"]:
            e=effects[name]["ndcg"][metric]
            comparisons.append({"policy":name,"baseline":baselines[name],"metric":metric,"delta":e["mean"],
                "CI low":e["ci95"][0],"CI high":e["ci95"][1]})
    text+=markdown(pd.DataFrame(comparisons))+"\n\n"
    text+="MC/exact на full graph reference имеют положительный watch-weighted point estimate и условные user-bootstrap CI. Это согласуется с несовпадением обучаемых целей, но не доказывает причинного улучшения рекомендаций. Веса и вторичная метрика исследованы после исходного test; множественные проверки и фиксированные SFT/RM ограничивают CI. Никакой новый test-победитель не назначается: строгий validation-отбор сохраняет SFT.\n\n"
    text+="## Группы наблюдаемой доли просмотра\n\n"
    text+=markdown(metrics[metrics.group.isin(["watch=0","0<watch<0.5","watch>=0.5"])])+"\n\n"
    text+="Группы определены последующим наблюдаемым outcome, поэтому различия между ними не являются причинной оценкой эффекта. При трёх RL seed сначала усредняются пользовательские метрики, затем пересэмплируются пользователи вместе с их весами. Вариативность обучения SFT/RM этим интервалом не охватывается.\n\n"
    text+="Воспроизведение: `python -m task2.relevance_objectives`. Нужны сохранённые per-user CSV и исходный `artifacts/task2/examples_test.npz`; для чтения и проверки взвешенных сумм достаточно небольших таблиц из архива.\n"
    if (ROOT/"task2/OBJECTIVE_VALIDATION.md").exists():
        text+="\n[Проверка на validation](OBJECTIVE_VALIDATION.md) дала более слабый weighted прирост MC/exact с CI через ноль. Эффект на втором периоде не подтверждён убедительно; исходная validation также уже использовалась для выбора checkpoint.\n"
    (ROOT/"task2/OBJECTIVE_DIAGNOSTIC.md").write_text(text)
    print(metrics.query("group in ['all','observed_target_watch_weighted']").to_string(index=False))


if __name__=="__main__":main()
