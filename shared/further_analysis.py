"""Summarize the frozen exploratory studies; never selects a new primary model."""
import json
import numpy as np
import pandas as pd
from shared.data import ROOT,save_json
from shared.analyze import bootstrap


def factorial():
    base=ROOT/"artifacts/task2_factorial";m=pd.read_csv(base/"metrics.csv");test=m[m.split=="test"].copy()
    assert len(test)==12 and len(m)==24,"Factorial study is incomplete"
    original=pd.read_csv(ROOT/"artifacts/task2/metrics.csv").set_index("model")
    for h,p,name in [(1,0.,"rl_watch_kl01"),(5,.2,"rl_balanced_5step")]:
        row=test[(test.seed==42)&(test.horizon==h)&(test.penalty==p)].iloc[0]
        for key in ["ndcg@10","recall@10","predicted_watch@10","history_repeat@10"]:
            assert np.isclose(row[key],original.loc[name,key],atol=1e-10),(name,key)
    tables={};ro=[]
    for _,r in test.iterrows():
        per=pd.read_csv(base/r["name"]/"per_user_test.csv").sort_values("user_id")
        assert per.user_id.is_unique and len(per)==2000
        key=(int(r.horizon),float(r.penalty));tables.setdefault(key,[]).append(per)
        steps=json.loads((base/r["name"]/"rollout.json").read_text())["steps"]
        ro.append({"name":r["name"],"seed":int(r.seed),"horizon":int(r.horizon),"penalty":float(r.penalty),**steps[-1]})
    pd.DataFrame(ro).to_csv(base/"fifth_step.csv",index=False)
    averaged={}
    for key,frames in tables.items():
        assert all(frames[0].user_id.equals(f.user_id) for f in frames)
        averaged[key]=np.stack([f.ndcg.to_numpy() for f in frames]).mean(0)
    effects={}
    for h in [1,5]:effects[f"repeat_penalty_h{h}"]=bootstrap(averaged[(h,.2)]-averaged[(h,0.)])
    for p in [0.,.2]:effects[f"horizon_penalty{p}"]=bootstrap(averaged[(5,p)]-averaged[(1,p)])
    effects["interaction"]=bootstrap((averaged[(5,.2)]-averaged[(5,0.)])-(averaged[(1,.2)]-averaged[(1,0.)]))
    save_json(base/"analysis.json",{"original_seed42_cells_reproduced":True,"effects_ndcg":effects,
        "uncertainty":"Average per-user metrics across 3 RL seeds, then bootstrap users; conditional on frozen SFT/RM and one test week; not ensemble ranking",
        "interpretation":"Post-hoc controlled diagnostic. All cells reported; no new primary selection"})


def temporal():
    base=ROOT/"artifacts/task1_temporal";rows=pd.read_csv(base/"summary.csv")
    assert len(rows)==3,"Temporal study is incomplete"
    a=json.loads((ROOT/"artifacts/task1/analysis.json").read_text())
    s=json.loads((ROOT/"artifacts/task1/selection.json").read_text())
    rows["role"]="exploratory"
    primary={"test_cutoff":"2021-08-16","valid_cutoff":"2021-08-09","selected":s["selected"],
        "features":len(s["columns"]),"manual_ndcg":a["mean_test_metrics"]["manual"]["ndcg@10"],
        "auto_ndcg":a["mean_test_metrics"]["auto_selected"]["ndcg@10"],"role":"original primary",
        **a["seed_averaged_paired_ndcg_delta"]}
    all_rows=pd.concat([rows,pd.DataFrame([primary])],ignore_index=True)
    all_rows.to_csv(base/"all_windows.csv",index=False)


def markdown_appendices():
    from shared.report import table
    t=pd.read_csv(ROOT/"artifacts/task1_temporal/all_windows.csv")
    text="# Дополнительная временная проверка\n\n"
    text+="Этот анализ добавлен после знакомства с исходным test. Грамматика, hyperparameters, выбор пользователей и правила кандидатов сохранены; в каждой новой неделе модель обучается заново на прошлом и выбирает признаки только на своём validation. Исходный основной результат не заменяется.\n\n"
    text+=table(t[["test_cutoff","manual_ndcg","auto_ndcg","mean","ci95","features","role"]])+"\n\n"
    text+="![Временная устойчивость](../figures/t1_temporal.svg)\n\n"
    text+="Пользователи и train-периоды перекрываются. Абсолютный NDCG между неделями зависит от состава событий и кандидатов; корректное сравнение методов — внутри каждой недели. Процедура поиска окон обязательно выбирает хотя бы один дополнительный блок: manual fallback здесь отсутствует, в отличие от поиска выражений. Это ограничение дизайна, а не гарантия выигрыша на validation или test.\n\n"
    text+="Команда: `python -m task1.temporal_study`. Настройки и границы времени: `artifacts/task1_temporal/study_protocol.json`, `test_*/time_boundaries.json`.\n"
    (ROOT/"task1/EXPLORATORY.md").write_text(text)
    m=pd.read_csv(ROOT/"artifacts/task2_factorial/metrics.csv");m=m[m.split=="test"]
    grouped=m.groupby(["horizon","penalty"])[["ndcg@10","predicted_watch@10","history_repeat@10"]].agg(["mean","std"])
    grouped.columns=[" ".join(c) for c in grouped.columns]
    text="# Дополнительные проверки RL и генерации\n\n"
    text+="Протоколы зафиксированы до новых запусков, но после просмотра основного test. Это исследовательская диагностика; все клетки публикуются, новая primary model не выбирается.\n\n## Горизонт × штраф\n\n"
    text+="KL=.1, CE=.2, 200 обновлений, 3 согласованных RL seed (42/43/44), неизменные SFT/RM. Для seed 42 обе исходные клетки точно повторили прежние метрики. Seeds меняют minibatches и sampling, а не полное обучение системы.\n\n"
    text+=table(grouped.reset_index())+"\n\n"
    a=read_json(ROOT/"artifacts/task2_factorial/analysis.json")
    effects=pd.DataFrame([{"effect":k,**v} for k,v in a["effects_ndcg"].items()])
    text+=table(effects)+"\n\n![Контролируемый эксперимент](../figures/t2_factorial.svg)\n\n"
    text+="При фиксированных настройках увеличение горизонта уменьшает NDCG; изолированный эффект repeat penalty мал и CI пересекает ноль. Bootstrap усредняет метрику каждого пользователя по трём RL seeds, затем ресэмплирует пользователей. Равны обновления, а не число действий: 12 800 для H=1 против 64 000 для H=5. Масштаб return также меняется. Это не универсальное утверждение о длинных горизонтах.\n\n## Декодирование пяти объектов\n\n"
    d=pd.read_csv(ROOT/"artifacts/task2_decoding/metrics.csv")
    text+=table(d[["policy","mode","hit@5","ndcg@5","mean_unique_items","max_item_user_fraction"]])+"\n\n"
    text+="![Декодирование](../figures/t2_decoding.svg)\n\n"
    text+="В обычном top-5 контекст не меняется. При авторегрессии предполагается потребление сгенерированного объекта; no-repeat лишь исключает уже выданные ID. Исторические объекты дополнительно не маскируются. Релевантность учитывает первое появление единственного next target, дубликаты не дают повторного gain. Эти slate@5 метрики не заменяют исходный full-catalog@10.\n\n"
    text+="SFT Hit@5: .1165 / .0540 / .1155 для one-shot / autoregressive / no-repeat. Запрет дублей восстанавливает большую часть потери. Но у unconstrained RL один item остаётся в 99.25% уникальных списков; устранение дублей не устраняет концентрацию между пользователями. Это офлайн-диагностика и не доказательство retention-эффекта.\n\n"
    text+="Команды: `python -m task2.factorial`, `python -m task2.decoding_study`. Согласованные парные CI и все per-user результаты находятся в соответствующих `artifacts/`.\n"
    (ROOT/"task2/EXPLORATORY.md").write_text(text)


def read_json(path):return json.loads(path.read_text())


if __name__=="__main__":
    factorial();temporal();markdown_appendices();print("Further studies summarized; primary selections unchanged")
