"""Summarize measured improvement studies without altering primary artifacts."""
import json
import numpy as np
import pandas as pd
from shared.data import ROOT, save_json
from shared.analyze import bootstrap
from shared.metrics import paired_bootstrap


def read(path): return json.loads((ROOT/path).read_text())


def markdown(frame, index=False):
    if index: frame=frame.reset_index()
    def clean(value):
        if isinstance(value,(float,np.floating)): return f"{value:.6f}" if np.isfinite(value) else "—"
        return str(value).replace("|","\\|").replace("\n"," ")
    rows=[list(frame.columns)]+[list(row) for row in frame.itertuples(index=False,name=None)]
    lines=["| "+" | ".join(clean(v) for v in rows[0])+" |",
           "| "+" | ".join(["---"]*len(frame.columns))+" |"]
    lines.extend("| "+" | ".join(clean(v) for v in row)+" |" for row in rows[1:])
    return "\n".join(lines)


def main():
    graph = ROOT/"artifacts/task1_relational"
    m = pd.read_csv(graph/"metrics.csv"); test = m[m.split == "test"]
    means = test.groupby("model").mean(numeric_only=True)
    temporal = []
    for date, folder in [("2021-07-26", "artifacts/task1_relational_temporal/test_2021-07-26"),
                         ("2021-08-02", "artifacts/task1_relational_temporal/test_2021-08-02"),
                         ("2021-08-09", "artifacts/task1_relational_temporal/test_2021-08-09"),
                         ("2021-08-16", "artifacts/task1_relational")]:
        frame = pd.read_csv(ROOT/folder/"metrics.csv"); g = frame[frame.split == "test"].groupby("model").mean(numeric_only=True)
        effect = read(folder+"/paired_test_ci.json")["ndcg"]
        temporal.append({"test_cutoff": date, "manual": g.loc["manual","ndcg@10"],
                         "graph": g.loc["graph_selected","ndcg@10"], "delta": effect["delta"],
                         "ci_low": effect["ci95"][0], "ci_high": effect["ci95"][1]})
    pd.DataFrame(temporal).to_csv(graph/"temporal_summary.csv", index=False)
    analysis = {"selected": read("artifacts/task1_relational/selection.json")["selected"],
        "mean_test": means.to_dict(orient="index"), "effect": read("artifacts/task1_relational/paired_test_ci.json"),
        "positive_weeks": sum(r["delta"] > 0 for r in temporal), "weeks": len(temporal),
        "interpretation": "Three CatBoost seeds averaged per query before bootstrap; graph RNG fixed42; overlapping retrospective weeks"}
    save_json(graph/"analysis.json", analysis)
    rank = pd.read_csv(ROOT/"artifacts/task1_ranking/metrics.csv")
    rank_summary = rank[rank.split == "test"].groupby("family").agg(
        ndcg=("ndcg@10","mean"), ndcg_sd=("ndcg@10","std"), recall=("recall@10","mean"), features=("features","first"))
    rank_summary.to_csv(ROOT/"artifacts/task1_ranking/summary.csv")
    selected = read("artifacts/task1_ranking/selection.json")["families"]
    text = "# Улучшения: обучаемые связи и проверка сильного baseline\n\n"
    text += "Все эксперименты этого приложения exploratory: исходный тест уже обсуждался. Внутри каждого исследования выбор использует validation; исходные результаты не заменены.\n\n"
    text += f"Графовые признаки: manual {means.loc['manual','ndcg@10']:.6f}, graph {means.loc['graph_selected','ndcg@10']:.6f}. Delta {analysis['effect']['ndcg']['delta']:+.6f}, 95% CI {analysis['effect']['ndcg']['ci95']}.\n\n"
    text += "16 обучаемых численных признаков: binary/normalized SVD dot/cos для16/32/64 координат; суммы и средние item-neighbour сходств top50/200. Каждый граф строго до snapshot, labels не используются. Скалярные similarity признаки не требуют выравнивания ориентации отдельных SVD между датами.\n\n"
    text += markdown(pd.DataFrame(temporal))+"\n\n"
    text += "Ранние недели, пользователи и train-периоды перекрываются. Доверительные интервалы условны на выборке и заданном поиске, без поправки на множественные исследования.\n\n"
    text += "Одинаковый grid для четырёх feature families: Logloss/YetiRank × depth4/6, до500trees, earlystop по NDCG@10.\n\n"+markdown(rank_summary,index=True)+"\n\n"
    text += "Validation-победители: "+", ".join(f"{name}: {v['objective']} depth{v['depth']}" for name,v in selected.items())+".\n\n"
    text += "Полные параметры и CSV — artifacts/task1_relational, task1_relational_temporal, task1_ranking.\n"
    scale_rows=[]
    for folder,label,has_split in [("task1_relational","50k histories / original queries",True),
                                   ("task1_population","200k histories / original queries",True),
                                   ("task1_cohort","200k histories / additional query users",False)]:
        frame=pd.read_csv(ROOT/"artifacts"/folder/"metrics.csv")
        if has_split:frame=frame[frame.split=="test"]
        average=frame.groupby("model").mean(numeric_only=True)
        graph_name="graph_selected" if folder=="task1_relational" else "graph"
        effect=read("artifacts/"+folder+"/paired_test_ci.json")["ndcg"]
        scale_rows.append({"setting":label,"manual":average.loc["manual","ndcg@10"],
            "graph":average.loc[graph_name,"ndcg@10"],"delta":effect["delta"],"ci_low":effect["ci95"][0],"ci_high":effect["ci95"][1]})
    pd.DataFrame(scale_rows).to_csv(ROOT/"artifacts/task1_population/summary.csv",index=False)
    decayed=pd.read_csv(ROOT/"artifacts/task1_decayed_graph/metrics.csv")
    decayed[decayed.split=="test"].groupby("family")[["ndcg@10","recall@10","map@10"]].mean().to_csv(ROOT/"artifacts/task1_decayed_graph/summary.csv")
    text += "\n## Историческая популяция и отдельная когорта оценочных пользователей\n\n"
    text += "К исходным 50 000 пользователям добавлены 150 000 случайных пользователей. Сохранены все строки их истории — всего 1 136 830. Обе модели видят одинаковую историю 200k пользователей. Обучающие запросы, labels и исходные test-кандидаты сохранены; 16 графовых признаков и параметры заморожены. Популяция не выбирается по test.\n\n"
    text += markdown(pd.DataFrame(scale_rows))+"\n\n"
    text += "Последняя строка — отдельные пользователи вне исходных 50k, без пересечения обучающих и валидационных запросов. Их прошлые связи использованы в графе: это warm-проверка. Сохранённые модели оцениваются без нового обучения или подбора. Каталог 200k шире исходного, поэтому сравниваются manual и graph внутри когорты. Датасет и test-неделя те же; проверка остаётся исследовательской.\n"
    text+="\n## Временное затухание и равный бюджет поиска\n\nДля графа проверены периоды полураспада ∞/10/21/60 дней и пять popularity coefficients; для manual — L2 1/5/20/100 и те же пять коэффициентов. Каждому семейству доступны 20 валидационных настроек. Выбраны граф без затухания и manual с L2=5. На новой полнокаталожной группе из 512 пользователей вне прежних test/validation групп: graph 0,139182 против manual 0,119677, Δ +0,019504, CI [0,007406; 0,031898]. Затухание не улучшило validation; более удачный вариант по test не выбирается.\n"
    full=pd.read_csv(ROOT/"artifacts/task1_full_catalog/metrics.csv")
    full_summary=full.groupby(["protocol","model"])[["ndcg@10","recall@10","map@10"]].mean().reset_index()
    full_summary.to_csv(ROOT/"artifacts/task1_full_catalog/summary.csv",index=False)
    calibrated=pd.read_csv(ROOT/"artifacts/task1_catalog_calibration/metrics.csv")
    calibrated[calibrated.split=="test"].groupby("model")[["ndcg@10","recall@10","map@10"]].mean().to_csv(ROOT/"artifacts/task1_catalog_calibration/summary.csv")
    temporal_catalog=[]
    for folder,date,mode in [("task1_rolling_catalog","2021-08-09","calibrated"),("task1_full_catalog_replication","2021-08-16","calibrated_full")]:
        frame=pd.read_csv(ROOT/f"artifacts/{folder}/metrics.csv")
        selected=frame[(frame["mode"]==mode)&(frame.split=="test")] if "mode" in frame else frame[frame.protocol==mode]
        means=selected.groupby("model")[["ndcg@10","recall@10"]].mean()
        effect=read(f"artifacts/{folder}/paired_calibrated_ci.json")["ndcg"]
        temporal_catalog.append({"date":date,"manual_ndcg":means.loc["manual","ndcg@10"],"graph_ndcg":means.loc["graph","ndcg@10"],
            "delta":effect["delta"],"ci_low":effect["ci95"][0],"ci_high":effect["ci95"][1]})
    pd.DataFrame(temporal_catalog).to_csv(ROOT/"artifacts/task1_rolling_catalog/summary.csv",index=False)
    gains={}
    for name in ["manual","graph"]:
        parts={}
        for label,folder in [("raw","task1_full_catalog"),("calibrated","task1_catalog_calibration")]:
            frames=[pd.read_csv(ROOT/f"artifacts/{folder}/per_query_{name}_{seed}.csv") for seed in [42,43,44]]
            parts[label]=pd.concat(frames).groupby("query_id",as_index=False)[["ndcg","recall","map"]].mean()
        gains[name]=paired_bootstrap(parts["raw"],parts["calibrated"])
    save_json(ROOT/"artifacts/task1_catalog_calibration/own_baseline_ci.json",gains)
    text+="\n## Полный каталог: важная граница результата\n\n"
    text+="Для 128 случайных пользователей из дополнительной когорты без дообучения перечислены все известные объекты, кроме прошлой истории. Targets прикреплены после перечисления; состав кандидатов от них не зависит. Получено 1 472 234 пары. Сравнение со 100 негативами выполнено на тех же пользователях.\n\n"
    text+=markdown(full_summary)+"\n\n"
    text+="Преимущество graph в условном ранжировании не перенеслось убедительно на полный каталог: Δ NDCG около −0,00128 с широким CI [−0,03319; 0,02936]. Улучшение на 100 негативах нельзя обобщать на полную выдачу. Возможные объяснения — изменение распределения негативных кандидатов и редкие нерелевантные объекты с высоким score. Одно изменение доли классов само по себе не меняет ранги.\n\n"
    text+="После этой диагностики отдельный опыт подбирает один popularity coefficient по полнокаталожной validation другой недели. Одинаковая сетка 0/0,1/0,25/0,5/1 доступна manual и graph, без переобучения. Это эвристическая коррекция score; она не восстанавливает неизвестные propensity. Исходный test уже обсуждался.\n\n"
    text+=markdown(calibrated[calibrated.split=="test"].groupby("model")[["ndcg@10","recall@10"]].mean(),index=True)+"\n"
    text+="\n## Полный каталог на более раннем периоде\n\nДля test 9 августа модели переобучены на соответствующих прошлых snapshots с validation 2 августа. Коэффициент alpha заново выбирается 2 августа и не переносится из будущего. Новая warm-когорта из 512 пользователей исключает прежние полнокаталожные test/validation группы.\n\n"+markdown(pd.DataFrame(temporal_catalog))+"\n\nИсторическая популяция общая, датасет уже известен. Два периода ограничивают силу обобщения.\n"
    (ROOT/"task1/IMPROVEMENTS.md").write_text(text)
    mask = pd.read_csv(ROOT/"artifacts/task2_masked/metrics.csv")
    mask_summary = mask[mask.split == "test"].groupby("name").mean(numeric_only=True)
    mask_summary.to_csv(ROOT/"artifacts/task2_masked/summary.csv")
    analytic = pd.read_csv(ROOT/"artifacts/task2_analytic/metrics.csv")
    warm = pd.read_csv(ROOT/"artifacts/task2_warmstart/metrics.csv")
    ctx = pd.read_csv(ROOT/"artifacts/task2_context_reward/metrics.csv")
    truth = pd.read_csv(ROOT/"artifacts/task2/reward_test_predictions.csv")
    pred = pd.read_csv(ROOT/"artifacts/task2_context_reward/predictions_test.csv")
    # Positional arrays share the exact original held-out examples; verify truth.
    # The original prediction CSV field names are inspected explicitly below.
    target_col = next(c for c in ["watch_true", "true_watch", "watch", "truth"] if c in truth)
    prediction_col = next(c for c in ["watch_reward_pred", "watch_pred", "pred_watch", "prediction"] if c in truth)
    np.testing.assert_array_equal(truth.user_id, pred.user_id)
    np.testing.assert_allclose(truth[target_col], pred.truth, equal_nan=True)
    observed = pred.truth.notna()
    effect = bootstrap((truth.loc[observed,target_col]-truth.loc[observed,prediction_col]).abs().to_numpy()
        -(pred.loc[observed,"truth"]-pred.loc[observed,"prediction"]).abs().to_numpy())
    save_json(ROOT/"artifacts/task2_context_reward/paired_mae_improvement.json", effect)
    text = "# Улучшения: функция reward, допустимые действия и представления\n\n"
    text += "Дополнительные исследования после исходного test; validation-only выбор внутри исследования, без замены primary.\n\n"
    text += "## Reward redesign\n\nMean/min ensemble не прошёл строгие guardrails, item-centred reward уменьшил концентрацию ценой raw watch. Правило выбрало SFT fallback. Старый auditor использован в optimized ensemble; третий monitor не участвует в выборе, но разделяет наблюдательные данные. Минимум не является калиброванной статистической границей.\n\n"
    text += "Аналитический comparator — inference reweight на SFT top100: q∝p_ref exp(alpha*r), exact optimum conditional single-step reward−KL/alpha. Выбран original RM alpha.25. Это не обученная RL-модель.\n\n"
    text += markdown(analytic[analytic.split == "test"][["mode","alpha","ndcg@10","recall@10","predicted_watch@10","max_item_user_fraction"]])+"\n\n"
    text += "## Исключение известного контекста\n\nМаскируются максимум32 IDs, одинаково для SFT/RL. Cold targets остаются misses. Нулевые повторы заданы support constraint, не выучены RL. KION агрегирует уникальные user–item пары; не переносить правило без проверки на повторное потребление.\n\n"
    text += markdown(mask_summary[["ndcg@10","recall@10","predicted_watch@10","max_item_user_fraction"]],index=True)+"\n\n"
    text += "Для selected RL средние по3 sampling seeds, SFT/RM фиксированы. NDCG относительно masked SFT почти неизменен, Recall снижается; относительно raw SFT значительная часть улучшения объясняется маской. Полный paired CI JSON включён.\n\n"
    text += "## Инициализация из графа\n\nSVD train graph инициирует tied embeddings, без изменения архитектуры, переходов и8epochs SFT. Один SFT seed; оба заранее заданных варианта опубликованы.\n\n"
    text += markdown(warm[warm.split == "test"][["name","ndcg@10","recall@10","predicted_watch@10","coverage@10"]])+"\n\n"
    text += "## Более информативная модель отклика\n\nCatBoost использует прошлые watched fractions, полную предыдущую историю и статические категории. Это расширение данных и архитектуры одновременно. Все агрегаты строго до дня target; same-day события исключены; неизвестные item stats/metadata остаются неизвестными. Выбор depth4/6/8 только по validation RMSE.\n\n"
    text += markdown(ctx[ctx["selected"].fillna(False).astype(bool)])+"\n\n"
    text += f"MAE original RM − enriched RM на тех же users: {effect['mean']:+.6f}, CI {effect['ci95']}. Это точность по наблюдаемым взаимодействиям, не качество на непоказанных объектах.\n"
    for folder, title in [("task2_context_rl", "RL с более точной RM"), ("task2_joint", "Графовая инициализация + маска + более точная RM")]:
        data = pd.read_csv(ROOT/"artifacts"/folder/"metrics.csv")
        sel = read("artifacts/"+folder+"/selection.json")["selected"]
        text += f"\n## {title}\n\nValidation выбрала {sel}. Правило требует NDCG и Recall >=99% baseline и рост концентрации <=2п.п.; выбор не меняется по test.\n\n"
        text += markdown(data[data.split=="valid"][["name","seed","ndcg@10","recall@10","context_rm_watch@10"]])+"\n\n"
        text += markdown(data[data.split=="test"][["name","seed","ndcg@10","recall@10","context_rm_watch@10"]])+"\n"
    original_per = pd.read_csv(ROOT/"artifacts/task2/per_user_sft.csv")
    joint_per = pd.read_csv(ROOT/"artifacts/task2_joint/per_user_masked_sft_42.csv")
    j = original_per.merge(joint_per,on="user_id",suffixes=("_a","_b"),validate="one_to_one")
    save_json(ROOT/"artifacts/task2_joint/versus_raw_sft.json", {"ndcg": bootstrap(j.ndcg_b-j.ndcg_a),
                                                               "recall": bootstrap(j.hit_b.astype(float)-j.hit_a.astype(float))})
    (ROOT/"task2/IMPROVEMENTS.md").write_text(text)
    full_data=pd.read_csv(ROOT/"artifacts/task2_full_data/metrics.csv")
    full_data[full_data.split=="test"].to_csv(ROOT/"artifacts/task2_full_data/summary.csv",index=False)
    within={}
    for masked in [False,True]:
        suffix="masked" if masked else "raw"
        a=pd.read_csv(ROOT/f"artifacts/task2_full_data/per_user_full_random_{suffix}.csv")
        b=pd.read_csv(ROOT/f"artifacts/task2_full_data/per_user_full_graph_{suffix}.csv")
        j=a.merge(b,on="user_id",suffixes=("_a","_b"),validate="one_to_one")
        within[suffix]={"ndcg":bootstrap(j.ndcg_b-j.ndcg_a),"recall":bootstrap(j.hit_b.astype(float)-j.hit_a.astype(float))}
    save_json(ROOT/"artifacts/task2_full_data/within_full_initializer_ci.json",within)
    text+="\n## Контроль полноты обучающих SFT данных\n\nВосстановлены все 181 733 обучающих перехода; исходные validation/test массивы по 2000 пользователей сохранены неизменными. Capped train из 100k переходов является точным подмножеством полного train. SVD дополнительно использует все train-связи, включая singleton/first-day события без next-day target. Это дополнительный этап обучения без labels, поэтому эффект нельзя объяснять только численной инициализацией.\n\n"
    text+=markdown(full_data[full_data.split=="test"][["variant","ndcg@10","recall@10","coverage@10","max_item_user_fraction"]])+"\n\n"
    text+="Все варианты обучаются восемь эпох; полный train даёт больше SGD updates. Сравнение cap поэтому не изолирует только объём данных. Внутри full обе инициализации имеют одинаковые обучающие примеры и бюджет оптимизатора. Validation выбрала full graph для raw и masked выдачи. Здесь один CPU SFT seed.\n"
    final_bandit=pd.read_csv(ROOT/"artifacts/task2_conditional_full_reference/metrics.csv")
    final_bandit[final_bandit.split=="test"].groupby("method")[["ndcg@10","recall@10","context_rm_watch@10","max_item_user_fraction"]].mean().to_csv(ROOT/"artifacts/task2_conditional_full_reference/summary.csv")
    text+="\n## Повтор conditional bandit на более сильном reference\n\nСохранены MC/exact objective, learning rate, KL, CE, число шагов и 100k fine-tuning контекстов. Новый reference — выбранная по validation graph SFT на всех 181 733 переходах.\n\n"
    text+=markdown(final_bandit[final_bandit.split=="test"].groupby("method")[["ndcg@10","recall@10","context_rm_watch@10"]].mean(),index=True)+"\n\nСтрогий validation-отбор снова оставил reference; test не меняет выбранную политику.\n"
    (ROOT/"task2/IMPROVEMENTS.md").write_text(text)
    ablation=pd.read_csv(ROOT/"artifacts/task2_reward_ablation/metrics.csv")
    variance=pd.read_csv(ROOT/"artifacts/task2_reward_ablation/variance.csv")
    text+="\n## Что даёт информация только о состоянии\n\nUser-only и item-only RM имеют одинаковую сетку depth 4/6 и выбор по validation RMSE. Full RM выбрана ранее; depth 6 также выигрывает среди 4/6.\n\n"
    text+=markdown(ablation[ablation.split=="test"][["family","mae","rmse","r2"]])+"\n\n"
    text+="User-only R²≈0,040, item-only≈0,076, full≈0,166. Гипотеза, что весь прирост full RM объясняется лишь склонностью пользователя к просмотру, не подтверждается. Однако user-only reward постоянен для всех кандидатов и точно сокращается в leave-one-out advantage.\n\n"
    text+=markdown(variance)+"\n\n"
    text+="Вариация разложена на 256 фиксированных test-контекстах и их SFT top-100 с равными весами кандидатов. Среднее между контекстами зависит также от состава кандидатов. Зависимость прогноза от кандидата не является причинным эффектом рекомендации и не гарантирует качество за пределами наблюдаемых действий.\n"
    bandit=pd.read_csv(ROOT/"artifacts/task2_conditional_bandit/metrics.csv")
    bandit_summary=bandit[bandit.split=="test"].groupby("method")[["ndcg@10","recall@10","context_rm_watch@10","max_item_user_fraction"]].mean()
    bandit_summary.to_csv(ROOT/"artifacts/task2_conditional_bandit/summary.csv")
    text+="\n## Условный bandit objective: MC и точный градиент\n\nОдинаковые frozen reference top-100, контексты, 200 обновлений, lr=3e-5, KL=0,5, CE=0,2 и три matched seed; общий кеш reward. Exact вычисляет ожидание по 100 действиям, MC использует четыре samples с leave-one-out baseline. Оптимизируется условный surrogate, а не полный expected reward.\n\n"+markdown(bandit_summary,index=True)+"\n\n"
    text+="Строгий validation-отбор оставил reference. Оба заранее заданных метода опубликованы на test как диагностика. Условные NDCG CI укладываются в заданную до результатов границу потери 0,005. Это не онлайн-оценка и не основание отменить validation fallback.\n"
    ret=pd.read_csv(ROOT/"artifacts/task2_return_reward/metrics.csv")
    ret[ret.split=="test"].to_csv(ROOT/"artifacts/task2_return_reward/summary.csv",index=False)
    text+="\n## Предсказание следующего активного дня\n\nТе же маски цензурирования трёхдневного окна и шесть заранее заданных обучений: depth 4/6 для user-only, item-only и full RM. Выбор по validation Brier.\n\n"+markdown(ret[ret.split=="test"][["model","auc","brier","logloss","n"]])+"\n\n"
    text+="Полная RM предсказывает proxy лучше исходной, но user-only почти столь же точна. User-only не различает действия; высокая предсказуемость возвращения не доказывает причинного влияния рекомендации. Возврат здесь — следующий записанный день новой user–item пары, а не наблюдение всех визитов. Для оценки влияния понадобятся exposure-логи, интервенции и онлайн-наблюдение.\n"
    replication=ROOT/"artifacts/task2_sft_replication/summary.csv"
    if replication.exists():
        gpu=pd.read_csv(replication)
        text+="\n## GPU: три SFT seed и перемешанный initializer\n\n18 обучений на Tesla T4: 100k/181 733 перехода × random/graph/permuted graph × seed 42/43/44, по восемь эпох. SHA всех train/validation/test массивов совпадает с CPU-протоколом. Checkpoint каждого запуска и средняя инициализация выбираются по raw validation NDCG; для обоих cap выбран graph.\n\n"
        text+=markdown(gpu[["training","method","masked","ndcg@10","ndcg@10_seed_sd","recall@10","max_item_user_fraction"]])+"\n\n"
        text+="Graph raw улучшает NDCG относительно random во всех трёх paired seed каждого cap. После history mask дополнительный выигрыш мал и неустойчив: capped Δ +0,002656, CI [−0,000707; 0,006215]; full Δ +0,000948, CI [−0,002632; 0,004548] после усреднения seed по каждому пользователю. Эти CI не отражают всю вариативность обучения. Дополнительно сохранён парный crossed bootstrap seed и пользователей как оценка чувствительности при всего трёх seed.\n\n"
        text+="Перемешивание строк сохраняет распределение и спектр initializer, но разрушает соответствие item, включая popularity; это не изолированный тест семантики. Raw permuted близок к random, но masked permuted конкурентен graph. Сильный CPU full graph masked 0,116802 одного seed не выдаётся за средний эффект всей системы. CPU/GPU результаты приведены отдельно: Torch и устройства различаются, одинаковый seed не гарантирует одинаковое обучение. Датасет/test уже изучены; это повторная проверка на них, а не новый внешний benchmark.\n\nКоманда: `python -m task2.sft_multiseed [--full-train]`. Оба приватных ноутбука и неизменные source manifests находятся в `kaggle/sft*_multiseed/`.\n"
    if (ROOT/"task2/OBJECTIVE_DIAGNOSTIC.md").exists():
        text+="\n## Несовпадение целей\n\n[Отдельная диагностика](OBJECTIVE_DIAGNOSTIC.md) сравнивает извлечение записанного next-target с тем же средним, взвешенным наблюдаемой долей просмотра target. Все девять фиксированных политик и группы опубликованы. Это вторичная post-hoc метрика; она не оценивает reward новой выдачи и не меняет validation-выбор.\n"
    if (ROOT/"task2/OBJECTIVE_VALIDATION.md").exists():
        text+="\nНа [исходной validation](OBJECTIVE_VALIDATION.md) weighted прирост MC/exact слабее, а CI пересекают ноль. Гипотеза несовпадения целей не получила убедительного подтверждения на втором периоде. Validation уже использовалась для выбора checkpoint; она также не является новым holdout.\n"
    stopping=ROOT/"artifacts/task2_validation_path"
    if (stopping/"runtime.json").exists():
        path=pd.read_csv(stopping/"metrics.csv");test=path[path.split=="test"].groupby("method").mean(numeric_only=True)
        columns=["ndcg@10","recall@10","context_rm_watch@10","max_item_user_fraction"]
        test[columns].to_csv(stopping/"summary.csv")
        selected=read("artifacts/task2_validation_path/selection.json")["selected"]
        text+=f"\n## Момент остановки по validation\n\n[Исследование траекторий](STOPPING_STUDY.md): двенадцать method/step вариантов, три matched RL seed; выбран {selected['method']} на шаге {selected['step']}. Короткий RL проходит исходные строгие ограничения, в отличие от fixed200. Это дополнительный поиск на той же validation после знакомства с исходным test.\n\n"+markdown(test[columns],index=True)+"\n\nПрирост ordinary NDCG к reference и к matched continued-SFT мал и неубедителен по условным user CI; прирост predicted reward нельзя трактовать как наблюдаемый отклик.\n"
        doc="# Остановка RL по validation и контроль дополнительных SFT шагов\n\n"
        doc+="Фиксированные 200 updates могли быть слишком длинным дообучением. До новых результатов заданы шаги 20/50/100/150/200/400 и два метода: Monte Carlo с четырьмя действиями и exact expectation по 100 кандидатам frozen reference. Для каждого три matched seed 42/43/44. Reference и Context RM фиксированы.\n\n"
        doc+="Learning rate 3e-5, KL=0,5, CE=0,2, batch=16 и исходные 100k fine-tuning контекстов сохранены. Каждому методу и continued-SFT контролю доступны одинаковые minibatches и кеш reward. Контроль имеет нулевой reward gradient и не участвует в выборе RL.\n\n"
        doc+="## Выбор до test\n\nДля 12 method/step вариантов максимизируется средний validation watch-proxy при NDCG и Recall не ниже 99% reference и max-item fraction не выше reference+0,02. Reference допускается как fallback. Test оценивает только reference, выбранный RL и continued-SFT на том же шаге. Никакой момент остановки по test не выбирается.\n\n"
        doc+=f"Выбран **{selected['method']}, {selected['step']} updates**. Это выбор настройки по средним трёх seed, а не выбор самого удачного seed. Числа ниже — средние метрик, не качество ансамбля logits.\n\n"
        doc+="## Validation path\n\n"+markdown(pd.read_csv(stopping/"validation_means.csv")[["method","step"]+columns])+"\n\n"
        doc+="## Test после фиксации настройки\n\n"+markdown(test[columns],index=True)+"\n\n"
        effects=read("artifacts/task2_validation_path/paired_test_ci.json")
        rows=[]
        for name,values in effects.items():
            effect=values["ndcg"];rows.append({"comparison":name,"delta NDCG":effect["mean"],"CI low":effect["ci95"][0],"CI high":effect["ci95"][1]})
        doc+=markdown(pd.DataFrame(rows))+"\n\n"
        doc+="Короткая RL-политика повышает predicted reward, сохраняя ordinary relevance в пределах точности этого эксперимента. Ordinary NDCG почти столь же хорошо меняется у matched continued-SFT. Поэтому точечный прирост точности нельзя целиком приписать reward gradient; CI к reference и контролю пересекают ноль.\n\n"
        doc+="## Проверка воспроизводимости и пределы\n\nНа шаге 200 все шесть RL checkpoints бит-в-бит совпали с прежними fixed200 моделями. Дополнительные validation scores не изменили обучение; prefix training indices и top100 support также совпали по SHA. Это соединяет эксперименты общей траекторией, без незаметной смены оптимизатора.\n\n"
        doc+="Выбор JSON сохранён и захеширован до test. Пользовательские CI условны на фиксированных SFT/RM, дополнительных настройках и уже исследованном KION. Они не учитывают uncertainty выбора по reused validation и не доказывают причинного или онлайн-эффекта. Старые fixed200 fallback результаты сохранены.\n\n"
        doc+="Воспроизведение: `python -m task2.validation_path` после full-data/Context RM/conditional full-reference этапов. Полные protocol, validation curves, per-user CSV, prefix identity и checksum выбора находятся в `artifacts/task2_validation_path/`.\n"
        (ROOT/"task2/STOPPING_STUDY.md").write_text(doc)
    (ROOT/"task2/IMPROVEMENTS.md").write_text(text)


if __name__ == "__main__": main()
