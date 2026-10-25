"""Render actual experiment artifacts to tables and exportable figures.

Never fills in absent results or substitutes published numbers.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shared.data import ROOT


def table(frame,columns=None):
    if columns: frame=frame[columns]
    names=list(frame.columns)
    lines=["| "+" | ".join(map(str,names))+" |","| "+" | ".join("---" for _ in names)+" |"]
    for row in frame.itertuples(index=False,name=None):
        values=[]
        for value in row:
            if isinstance(value,(float,np.floating)): values.append(f"{value:.5f}" if np.isfinite(value) else "—")
            else: values.append(str(value).replace("|","/"))
        lines.append("| "+" | ".join(values)+" |")
    return "\n".join(lines)


def task1(root,output=None):
    base=root/"task1"; prompt=root/"task1_promptfe"
    met=pd.read_csv(base/"metrics.csv")
    test=met[met.split=="test"]
    columns=["ndcg@10","recall@10","map@10"]
    agg=test.groupby("model")[columns].agg(["mean","std","count"])
    rows=[]
    for model,row in agg.iterrows():
        r={"model":model,"seeds":int(row[("ndcg@10","count")])}
        for c in columns: r[c]=row[(c,"mean")]; r[c+" std"]=row[(c,"std")]
        rows.append(r)
    frame=pd.DataFrame(rows)
    text="# Фактические результаты задания 1\n\n"+table(frame)+"\n\n"
    select=json.loads((base/"selection.json").read_text())
    text+=f"Выбранный на validation набор окон: `{select['selected']}`. Порядок блоков: `{select['window_order']}`.\n\n"
    ci=json.loads((base/"paired_test_ci.json").read_text())
    cirows=[{"seed":seed,"delta NDCG":v["ndcg"]["delta"],"CI low":v["ndcg"]["ci95"][0],"CI high":v["ndcg"]["ci95"][1]} for seed,v in ci.items()]
    text+="Парный bootstrap по пользователям одного test-окна, auto минус manual:\n\n"+table(pd.DataFrame(cirows))+"\n\n"
    if (prompt/"metrics.csv").exists():
        p=pd.read_csv(prompt/"metrics.csv")
        text+="## LLM-поиск выражений\n\n"+table(p)+"\n\n"
        selections=json.loads((prompt/"selection.json").read_text())
        text+="Выбранные выражения:\n\n```json\n"+json.dumps(selections["selected_expressions"],ensure_ascii=False,indent=2)+"\n```\n\n"
        proposals=json.loads((prompt/"proposals.json").read_text())
        status=pd.DataFrame(proposals).groupby("method").agg(attempts=("valid","size"),valid_candidates=("valid","sum"))
        text+="Фактический бюджет поиска:\n\n"+table(status.reset_index())+"\n\n"
        if (prompt/"paired_test_ci.json").exists():
            pci=json.loads((prompt/"paired_test_ci.json").read_text())
            cr=[{"method vs manual":name,"delta NDCG":v["ndcg"]["delta"],"CI low":v["ndcg"]["ci95"][0],"CI high":v["ndcg"]["ci95"][1]} for name,v in pci.items()]
            text+="Парные сравнения LLM-поиска на test (один seed):\n\n"+table(pd.DataFrame(cr))+"\n\n"
    text+="Метрики относятся к условному ранжированию с синтетическими негативами. Холодные объекты/пользователи перечислены в protocol.json. CI не оценивает временную устойчивость. Полные настройки — в README и protocol.json.\n"
    (output or ROOT/"task1/RESULTS.md").write_text(text)
    fig,ax=plt.subplots(figsize=(9,4))
    ax.barh(frame.model,frame["ndcg@10"])
    ax.set_xlabel("Test NDCG@10 (fixed sampled candidates)"); ax.set_title("Manual and automatically generated features")
    fig.tight_layout(); fig.savefig(base/"comparison.png",dpi=160); fig.savefig(base/"comparison.pdf"); plt.close(fig)


def task2(root,output=None):
    base=root/"task2"; met=pd.read_csv(base/"metrics.csv")
    cols=["model","selected_on_validation","ndcg@10","recall@10","predicted_watch@10","auditor_watch@10","coverage@10","history_repeat@10","max_item_user_fraction"]
    text="# Фактические результаты задания 2\n\n"+table(met,cols)+"\n\n"
    quality=json.loads((base/"reward_quality.json").read_text()); qrows=[]
    for split,models in quality.items():
        for name,values in models.items(): qrows.append({"split":split,"model":name,**values})
    text+="## Качество reward model на наблюдаемых targets\n\n"+table(pd.DataFrame(qrows))+"\n\n"
    sel=json.loads((base/"selection.json").read_text())
    text+=f"Выбор на validation: `{sel['selected']}`. Критерий: {sel['criterion']}.\n\n"
    if (base/"selected_test_ci.json").exists():
        ci=json.loads((base/"selected_test_ci.json").read_text())
        text+="Парная разница test NDCG выбранного варианта и SFT: "+json.dumps(ci)+".\n\n"
    roll=json.loads((base/"rollout_diagnostics.json").read_text())
    r=[]
    for name,v in roll.items():
        for step in v["steps"]: r.append({"model":name,**step})
    text+="## Пять гипотетических шагов\n\n"+table(pd.DataFrame(r))+"\n\n"
    text+="Предсказанный reward и его рост не доказывают улучшение реального отклика или удержания. Generated context предполагает потребление рекомендованного объекта; реальные переходы не наблюдались. Auditor разделяет ошибки данных с основной RM. Оценка проводится офлайн на одном временном окне.\n"
    (output or ROOT/"task2/RESULTS.md").write_text(text)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.scatter(met["ndcg@10"],met["predicted_watch@10"])
    for _,row in met.iterrows(): ax.annotate(row.model,(row["ndcg@10"],row["predicted_watch@10"]),fontsize=8)
    ax.set_xlabel("Test NDCG@10"); ax.set_ylabel("Predicted watch reward@10"); ax.set_title("Relevance / model-based reward tradeoff")
    fig.tight_layout(); fig.savefig(base/"tradeoff.png",dpi=160); fig.savefig(base/"tradeoff.pdf"); plt.close(fig)


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--task",choices=["task1","task2"],required=True); ap.add_argument("--root",default="artifacts")
    ap.add_argument("--output",default=None,help="Write an independent report, e.g. task2/GPU_RESULTS.md")
    args=ap.parse_args(); {"task1":task1,"task2":task2}[args.task](ROOT/args.root,ROOT/args.output if args.output else None)
