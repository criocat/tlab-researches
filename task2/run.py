"""GPT-2 next-item -> learned reward -> on-policy group REINFORCE.

python -m task2.run --users 50000 --epochs 8 --rl-steps 200
All tunable choices use validation; full-catalog test follows frozen selection.
"""
import argparse
import copy
import importlib.metadata
import resource
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, brier_score_loss
from shared.data import ROOT, load_sample, save_json
from task2.data import make_examples
from task2.model import GPT2Rec, RewardModel, append_tokens


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_batch(data, idx, device):
    return (torch.as_tensor(data["x"][idx],device=device,dtype=torch.long),
            torch.as_tensor(data["y"][idx],device=device,dtype=torch.long),
            torch.as_tensor(data["reward"][idx],device=device),
            torch.as_tensor(data["return_mask"][idx],device=device))


def action_logits(model, x):
    logits = model(x)
    logits[:,:2] = -1e9
    return logits


@torch.no_grad()
def evaluate(model, data, rewards=None, reference=None, counts=None, batch=64):
    model.eval()
    device=next(model.parameters()).device
    ranks, top, preds, audit_preds, repeats, kl, ent, nll = [],[],[],[],[],[],[],[]
    for start in range(0,len(data["y"]),batch):
        idx=slice(start,start+batch)
        x,y,_,_=tensor_batch(data,idx,device)
        logits=action_logits(model,x)
        logp=logits.log_softmax(-1)
        p=logp.exp()
        # Deterministic exact rank, tie broken by token ID.
        target=logits.gather(1,y[:,None])
        ids=torch.arange(logits.shape[1],device=device)[None,:]
        rank=1+((logits>target)|((logits==target)&(ids<y[:,None]))).sum(1)
        rec=logits.topk(10,dim=1).indices
        ranks.extend(rank.cpu().tolist()); top.append(rec.cpu().numpy())
        repeats.extend(rec[:,:,None].eq(x[:,None,:]).any(-1).float().mean(1).cpu().tolist())
        ent.extend((-(p*logp).sum(1)).cpu().tolist())
        nll.extend((-logp.gather(1,y[:,None]).squeeze(1)).cpu().tolist())
        if rewards:
            a,b=rewards
            preds.append(a(x,rec).cpu().numpy())
            audit_preds.append(b(x,rec).cpu().numpy())
        if reference is not None:
            ref=action_logits(reference,x).log_softmax(-1)
            kl.extend((p*(logp-ref)).sum(1).cpu().tolist())
    ranks=np.asarray(ranks); top=np.concatenate(top)
    vocab=model.config["vocab"]-2
    freq=np.bincount(top.ravel(),minlength=vocab+2)[2:]
    f=freq/freq.sum()
    metrics={"recall@10":float((ranks<=10).mean()),
             "ndcg@10":float(np.where(ranks<=10,1/np.log2(ranks+1),0).mean()),
             "mrr@10":float(np.where(ranks<=10,1/ranks,0).mean()),
             "coverage@10":float((freq>0).mean()),
             "most_frequent_item_share":float(f.max()),"exposure_entropy":float(-(f[f>0]*np.log(f[f>0])).sum()),
             "history_repeat@10":float(np.mean(repeats)),"policy_entropy":float(np.mean(ent))}
    metrics["max_item_user_fraction"]=float(freq.max()/len(data["y"]))
    metrics["top1_max_user_fraction"]=float(np.bincount(top[:,0]).max()/len(data["y"]))
    if len(top)>1:
        rng=np.random.default_rng(42)
        pairs=rng.integers(0,len(top),size=(1000,2))
        pairs=pairs[pairs[:,0]!=pairs[:,1]]
        metrics["between_user_personalization"]=float(np.mean([1-len(set(top[a])&set(top[b]))/len(set(top[a])|set(top[b])) for a,b in pairs]))
    warm=data["y"]>=2
    metrics["cold_target_fraction"]=float((~warm).mean())
    if warm.any():
        metrics["warm_full_catalog_nll"]=float(np.asarray(nll)[warm].mean())
        metrics["warm_recall@10"]=float((ranks[warm]<=10).mean())
        metrics["warm_ndcg@10"]=float(np.where(ranks[warm]<=10,1/np.log2(ranks[warm]+1),0).mean())
    if counts is not None:
        probs=(counts+1)/(counts[2:].sum()+vocab)
        metrics["novelty@10"]=float(-np.log2(probs[top]).mean())
    if kl:
        metrics["kl_to_reference"]=float(np.mean(kl))
    if preds:
        a=np.concatenate(preds); b=np.concatenate(audit_preds)
        metrics.update({"predicted_watch@10":float(a[:,:,0].mean()),
                        "predicted_return3d@10":float(a[:,:,1].mean()),
                        "auditor_watch@10":float(b[:,:,0].mean()),
                        "reward_model_disagreement":float(np.abs(a[:,:,0]-b[:,:,0]).mean())})
    per=pd.DataFrame({"user_id":data["user"],"target":data["y"],"rank":ranks,
                      "context_length":(data["x"]!=0).sum(1),
                      "ndcg":np.where(ranks<=10,1/np.log2(ranks+1),0),"hit":ranks<=10})
    return metrics,per,top


def train_policy(model,data,valid,args,out):
    device=next(model.parameters()).device
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    rng=np.random.default_rng(args.seed)
    best=-1
    logs=[]
    for epoch in range(args.epochs):
        model.train()
        t=time.perf_counter(); losses=[]
        order=rng.permutation(len(data["y"]))
        for start in range(0,len(order),args.batch):
            x,y,_,_=tensor_batch(data,order[start:start+args.batch],device)
            loss=F.cross_entropy(action_logits(model,x),y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step(); losses.append(float(loss.detach()))
        met,_,_=evaluate(model,valid,batch=args.batch)
        row={"epoch":epoch+1,"train_loss":float(np.mean(losses)),"seconds":time.perf_counter()-t,**met}
        logs.append(row); print("SFT",row,flush=True)
        if met["ndcg@10"]>best:
            best=met["ndcg@10"]
            torch.save(model.state_dict(),out/"sft_best.pt")
        pd.DataFrame(logs).to_csv(out/"sft_training.csv",index=False)
    model.load_state_dict(torch.load(out/"sft_best.pt",map_location=device,weights_only=True))
    model.eval()


def reward_eval(model,data,batch=256):
    model.eval(); device=next(model.parameters()).device
    preds=[]
    with torch.no_grad():
        for s in range(0,len(data["y"]),batch):
            x,y,_,_=tensor_batch(data,slice(s,s+batch),device)
            preds.append(model(x,y).cpu().numpy())
    pred=np.concatenate(preds); truth=data["reward"]
    observed=np.isfinite(truth[:,0])
    result={"watch_mae":float(mean_absolute_error(truth[observed,0],pred[observed,0])),
            "watch_rmse":float(np.sqrt(mean_squared_error(truth[observed,0],pred[observed,0]))),
            "watch_r2":float(r2_score(truth[observed,0],pred[observed,0])),"n":len(truth),
            "observed_watch_n":int(observed.sum())}
    mask=data["return_mask"].astype(bool)
    if mask.any():
        yt=truth[mask,1]; yp=pred[mask,1]
        result["return3d_n"]=int(mask.sum())
        result["return3d_brier"]=float(brier_score_loss(yt,yp))
        result["return3d_auc"]=float(roc_auc_score(yt,yp)) if len(np.unique(yt))>1 else None
    return result,pred


def train_reward(model,train,valid,args,out,name,seed):
    device=next(model.parameters()).device
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    rng=np.random.default_rng(seed); best=float("inf"); logs=[]
    for epoch in range(args.reward_epochs):
        model.train(); losses=[]; epoch_start=time.perf_counter()
        order=rng.permutation(len(train["y"]))
        for s in range(0,len(order),args.batch*2):
            x,y,reward,mask=tensor_batch(train,order[s:s+args.batch*2],device)
            pred=model(x,y)
            observed=torch.isfinite(reward[:,0])
            watch_loss=((pred[:,0]-reward[:,0].nan_to_num()).square()*observed).sum()/observed.sum().clamp(min=1)
            loss=watch_loss + .2*((pred[:,1]-reward[:,1]).square()*mask).sum()/mask.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        met,_=reward_eval(model,valid)
        logs.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"seconds":time.perf_counter()-epoch_start,**met})
        print(name,logs[-1],flush=True)
        if met["watch_rmse"]<best:
            best=met["watch_rmse"]
            torch.save(model.state_dict(),out/f"{name}.pt")
    pd.DataFrame(logs).to_csv(out/f"{name}_training.csv",index=False)
    model.load_state_dict(torch.load(out/f"{name}.pt",map_location=device,weights_only=True))
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)


def rl_finetune(policy,reference,reward,train,args,config,out):
    """On-policy REINFORCE with a leave-one-out group baseline.

    Each context produces G independent trajectories. Baseline for trajectory g
    excludes its own reward. Every update draws fresh trajectories; no off-policy
    importance-ratio claims. Exact KL to frozen SFT policy over the whole catalog.
    """
    device=next(policy.parameters()).device
    opt=torch.optim.AdamW(policy.parameters(),lr=args.rl_lr,weight_decay=.01)
    rng=np.random.default_rng(args.seed+100)
    logs=[]; group=args.group; stage_start=time.perf_counter()
    policy.eval()  # Dropout disabled for identical sampling and gradient policies.
    for step in range(args.rl_steps):
        idx=rng.integers(0,len(train["y"]),size=args.rl_batch)
        original,y,_,_=tensor_batch(train,idx,device)
        history=original.repeat_interleave(group,0)
        logps=[]; rewards=[]; kls=[]
        for horizon in range(config["horizon"]):
            logits=action_logits(policy,history)
            logp=logits.log_softmax(-1); probs=logp.exp()
            with torch.no_grad():
                ref=action_logits(reference,history).log_softmax(-1)
                action=torch.multinomial(probs.detach(),1).squeeze(1)
                predicted=reward(history,action)
                r=predicted[:,0]
                if config["reward"]=="balanced":
                    # Penalize known/earlier-generated repetitions; completion
                    # alone has no reason to prefer fresh or diverse content.
                    seen=history.eq(action[:,None]).any(1).float()
                    r=r-.2*seen
                elif config["reward"]=="return":
                    r=.5*r+.5*predicted[:,1]
            logps.append(logp.gather(1,action[:,None]).squeeze(1))
            kls.append((probs*(logp-ref)).sum(1))
            rewards.append(r)
            history=append_tokens(history,action)
        returns=[]; ret=torch.zeros_like(rewards[0])
        for r in reversed(rewards):
            ret=r+.95*ret; returns.insert(0,ret)
        lp=torch.stack(logps,dim=1).reshape(args.rl_batch,group,-1)
        ret=torch.stack(returns,dim=1).reshape(args.rl_batch,group,-1)
        loo=(ret.sum(1,keepdim=True)-ret)/(group-1)
        advantage=ret-loo
        discount=torch.pow(torch.tensor(.95,device=device),torch.arange(config["horizon"],device=device))
        pg=-(lp*advantage.detach()*discount).mean()
        kl=torch.stack(kls).mean()
        ce=F.cross_entropy(action_logits(policy,original),y)
        loss=config.get("pg_weight",1.)*pg+config["beta"]*kl+config["ce"]*ce
        opt.zero_grad(); loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),1)
        opt.step()
        if step%20==0 or step+1==args.rl_steps:
            row={"step":step+1,"objective":float(loss.detach()),"reward":float(torch.stack(rewards).mean()),
                 "kl":float(kl.detach()),"ce":float(ce.detach()),"grad_norm":float(grad),
                 "seconds":time.perf_counter()-stage_start}
            logs.append(row); print(config["name"],row,flush=True)
            pd.DataFrame(logs).to_csv(out/f'{config["name"]}_training.csv',index=False)
    torch.save(policy.state_dict(),out/f'{config["name"]}.pt')
    return policy


@torch.no_grad()
def rollout_diagnostics(model,data,rewards,reference,args,mapping):
    device=next(model.parameters()).device
    n=min(256,len(data["y"])); x=torch.as_tensor(data["x"][:n],device=device,dtype=torch.long)
    original=x.clone(); generated=[]; rows=[]
    inverse={v:int(k) for k,v in mapping.items()}
    for step in range(5):
        logits=action_logits(model,x); action=logits.argmax(1)
        free=rewards[0](x,action)[:,0]
        fixed=rewards[0](original,action)[:,0]
        audit=rewards[1](x,action)[:,0]
        lp=logits.log_softmax(-1)
        ref=action_logits(reference,x).log_softmax(-1)
        p=lp.exp()
        repeat=torch.stack(generated,dim=1).eq(action[:,None]).any(1).float().mean() if generated else torch.tensor(0.)
        rows.append({"step":step+1,"predicted_reward_free_context":float(free.mean()),
                     "predicted_reward_frozen_context":float(fixed.mean()),
                     "auditor_reward":float(audit.mean()),"within_rollout_repeat":float(repeat),
                     "history_repeat":float(original.eq(action[:,None]).any(1).float().mean()),
                     "kl_on_generated_context":float((p*(lp-ref)).sum(1).mean()),
                     "distinct_items":int(action.unique().numel())})
        generated.append(action); x=append_tokens(x,action)
    trace=torch.stack(generated,1).cpu().numpy()
    examples=[{"user_id":int(data["user"][i]),"items":[inverse[int(z)] for z in trace[i]]} for i in range(min(n,10))]
    return {"steps":rows,"examples":examples,"interpretation":"Generated items are hypothetical, not observed consumption. Frozen-context scores are a diagnostic, not a counterfactual estimator."}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--users",type=int,default=50000)
    ap.add_argument("--max-train",type=int,default=100000)
    ap.add_argument("--max-eval",type=int,default=2000)
    ap.add_argument("--max-len",type=int,default=32)
    ap.add_argument("--dim",type=int,default=64)
    ap.add_argument("--epochs",type=int,default=8)
    ap.add_argument("--reward-epochs",type=int,default=5)
    ap.add_argument("--batch",type=int,default=128)
    ap.add_argument("--lr",type=float,default=.001)
    ap.add_argument("--rl-lr",type=float,default=.0001)
    ap.add_argument("--rl-steps",type=int,default=200)
    ap.add_argument("--rl-batch",type=int,default=16)
    ap.add_argument("--group",type=int,default=4)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",default="artifacts/task2")
    args=ap.parse_args()
    assert args.group>=2
    set_seed(args.seed); torch.set_num_threads(6)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    events,items,users=load_sample(args.users)
    data,mapping,counts,audit=make_examples(events,args.max_len,args.max_train,args.max_eval,args.seed)
    save_json(out/"protocol.json",{"args":vars(args),"device":str(device),"audit":audit,
        "versions":{n:importlib.metadata.version(n) for n in ["torch","numpy","pandas","scikit-learn"]},
        "reward_proxy":"observed watch fraction + censored 3-day next-active-day indicator"})
    save_json(out/"item_mapping.json",mapping)
    for name,d in data.items():
        np.savez_compressed(out/f"examples_{name}.npz",**d)
    print("Data",audit,"device",device,flush=True)
    base=GPT2Rec(len(mapping)+2,max_len=args.max_len,dim=args.dim).to(device)
    save_json(out/"model_config.json",base.config)
    timings={"data_seconds":time.perf_counter()-start}
    stage_start=time.perf_counter()
    train_policy(base,data["train"],data["valid"],args,out)
    timings["sft_seconds"]=time.perf_counter()-stage_start
    for p in base.parameters(): p.requires_grad_(False)
    rms=[]; stage_start=time.perf_counter()
    for i,name in enumerate(["reward","auditor"]):
        set_seed(args.seed+i+10)
        rm=RewardModel(len(mapping)+2).to(device)
        train_reward(rm,data["train"],data["valid"],args,out,name,args.seed+i+10)
        rms.append(rm)
    timings["reward_seconds"]=time.perf_counter()-stage_start
    reward_metrics={}
    observed_watch=np.isfinite(data["train"]["reward"][:,0])
    watch_targets=data["train"]["y"][observed_watch]
    watch_values=data["train"]["reward"][observed_watch,0]
    watch_count=np.bincount(watch_targets,minlength=len(mapping)+2)
    watch_sum=np.bincount(watch_targets,weights=watch_values,minlength=len(mapping)+2)
    global_watch=float(watch_values.mean())
    item_watch=(watch_sum+20*global_watch)/(watch_count+20)
    # RM quality on logged targets is evaluated independently of policy generation.
    for split in ["valid","test"]:
        reward_metrics[split]={name:reward_eval(rm,data[split])[0] for name,rm in zip(["reward","auditor"],rms)}
        mean=float(np.nanmean(data["train"]["reward"][:,0]))
        reward_metrics[split]["global_mean_baseline"]={"watch_mae":float(np.nanmean(np.abs(data[split]["reward"][:,0]-mean))),
            "watch_rmse":float(np.sqrt(np.nanmean((data[split]["reward"][:,0]-mean)**2)))}
        item_pred=item_watch[data[split]["y"]]
        item_true=data[split]["reward"][:,0]
        reward_metrics[split]["shrunk_item_mean_baseline"]={"watch_mae":float(np.nanmean(np.abs(item_true-item_pred))),
            "watch_rmse":float(np.sqrt(np.nanmean((item_true-item_pred)**2))),"prior_weight":20}
    save_json(out/"reward_quality.json",reward_metrics)
    configs=[{"name":"rl_watch_unconstrained","reward":"watch","beta":0.,"ce":0.,"horizon":1},
             {"name":"rl_watch_kl01","reward":"watch","beta":.1,"ce":.2,"horizon":1},
             {"name":"rl_watch_kl1","reward":"watch","beta":1.,"ce":.2,"horizon":1},
             {"name":"rl_balanced_5step","reward":"balanced","beta":.1,"ce":.2,"horizon":5},
             {"name":"rl_return_5step","reward":"return","beta":.1,"ce":.2,"horizon":5}]
    valid_metrics={"sft":evaluate(base,data["valid"],rms,base,counts,args.batch)[0]}
    timings["rl_seconds"]={}
    for config in configs:
        set_seed(args.seed+100)
        policy=copy.deepcopy(base)
        for p in policy.parameters(): p.requires_grad_(True)
        stage_start=time.perf_counter()
        rl_finetune(policy,base,rms[0],data["train"],args,config,out)
        timings["rl_seconds"][config["name"]]=time.perf_counter()-stage_start
        valid_metrics[config["name"]]=evaluate(policy,data["valid"],rms,base,counts,args.batch)[0]
        del policy
    # Predeclared selection: maximize predicted watch while retaining >=95% of
    # baseline validation NDCG; SFT is eligible as a fallback.
    feasible=[n for n,m in valid_metrics.items() if m["ndcg@10"]>=.95*valid_metrics["sft"]["ndcg@10"]]
    selected=max(feasible,key=lambda n:valid_metrics[n]["predicted_watch@10"])
    save_json(out/"selection.json",{"selected":selected,"criterion":"max validation predicted_watch@10 subject to ndcg@10 >= 0.95 * SFT", "valid_metrics":valid_metrics,"configs":configs})
    rows=[]; rollouts={}; baseline_per=None; baseline_top=None; distributions={}; strata=[]
    meta=items.set_index("item_id")
    inverse={v:int(k) for k,v in mapping.items()}
    genres=np.array([str(meta.loc[inverse[token],"genre"]) if token in inverse else "unknown" for token in range(len(mapping)+2)])
    types=np.array([str(meta.loc[inverse[token],"content_type"]) if token in inverse else "unknown" for token in range(len(mapping)+2)])
    for name in ["sft"]+[c["name"] for c in configs]:
        if name=="sft": policy=base
        else:
            policy=copy.deepcopy(base)
            policy.load_state_dict(torch.load(out/f"{name}.pt",map_location=device,weights_only=True))
        met,per,top=evaluate(policy,data["test"],rms,base,counts,args.batch)
        if name=="sft": baseline_per=per.copy(); baseline_top=top.copy()
        overlap=np.array([len(set(a)&set(b))/10 for a,b in zip(top,baseline_top)])
        met["overlap_with_sft@10"]=float(overlap.mean())
        met["first_genre_diversity@10"]=float(np.mean([len(set(genres[r]))/10 for r in top]))
        met["film_share@10"]=float((types[top]=="film").mean())
        distributions[name]={"first_genre":{str(k):float(v) for k,v in pd.Series(genres[top].ravel()).value_counts(normalize=True).items()},
                             "content_type":{str(k):float(v) for k,v in pd.Series(types[top].ravel()).value_counts(normalize=True).items()}}
        rows.append({"model":name,"selected_on_validation":name==selected,**met})
        per.to_csv(out/f"per_user_{name}.csv",index=False)
        groups=pd.cut(per.context_length,bins=[0,1,5,15,100000],labels=["1","2-5","6-15","16+"])
        for group,g in per.groupby(groups,observed=True):
            strata.append({"model":name,"context_length":str(group),"users":len(g),
                           "ndcg@10":float(g.ndcg.mean()),"recall@10":float(g.hit.mean())})
        np.save(out/f"top10_{name}.npy",top)
        rollouts[name]=rollout_diagnostics(policy,data["test"],rms,base,args,mapping)
        if name==selected and name!="sft":
            delta=(per.ndcg-baseline_per.ndcg).to_numpy()
            rng=np.random.default_rng(args.seed)
            ci=np.quantile([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)],[.025,.975])
            save_json(out/"selected_test_ci.json",{"ndcg_delta":float(delta.mean()),"ci95":ci.tolist()})
        print("TEST",rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(strata).to_csv(out/"history_length_metrics.csv",index=False)
    save_json(out/"rollout_diagnostics.json",rollouts)
    save_json(out/"recommendation_metadata.json",distributions)
    timings.update({"total_seconds":time.perf_counter()-start,"max_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else None})
    save_json(out/"runtime.json",timings)


if __name__=="__main__": main()
