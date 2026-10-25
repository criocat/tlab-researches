import numpy as np
import pandas as pd
from shared.data import VAL_CUTOFF, TEST_CUTOFF


def make_examples(events, max_len=32, max_train=100000, max_eval=2000, seed=42):
    cutoff = pd.Timestamp(VAL_CUTOFF)
    trained = events[events.timestamp < cutoff]
    vocabulary = sorted(trained.item_id.unique().tolist())
    mapping = {int(item): i+2 for i,item in enumerate(vocabulary)}
    # PAD=0, unknown history item=1. Neither is a recommendable action.
    splits = {k: [] for k in ["train","valid","test"]}
    cold = {k:0 for k in splits}
    total = {k:0 for k in splits}
    ends = {"train":cutoff, "valid":pd.Timestamp(TEST_CUTOFF),
            "test":events.timestamp.max()+pd.Timedelta(days=1)}
    for u,g in events.groupby("user_id",sort=True):
        groups = list(g.groupby("timestamp",sort=True))
        history = []
        for j,(t,day) in enumerate(groups):
            split = "train" if t<cutoff else "valid" if t<pd.Timestamp(TEST_CUTOFF) else "test"
            if history:
                x = np.zeros(max_len,dtype=np.int64)
                ctx = history[-max_len:]
                x[:len(ctx)] = ctx
                next_t = groups[j+1][0] if j+1<len(groups) else None
                # Three-day follow-up fits the seven-day validation/test splits.
                # An incomplete observation horizon is censored, not a negative.
                return_observed = t+pd.Timedelta(days=3) < ends[split]
                returns = float(next_t is not None and next_t <= t+pd.Timedelta(days=3))
                for row in day.sort_values("item_id").itertuples():
                    total[split]+=1
                    token = mapping.get(int(row.item_id))
                    if token is None:
                        cold[split]+=1
                        # Keep cold held-out targets as un-recommendable UNK.
                        # They count as misses; never move to a later warm day.
                        token=1
                    # All targets on one day have exactly the same prior-day
                    # context. No arbitrary same-day order leaks into a target.
                    splits[split].append((x.copy(),token,float(row.watched_pct)/100,returns,float(return_observed),int(u),str(t.date())))
            history.extend(mapping.get(int(i),1) for i in day.sort_values("item_id").item_id)
    rng = np.random.default_rng(seed)
    result = {}
    for name, rows in splits.items():
        # At evaluation use each user's earliest target day in the split. When
        # multiple targets share a day, retain one deterministically sampled item.
        if name!="train":
            users = {}
            for row in rows:
                if row[5] not in users:
                    users[row[5]] = []
                if not users[row[5]] or users[row[5]][0][6]==row[6]:
                    users[row[5]].append(row)
            rows = [v[int(rng.integers(len(v)))] for _,v in sorted(users.items())]
        n = max_train if name=="train" else max_eval
        if len(rows)>n:
            rows = [rows[i] for i in np.sort(rng.choice(len(rows),n,replace=False))]
        if not rows:
            raise ValueError(f"No examples in split {name}")
        result[name] = {"x":np.stack([r[0] for r in rows]),
                        "y":np.array([r[1] for r in rows]),
                        "reward":np.array([[r[2],r[3]] for r in rows],dtype=np.float32),
                        "return_mask":np.array([r[4] for r in rows],dtype=np.float32),
                        "user":np.array([r[5] for r in rows]), "date":np.array([r[6] for r in rows])}
    counts = np.bincount([mapping[int(i)] for i in trained.item_id],minlength=len(mapping)+2)
    # Freeze every item's metadata/proxy statistic at the policy training cutoff.
    return result, mapping, counts, {"raw_examples":total,"cold_targets":cold,
                                    "retained_examples":{k:len(v["y"]) for k,v in result.items()},
                                    "retained_cold_targets":{k:int((v["y"]==1).sum()) for k,v in result.items()},
                                    "vocabulary_items":len(mapping), "vocabulary_cutoff":str(cutoff)}
