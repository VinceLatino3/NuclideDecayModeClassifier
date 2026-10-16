import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    from src.features import load_main_dataset, TIER1
    from src.split import SEED, element_split
    from src.compare import fold_splits, RF_TUNED
except ModuleNotFoundError:
    from features import load_main_dataset, TIER1
    from split import SEED, element_split
    from compare import fold_splits, RF_TUNED

SEP_SET = TIER1 + ["binding", "massexcess", "sn", "sp"]

if __name__ == "__main__":
    df = load_main_dataset()
    train, _ = element_split(df)       
    sub = train.dropna(subset=SEP_SET) 

    caught = {}
    for name, feats in {"T1": TIER1, "sep": SEP_SET}.items():
        hit = pd.Series(False, index=sub.index)
        for tr_idx, ev_idx in fold_splits(sub, True, SEED):
            ft, fe = sub.iloc[tr_idx], sub.iloc[ev_idx]
            rf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **RF_TUNED)
            rf.fit(ft[feats], ft["label"])
            hit.loc[fe.index] = rf.predict(fe[feats]) == "other"
        caught[name] = hit

    other = sub["label"] == "other"
    out = pd.DataFrame({"mode": sub.loc[other, "decay_1"],
                        "T1": caught["T1"][other], "sep": caught["sep"][other]})
    print("Inside 'other': caught with Tier 1 vs. with +sep, by original decay mode")
    print(out.groupby("mode").agg(total=("T1", "size"), T1=("T1", "sum"), sep=("sep", "sum")))