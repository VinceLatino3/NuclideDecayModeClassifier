import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from src.features import load_main_dataset, TIER1
    from src.split import SEED, random_split, element_split
    from src.mlp import train_mlp, LABEL_TO_ID
    from src.mlp_cv import inner_split, predict
except ModuleNotFoundError:
    from features import load_main_dataset, TIER1
    from split import SEED, random_split, element_split
    from mlp import train_mlp, LABEL_TO_ID
    from mlp_cv import inner_split, predict

FOLD_SEEDS = [SEED, SEED + 1, SEED + 2]

D = dict(hidden=64, layers=2, dropout=0.1, weight_decay=1e-4, lr=1e-3, batch_size=64)
MLPS = {
    "MLP default, power 0": dict(D, weight_power=0.0),
    "MLP default, power 1": dict(D, weight_power=1.0),
    "MLP top1": dict(hidden=128, layers=1, dropout=0.1, weight_decay=1e-3,
                     lr=3e-3, batch_size=128, weight_power=0.75),
    "MLP top2": dict(hidden=128, layers=1, dropout=0.1, weight_decay=1e-5,
                     lr=1e-3, batch_size=64, weight_power=0.75),
    "MLP top3": dict(hidden=256, layers=1, dropout=0.1, weight_decay=1e-3,
                     lr=1e-3, batch_size=128, weight_power=0.5),
}
RF_DEFAULT = dict(n_estimators=300, max_depth=None, min_samples_leaf=1,
                  max_features="sqrt", class_weight=None)
RF_TUNED = dict(n_estimators=300, max_depth=20, min_samples_leaf=3,
                max_features="sqrt", class_weight="balanced_subsample")


def fold_splits(train, grouped, seed):
    if grouped:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        return list(cv.split(train, train["label"], groups=train["z"]))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(cv.split(train, train["label"]))


def mlp_scores(train, splits, grouped, params):
    out = []
    for tr_idx, ev_idx in splits:
        fold_train, fold_eval = train.iloc[tr_idx], train.iloc[ev_idx]
        tr, va = inner_split(fold_train, grouped)
        with redirect_stdout(io.StringIO()):
            model, _ = train_mlp(tr, va, **params)
        scaler = StandardScaler().fit(tr[TIER1])
        pred = predict(model, scaler, fold_eval)
        true = fold_eval["label"].map(LABEL_TO_ID).to_numpy()
        out.append(f1_score(true, pred, average="macro"))
    return out


def rf_scores(train, splits, params):
    X, y = train[TIER1], train["label"]
    out = []
    for tr_idx, ev_idx in splits:
        rf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
        rf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        out.append(f1_score(y.iloc[ev_idx], rf.predict(X.iloc[ev_idx]), average="macro"))
    return out


if __name__ == "__main__":
    df = load_main_dataset()
    models = {name: ("mlp", p) for name, p in MLPS.items()}
    models["RF default"] = ("rf", RF_DEFAULT)
    if RF_TUNED is not None:
        models["RF tuned"] = ("rf", RF_TUNED)
    reference = "RF tuned" if RF_TUNED is not None else "RF default"

    for split_name, split_fn, grouped in [("RANDOM", random_split, False),
                                          ("ELEMENT", element_split, True)]:
        train, _ = split_fn(df)          # test set deliberately ignored
        scores = {name: [] for name in models}
        for seed in FOLD_SEEDS:
            splits = fold_splits(train, grouped, seed)
            for name, (kind, params) in models.items():
                s = mlp_scores(train, splits, grouped, params) if kind == "mlp" \
                    else rf_scores(train, splits, params)
                scores[name] += s

        print(f"\n=== {split_name} split: macro-F1 over {len(FOLD_SEEDS) * 5} folds ===")
        ref = np.array(scores[reference])
        rows = []
        for name, s in scores.items():
            s = np.array(s)
            rows.append({"model": name, "mean": s.mean().round(3), "std": s.std().round(3),
                         f"diff vs {reference}": (s - ref).mean().round(3),
                         "folds won": f"{int((s > ref).sum())}/{len(s)}"})
        print(pd.DataFrame(rows).to_string(index=False))