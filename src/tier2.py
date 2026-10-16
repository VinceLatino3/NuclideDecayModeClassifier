import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

try:
    from src.features import load_main_dataset, TIER1
    from src.split import SEED, random_split, element_split
    from src.mlp import train_mlp, LABELS
    from src.mlp_cv import inner_split, predict
    from src.compare import fold_splits, MLPS, RF_TUNED
except ModuleNotFoundError:
    from features import load_main_dataset, TIER1
    from split import SEED, random_split, element_split
    from mlp import train_mlp, LABELS
    from mlp_cv import inner_split, predict
    from compare import fold_splits, MLPS, RF_TUNED

FOLD_SEEDS = [SEED, SEED + 1, SEED + 2]
MEASURED_ONLY = False

COLS_2A = ["binding", "massexcess"]
COLS_2B = ["sn", "sp"]
COLS_2C = ["qa", "qec", "qbm", "qbm_n"]
LAYERS = [
    ("T1", TIER1),
    ("+mass", TIER1 + COLS_2A),
    ("+sep", TIER1 + COLS_2A + COLS_2B),
    ("+Q", TIER1 + COLS_2A + COLS_2B + COLS_2C),
]
MODELS = {"MLP": ("mlp", MLPS["MLP default, power 0"]), "RF": ("rf", RF_TUNED)}


def run_model(train, grouped, features, kind, params):
    f1s, trues, preds = [], [], []
    for seed in FOLD_SEEDS:
        for tr_idx, ev_idx in fold_splits(train, grouped, seed):
            fold_train, fold_eval = train.iloc[tr_idx], train.iloc[ev_idx]
            if kind == "mlp":
                tr, va = inner_split(fold_train, grouped)
                with redirect_stdout(io.StringIO()):
                    model, _ = train_mlp(tr, va, features=features, **params)
                scaler = StandardScaler().fit(tr[features])
                pred = np.array(LABELS)[predict(model, scaler, fold_eval, features)]
            else:
                rf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
                rf.fit(fold_train[features], fold_train["label"])
                pred = rf.predict(fold_eval[features])
            true = fold_eval["label"].to_numpy()
            f1s.append(f1_score(true, pred, average="macro"))
            trues.append(true)
            preds.append(pred)
    return f1s, np.concatenate(trues), np.concatenate(preds)


if __name__ == "__main__":
    df = load_main_dataset()
    tag = "measured" if MEASURED_ONLY else "all"
    rows = []

    for last in (1, 2, 3):                 
        needed = LAYERS[last][1]
        for split_name, split_fn, grouped in [("RANDOM", random_split, False),
                                              ("ELEMENT", element_split, True)]:
            train, _ = split_fn(df)           # test set deliberately ignored
            if MEASURED_ONLY:
                train = train[~train["extrapolated"]]
            sub = train.dropna(subset=needed)

            print(f"\n=== rows complete through '{LAYERS[last][0]}' | {split_name} split | "
                  f"{len(sub)} of {len(train)} rows ===")
            print(sub["label"].value_counts().to_dict())

            table = []
            for layer_name, feats in LAYERS[: last + 1]:
                for model_name, (kind, params) in MODELS.items():
                    f1s, true, pred = run_model(sub, grouped, feats, kind, params)
                    per_class = f1_score(true, pred, labels=LABELS, average=None, zero_division=0)
                    table.append({
                        "layer": layer_name, "model": model_name,
                        "macro_f1": np.mean(f1s), "std": np.std(f1s),
                        "macro_f1_4class": float(np.mean(per_class[:4])),
                        **{f"f1_{lab}": v for lab, v in zip(LABELS, per_class)},
                    })
            print(pd.DataFrame(table).round(3).to_string(index=False))
            rows += [{"experiment": LAYERS[last][0], "split": split_name,
                      "n_rows": len(sub), **r} for r in table]

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(f"results/tier2_{tag}.csv", index=False)