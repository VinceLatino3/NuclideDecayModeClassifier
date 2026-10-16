from pathlib import Path
import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

try:
    from src.features import load_main_dataset
    from src.split import SEED, random_split, element_split
    from src.mlp import train_mlp, LABELS
    from src.mlp_cv import inner_split, predict
    from src.tier2 import LAYERS, MODELS
except ModuleNotFoundError:
    from features import load_main_dataset
    from split import SEED, random_split, element_split
    from mlp import train_mlp, LABELS
    from mlp_cv import inner_split, predict
    from tier2 import LAYERS, MODELS

MEASURED_ONLY = True    
N_BOOT = 1000
CODE = {lab: i for i, lab in enumerate(LABELS)}


def encode(labels):
    return np.array([CODE[x] for x in labels])


def f1_vector(t, p):
    # per-class F1 from integer-coded labels; NaN for classes absent from the truth
    k = len(LABELS)
    tp = np.bincount(t[t == p], minlength=k)
    ct = np.bincount(t, minlength=k)
    cp = np.bincount(p, minlength=k)
    f1 = 2 * tp / np.maximum(ct + cp, 1)
    f1[ct == 0] = np.nan
    return f1


def bootstrap(t, p, groups, n=N_BOOT):
    # 95% interval for (5-class, 4-class) macro-F1; resamples rows, or whole elements if groups given
    rng = np.random.default_rng(SEED)
    if groups is not None:
        members = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
        keys = np.array(list(members))
    stats = []
    for _ in range(n):
        if groups is None:
            idx = rng.integers(0, len(t), len(t))
        else:
            idx = np.concatenate([members[g] for g in rng.choice(keys, len(keys))])
        f1 = f1_vector(t[idx], p[idx])
        stats.append((np.nanmean(f1), np.nanmean(f1[:4])))
    return np.percentile(stats, [2.5, 97.5], axis=0)   # row 0 = low, row 1 = high


def fit_predict(tr, te, grouped, feats, kind, params):
    if kind == "mlp":
        fit, val = inner_split(tr, grouped)             # early-stopping set, from train only
        with redirect_stdout(io.StringIO()):
            model, _ = train_mlp(fit, val, features=feats, **params)
        scaler = StandardScaler().fit(fit[feats])
        return np.array(LABELS)[predict(model, scaler, te, feats)]
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
    rf.fit(tr[feats], tr["label"])
    return rf.predict(te[feats])


if __name__ == "__main__":
    print("Scoring on the TEST sets: run once per setting and do not tune on these numbers.")
    df = load_main_dataset()
    tag = "measured" if MEASURED_ONLY else "all"
    rows = []

    for split_name, split_fn, grouped in [("RANDOM", random_split, False),
                                          ("ELEMENT", element_split, True)]:
        train, test = split_fn(df)
        if MEASURED_ONLY:
            train, test = train[~train["extrapolated"]], test[~test["extrapolated"]]

        for last in (0, 1, 2, 3):        # 0 = Tier 1 on every row; 1-3 = rows complete through that layer
            tr = train.dropna(subset=LAYERS[last][1])
            te = test.dropna(subset=LAYERS[last][1])
            t = encode(te["label"])
            groups = te["z"].to_numpy() if grouped else None

            print(f"\n=== {split_name} | rows complete through '{LAYERS[last][0]}' | "
                  f"train {len(tr)}, test {len(te)} ===")
            print(te["label"].value_counts().to_dict())

            table = []
            for layer_name, feats in LAYERS[: last + 1]:
                for model_name, (kind, params) in MODELS.items():
                    p = encode(fit_predict(tr, te, grouped, feats, kind, params))
                    f1 = f1_vector(t, p)
                    lo, hi = bootstrap(t, p, groups)
                    table.append({
                        "layer": layer_name, "model": model_name,
                        "macro_f1": np.nanmean(f1),
                        "macro_f1_4class": np.nanmean(f1[:4]),
                        "ci_4class": f"[{lo[1]:.3f}, {hi[1]:.3f}]",
                        **{f"f1_{lab}": v for lab, v in zip(LABELS, f1)},
                    })
            print(pd.DataFrame(table).round(3).to_string(index=False))
            rows += [{"split": split_name, "experiment": LAYERS[last][0],
                      "n_train": len(tr), "n_test": len(te), **r} for r in table]

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(f"results/final_eval_{tag}.csv", index=False)