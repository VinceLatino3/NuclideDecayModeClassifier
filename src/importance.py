import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

try:
    from src.features import load_main_dataset, TIER1
    from src.split import SEED, random_split, element_split
    from src.mlp import train_mlp, LABELS
    from src.mlp_cv import inner_split, predict
    from src.tier2 import LAYERS, MODELS
    from src.compare import fold_splits
    from src.final_eval import encode, f1_vector
except ModuleNotFoundError:
    from features import load_main_dataset, TIER1
    from split import SEED, random_split, element_split
    from mlp import train_mlp, LABELS
    from mlp_cv import inner_split, predict
    from tier2 import LAYERS, MODELS
    from compare import fold_splits
    from final_eval import encode, f1_vector

N_REPEATS = 10
LAYER_IDX = [2, 3]        # +sep and +Q (rows complete through that layer)
SPLITS = [("RANDOM", random_split, False), ("ELEMENT", element_split, True)]
GROUPS = [
    ("Tier 1 structure (Z, N and derived)", TIER1),
    ("mass surface (binding, mass excess)", ["binding", "massexcess"]),
    ("separation energies (sn, sp)", ["sn", "sp"]),
    ("Q-values (qa, qec, qbm, qbm_n)", ["qa", "qec", "qbm", "qbm_n"]),
]


def train_model(ftrain, grouped, feats, kind, params):
    # returns a function mapping a feature DataFrame to integer-coded predictions
    if kind == "mlp":
        fit, val = inner_split(ftrain, grouped)
        with redirect_stdout(io.StringIO()):
            model, _ = train_mlp(fit, val, features=feats, **params)
        scaler = StandardScaler().fit(fit[feats])
        return lambda D: encode(np.array(LABELS)[predict(model, scaler, D, feats)])
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
    rf.fit(ftrain[feats], ftrain["label"])
    return lambda D: encode(rf.predict(D[feats]))


if __name__ == "__main__":
    df = load_main_dataset()
    rng = np.random.default_rng(SEED)
    all_rows = []

    for layer_i in LAYER_IDX:
        layer_name, feats = LAYERS[layer_i]
        groups = [(g, cols) for g, cols in GROUPS if all(c in feats for c in cols)]
        for split_name, split_fn, grouped in SPLITS:
            train, _ = split_fn(df)             # test set deliberately ignored
            sub = train.dropna(subset=feats)
            rows = []
            for model_name, (kind, params) in MODELS.items():
                drops = {g: [] for g, _ in groups}
                for tr_idx, ev_idx in fold_splits(sub, grouped, SEED):
                    ftrain, feval = sub.iloc[tr_idx], sub.iloc[ev_idx]
                    predict_fn = train_model(ftrain, grouped, feats, kind, params)
                    X = feval[feats].copy()
                    t = encode(feval["label"])
                    base = f1_vector(t, predict_fn(X))
                    for g, cols in groups:
                        reps = []
                        for _ in range(N_REPEATS):
                            Xp = X.copy()
                            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]  # shuffle group jointly
                            reps.append(base - f1_vector(t, predict_fn(Xp)))
                        drops[g].append(np.mean(reps, axis=0))
                for g, _ in groups:
                    d = np.array(drops[g])                  # folds x classes
                    four = np.nanmean(d[:, :4], axis=1)
                    rows.append({"layer": layer_name, "split": split_name, "model": model_name,
                                 "group": g, "drop_4class": four.mean(), "sd": four.std(),
                                 **{f"drop_{lab}": v for lab, v in zip(LABELS, np.nanmean(d, axis=0))}})

            print(f"\n=== rows complete through '{layer_name}' | {split_name} split | {len(sub)} rows ===")
            out = pd.DataFrame(rows).sort_values(["model", "drop_4class"], ascending=[True, False])
            print(out.drop(columns=["layer", "split"]).round(3).to_string(index=False))
            all_rows += rows

    pd.DataFrame(all_rows).to_csv("results/permutation_importance.csv", index=False)