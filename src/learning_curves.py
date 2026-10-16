from pathlib import Path
import io
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from src.features import load_main_dataset
    from src.split import SEED, random_split, element_split
    from src.mlp import train_mlp, LABELS
    from src.mlp_cv import inner_split, predict
    from src.tier2 import LAYERS, MODELS
    from src.compare import fold_splits
    from src.final_eval import encode, f1_vector
except ModuleNotFoundError:
    from features import load_main_dataset
    from split import SEED, random_split, element_split
    from mlp import train_mlp, LABELS
    from mlp_cv import inner_split, predict
    from tier2 import LAYERS, MODELS
    from compare import fold_splits
    from final_eval import encode, f1_vector

FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
LAYER_IDX = [0, 2]        # Tier 1 and +sep (rows complete through that layer)
SPLITS = [("RANDOM", random_split, False), ("ELEMENT", element_split, True)]


def fit_predict(ftrain, feval, grouped, feats, kind, params):
    if kind == "mlp":
        fit, val = inner_split(ftrain, grouped)
        with redirect_stdout(io.StringIO()):
            model, _ = train_mlp(fit, val, features=feats, **params)
        scaler = StandardScaler().fit(fit[feats])
        return encode(np.array(LABELS)[predict(model, scaler, feval, feats)])
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
    rf.fit(ftrain[feats], ftrain["label"])
    return encode(rf.predict(feval[feats]))


if __name__ == "__main__":
    df = load_main_dataset()
    rows = []

    for layer_i in LAYER_IDX:
        layer_name, feats = LAYERS[layer_i]
        for split_name, split_fn, grouped in SPLITS:
            train, _ = split_fn(df)             # test set deliberately ignored
            sub = train.dropna(subset=feats)
            folds = fold_splits(sub, grouped, SEED)
            for model_name, (kind, params) in MODELS.items():
                for frac in FRACTIONS:
                    scores, sizes = [], []
                    for tr_idx, ev_idx in folds:
                        ftrain, feval = sub.iloc[tr_idx], sub.iloc[ev_idx]
                        if frac < 1.0:
                            ftrain, _ = train_test_split(ftrain, train_size=frac,
                                                         stratify=ftrain["label"], random_state=SEED)
                        p = fit_predict(ftrain, feval, grouped, feats, kind, params)
                        f1 = f1_vector(encode(feval["label"]), p)
                        scores.append(np.nanmean(f1[:4]))      # four-class macro-F1
                        sizes.append(len(ftrain))
                    rows.append({"layer": layer_name, "split": split_name, "model": model_name,
                                 "fraction": frac, "n_train": float(np.mean(sizes)),
                                 "f1_4class": float(np.mean(scores)), "std": float(np.std(scores))})
                    print(f"{layer_name:5s} {split_name:7s} {model_name:3s} frac {frac:.1f} "
                          f"n_train {int(np.mean(sizes)):5d}  4-class F1 {np.mean(scores):.3f} "
                          f"(+/- {np.std(scores):.3f})")

    res = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    res.to_csv("results/learning_curves.csv", index=False)

    fig, axes = plt.subplots(len(LAYER_IDX), 2, figsize=(11, 4.2 * len(LAYER_IDX)), sharey=True)
    for r, layer_i in enumerate(LAYER_IDX):
        for c, (split_name, _, _) in enumerate(SPLITS):
            ax = axes[r][c]
            g = res[(res["layer"] == LAYERS[layer_i][0]) & (res["split"] == split_name)]
            for model_name, gm in g.groupby("model"):
                ax.errorbar(gm["n_train"], gm["f1_4class"], yerr=gm["std"],
                            marker="o", capsize=3, label=model_name)
            ax.set_title(f"{LAYERS[layer_i][0]} features, {split_name.lower()} split")
            ax.set_xlabel("training rows")
            ax.set_ylabel("four-class macro-F1 (CV)")
            ax.grid(alpha=0.3)
            ax.legend()
    fig.tight_layout()
    fig.savefig("figures/learning_curves.png", dpi=150)
    plt.show()