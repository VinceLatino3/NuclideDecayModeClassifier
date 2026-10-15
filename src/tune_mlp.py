import io
import random
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

try:
    from src.features import load_main_dataset, TIER1
    from src.split import SEED, element_split
    from src.mlp import train_mlp, LABEL_TO_ID, DEVICE
    from src.mlp_cv import inner_split, predict
except ModuleNotFoundError:
    from features import load_main_dataset, TIER1
    from split import SEED, element_split
    from mlp import train_mlp, LABEL_TO_ID, DEVICE
    from mlp_cv import inner_split, predict

N_CONFIGS = 30
OUT = Path("results/mlp_tuning.csv")

SPACE = {
    "hidden": [32, 64, 128, 256],
    "layers": [1, 2, 3],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "weight_decay": [1e-5, 1e-4, 1e-3, 1e-2],
    "lr": [3e-4, 1e-3, 3e-3],
    "batch_size": [32, 64, 128],
    "weight_power": [0.0, 0.25, 0.5, 0.75, 1.0],
}
DEFAULT = dict(hidden=64, layers=2, dropout=0.1, weight_decay=1e-4, lr=1e-3, batch_size=64)


def sample_configs(n, seed=SEED):
    rng = random.Random(seed)
    configs = []
    while len(configs) < n:
        cfg = {k: rng.choice(v) for k, v in SPACE.items()}
        if cfg not in configs:
            configs.append(cfg)
    return configs


def cv_score(train, params):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs, f1s = [], []
    for tr_idx, ev_idx in cv.split(train, train["label"], groups=train["z"]):
        fold_train, fold_eval = train.iloc[tr_idx], train.iloc[ev_idx]
        tr, va = inner_split(fold_train, grouped=True)      # early-stopping set

        with redirect_stdout(io.StringIO()):                # silence per-epoch printing
            model, _ = train_mlp(tr, va, **params)

        scaler = StandardScaler().fit(tr[TIER1])
        pred = predict(model, scaler, fold_eval)
        true = fold_eval["label"].map(LABEL_TO_ID).to_numpy()
        accs.append(accuracy_score(true, pred))
        f1s.append(f1_score(true, pred, average="macro"))
    return np.mean(f1s), np.std(f1s), np.mean(accs)


if __name__ == "__main__":
    print("device:", DEVICE)
    df = load_main_dataset()
    train, _ = element_split(df)            # test set deliberately ignored
    OUT.parent.mkdir(exist_ok=True)

    # first two rows re-run the untuned settings as a check on this harness
    configs = [dict(DEFAULT, weight_power=0.0), dict(DEFAULT, weight_power=1.0)]
    configs += sample_configs(N_CONFIGS)

    rows = []
    for i, cfg in enumerate(configs, 1):
        t0 = time.time()
        f1_mean, f1_std, acc = cv_score(train, cfg)
        rows.append({**cfg, "macro_f1": f1_mean, "f1_std": f1_std, "accuracy": acc})
        pd.DataFrame(rows).to_csv(OUT, index=False)         # saved after every config
        print(f"[{i}/{len(configs)}] macro-F1 {f1_mean:.3f} (+/- {f1_std:.3f})  "
              f"acc {acc:.3f}  {time.time() - t0:.0f}s  {cfg}")

    res = pd.DataFrame(rows)
    print("\nTop 5 configurations:")
    print(res.sort_values("macro_f1", ascending=False).head(5).round(3).to_string(index=False))

    print("\nMean macro-F1 by hyperparameter value (random configs only; noisy):")
    random_rows = pd.DataFrame(rows[2:])
    for k in SPACE:
        print(f"\n{k}")
        print(random_rows.groupby(k)["macro_f1"].agg(["mean", "count"]).round(3).to_string())