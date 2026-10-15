import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

try:
    from src.features import load_main_dataset, TIER1
    from src.split import SEED, element_split
except ModuleNotFoundError:
    from features import load_main_dataset, TIER1
    from split import SEED, element_split

N_CONFIGS = 30
OUT = Path("results/rf_tuning.csv")

SPACE = {
    "n_estimators": [300, 600],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": ["sqrt", 0.5, None],
    "class_weight": [None, "balanced", "balanced_subsample"],
}
DEFAULT = dict(n_estimators=300, max_depth=None, min_samples_leaf=1,
               max_features="sqrt", class_weight=None)


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
    X, y = train[TIER1], train["label"]
    accs, f1s = [], []
    for tr_idx, ev_idx in cv.split(X, y, groups=train["z"]):
        rf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
        rf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = rf.predict(X.iloc[ev_idx])
        accs.append(accuracy_score(y.iloc[ev_idx], pred))
        f1s.append(f1_score(y.iloc[ev_idx], pred, average="macro"))
    return np.mean(f1s), np.std(f1s), np.mean(accs)


if __name__ == "__main__":
    df = load_main_dataset()
    train, _ = element_split(df)            # test set deliberately ignored
    OUT.parent.mkdir(exist_ok=True)

    configs = [DEFAULT] + sample_configs(N_CONFIGS)   # first row should reproduce ~0.717
    rows = []
    for i, cfg in enumerate(configs, 1):
        t0 = time.time()
        f1_mean, f1_std, acc = cv_score(train, cfg)
        rows.append({**cfg, "macro_f1": f1_mean, "f1_std": f1_std, "accuracy": acc})
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"[{i}/{len(configs)}] macro-F1 {f1_mean:.3f} (+/- {f1_std:.3f})  "
              f"acc {acc:.3f}  {time.time() - t0:.0f}s  {cfg}")

    res = pd.DataFrame(rows)
    print("\nTop 5 configurations:")
    print(res.sort_values("macro_f1", ascending=False).head(5).round(3).to_string(index=False))

    print("\nMean macro-F1 by hyperparameter value (random configs only; noisy):")
    random_rows = pd.DataFrame(rows[1:]).astype({k: str for k in SPACE}).fillna("None")
    for k in SPACE:
        print(f"\n{k}")
        print(random_rows.groupby(k)["macro_f1"].agg(["mean", "count"]).round(3).to_string())