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

MEASURED_ONLY = False                    # run once with False and once with True
N_BOOT = 1000
SEEDS = [SEED + i for i in range(5)]     # training seeds; scores are averaged over these
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


def score(t, ps):
    # (5-class, 4-class) macro-F1, averaged over the per-seed predictions in ps
    f1s = [f1_vector(t, p) for p in ps]
    return (np.mean([np.nanmean(f) for f in f1s]),
            np.mean([np.nanmean(f[:4]) for f in f1s]))


def make_boots(t, groups):
    # bootstrap index sets: rows (random split) or whole elements (element split)
    rng = np.random.default_rng(SEED)
    if groups is None:
        return [rng.integers(0, len(t), len(t)) for _ in range(N_BOOT)]
    members = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    keys = np.array(list(members))
    return [np.concatenate([members[g] for g in rng.choice(keys, len(keys))])
            for _ in range(N_BOOT)]


def ci4(t, ps, boots):
    vals = [score(t[i], [p[i] for p in ps])[1] for i in boots]
    return np.percentile(vals, [2.5, 97.5])


def diff_ci(t, ps_a, ps_b, boots):
    # paired difference in four-class macro-F1: both sets are scored on the same resamples
    point = score(t, ps_a)[1] - score(t, ps_b)[1]
    d = [score(t[i], [p[i] for p in ps_a])[1] - score(t[i], [p[i] for p in ps_b])[1]
         for i in boots]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return point, lo, hi


def fit_predict(tr, te, grouped, feats, kind, params):
    # one prediction array per training seed
    preds = []
    for s in SEEDS:
        if kind == "mlp":
            fit, val = inner_split(tr, grouped, s)
            with redirect_stdout(io.StringIO()):
                model, _ = train_mlp(fit, val, features=feats, seed=s, **params)
            scaler = StandardScaler().fit(fit[feats])
            pred = np.array(LABELS)[predict(model, scaler, te, feats)]
        else:
            rf = RandomForestClassifier(random_state=s, n_jobs=-1, **params)
            rf.fit(tr[feats], tr["label"])
            pred = rf.predict(te[feats])
        preds.append(encode(pred))
    return preds


if __name__ == "__main__":
    print("Scoring on the TEST sets (seed-averaged). Report these numbers whatever they show.")
    df = load_main_dataset()
    tag = "measured" if MEASURED_ONLY else "all"
    rows, contrast_rows = [], []

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
            boots = make_boots(t, groups)

            print(f"\n=== {split_name} | rows complete through '{LAYERS[last][0]}' | "
                  f"train {len(tr)}, test {len(te)} ===")
            print(te["label"].value_counts().to_dict())

            preds, table = {}, []
            for layer_name, feats in LAYERS[: last + 1]:
                for model_name, (kind, params) in MODELS.items():
                    ps = fit_predict(tr, te, grouped, feats, kind, params)
                    preds[(layer_name, model_name)] = ps
                    m5, m4 = score(t, ps)
                    lo, hi = ci4(t, ps, boots)
                    per_class = np.nanmean([f1_vector(t, p) for p in ps], axis=0)
                    table.append({
                        "layer": layer_name, "model": model_name,
                        "macro_f1": m5, "macro_f1_4class": m4,
                        "ci_4class": f"[{lo:.3f}, {hi:.3f}]",
                        "seed_sd_4class": np.std([np.nanmean(f1_vector(t, p)[:4]) for p in ps]),
                        **{f"f1_{lab}": v for lab, v in zip(LABELS, per_class)},
                    })
            print(pd.DataFrame(table).round(3).to_string(index=False))

            pairs = [(f"{lay}: MLP - RF", preds[(lay, "MLP")], preds[(lay, "RF")])
                     for lay, _ in LAYERS[: last + 1]]
            pairs += [(f"{m}: {lay} - T1", preds[(lay, m)], preds[("T1", m)])
                      for lay, _ in LAYERS[1 : last + 1] for m in MODELS]
            ctable = []
            for name, a, b in pairs:
                point, lo, hi = diff_ci(t, a, b, boots)
                ctable.append({"contrast": name, "diff_4class": point,
                               "ci_95": f"[{lo:+.3f}, {hi:+.3f}]"})
            print("\nPaired contrasts, four-class macro-F1 (an interval excluding 0 suggests a real difference):")
            print(pd.DataFrame(ctable).round(3).to_string(index=False))

            meta = {"split": split_name, "experiment": LAYERS[last][0],
                    "n_train": len(tr), "n_test": len(te)}
            rows += [{**meta, **r} for r in table]
            contrast_rows += [{**meta, **c} for c in ctable]

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(f"results/final_eval_{tag}.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(f"results/final_eval_contrasts_{tag}.csv", index=False)