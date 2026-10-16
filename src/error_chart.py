from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    from src.features import load_main_dataset
    from src.split import SEED, random_split, element_split
    from src.tier2 import LAYERS
    from src.compare import RF_TUNED
except ModuleNotFoundError:
    from features import load_main_dataset
    from split import SEED, random_split, element_split
    from tier2 import LAYERS
    from compare import RF_TUNED

COLORS = {"beta_minus": "tab:blue", "beta_plus_ec": "tab:red", "alpha": "tab:green",
          "stable": "black", "other": "tab:orange"}

if __name__ == "__main__":
    df = load_main_dataset()
    needed = LAYERS[3][1]                       # rows complete through Q-values
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)

    for r, (split_name, split_fn) in enumerate([("random", random_split),
                                                ("element", element_split)]):
        train, test = split_fn(df)
        tr, te = train.dropna(subset=needed), test.dropna(subset=needed)
        true = te["label"].to_numpy()
        for c, layer_i in enumerate([0, 3]):    # Tier 1 vs. +Q
            layer_name, feats = LAYERS[layer_i]
            rf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **RF_TUNED)
            rf.fit(tr[feats], tr["label"])
            wrong = rf.predict(te[feats]) != true

            ax = axes[r][c]
            ax.scatter(df["n"], df["z"], s=3, c="lightgray")                    # every nuclide
            ax.scatter(te.loc[~wrong, "n"], te.loc[~wrong, "z"], s=5, c="gray")  # correct test nuclides
            for lab, color in COLORS.items():                                    # errors, by true class
                m = wrong & (true == lab)
                ax.scatter(te.loc[m, "n"], te.loc[m, "z"], s=30, c=color,
                           label=f"{lab}: {int(m.sum())}")
            ax.set_title(f"{split_name} split, {layer_name} features: "
                         f"{int(wrong.sum())} errors of {len(te)} test nuclides")
            ax.legend(title="errors by true class", fontsize=8, loc="upper left")
            if r == 1:
                ax.set_xlabel("Neutrons (N)")
            if c == 0:
                ax.set_ylabel("Protons (Z)")

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/error_chart.png", dpi=150)
    plt.show()