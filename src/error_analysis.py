import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict

try:
    from src.features import load_main_dataset, TIER1
    from src.split import SEED, random_split, element_split
except ModuleNotFoundError:
    from features import load_main_dataset, TIER1
    from split import SEED, random_split, element_split

LABELS = ["beta_minus", "beta_plus_ec", "alpha", "stable", "other"]

if __name__ == "__main__":
    df = load_main_dataset()

    for name, split_fn, grouped in [("RANDOM", random_split, False),
                                    ("ELEMENT", element_split, True)]:
        train, _ = split_fn(df)          # test set deliberately ignored
        X, y = train[TIER1], train["label"]

        if grouped:
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
            groups = train["z"]
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            groups = None

        rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
        pred = cross_val_predict(rf, X, y, groups=groups, cv=cv)

        print(f"\n=== {name} split: random forest, cross-validated predictions on train ===")
        print(classification_report(y, pred, labels=LABELS, digits=3))
        print("Confusion matrix (rows = true label, columns = predicted):")
        print(pd.DataFrame(confusion_matrix(y, pred, labels=LABELS),
                           index=LABELS, columns=LABELS))
        other = train["label"] == "other"
        sub = pd.DataFrame({"mode": train.loc[other, "decay_1"],
                            "caught": pred[other.to_numpy()] == "other"})
        print("\nInside 'other': original decay mode, number caught, total")
        print(sub.groupby("mode")["caught"].agg(["sum", "count"]))