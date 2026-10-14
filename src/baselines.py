from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.features import load_main_dataset, TIER1
    from sercsplit import SEED, random_split, element_split
except ModuleNotFoundError:
    from features import load_main_dataset, TIER1
    from split import SEED, random_split, element_split

MODELS = {
    "majority class": DummyClassifier(strategy="most_frequent"),
    "random forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
}

if __name__ == "__main__":
    df = load_main_dataset()
    for split, split_fn, grouped in [("RANDOM", random_split, False), ("ELEMENT", element_split, True)]:
        train, _ = split_fn(df)
        X, y = train[TIER1], train["label"]

        if grouped:
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
            groups = train["z"]
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            groups = None
        print(f"\n=== {split} split: 5-fold CV on train ===")
        for model_name, model in MODELS.items():
            res = cross_validate(model, X, y, groups = groups, cv=cv, scoring=["accuracy", "f1_macro"])
            print(f"{model_name:20s} "
                  f"accuracy {res['test_accuracy'].mean():.3f} (+/- {res['test_accuracy'].std():.3f})  "
                  f"macro-F1 {res['test_f1_macro'].mean():.3f} (+/- {res['test_f1_macro'].std():.3f})")