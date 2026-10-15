import io
from contextlib import redirect_stdout

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from src.features import load_main_dataset, TIER1
    from src.split import SEED, random_split, element_split
    from src.mlp import train_mlp, LABELS, LABEL_TO_ID, DEVICE
except ModuleNotFoundError:
    from features import load_main_dataset, TIER1
    from split import SEED, random_split, element_split
    from mlp import train_mlp, LABELS, LABEL_TO_ID, DEVICE


def inner_split(fold_train, grouped):
    # carve an early-stopping validation set out of the fold's training rows
    if grouped:
        sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
        tr_idx, va_idx = next(sgkf.split(fold_train, fold_train["label"], groups=fold_train["z"]))
        return fold_train.iloc[tr_idx], fold_train.iloc[va_idx]
    return train_test_split(fold_train, test_size=0.15,
                            stratify=fold_train["label"], random_state=SEED)


def predict(model, scaler, df):
    X = torch.tensor(scaler.transform(df[TIER1]), dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        return model(X).argmax(dim=1).cpu().numpy()


def cv_mlp(train, grouped, use_class_weights):
    if grouped:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = cv.split(train, train["label"], groups=train["z"])
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = cv.split(train, train["label"])

    accs, f1s, all_true, all_pred = [], [], [], []
    for tr_idx, ev_idx in splits:
        fold_train, fold_eval = train.iloc[tr_idx], train.iloc[ev_idx]
        tr, va = inner_split(fold_train, grouped)

        with redirect_stdout(io.StringIO()):   # silence the per-epoch printing
            model, _ = train_mlp(tr, va, use_class_weights=use_class_weights)

        # same scaler train_mlp fit internally: fit on `tr` only
        scaler = StandardScaler().fit(tr[TIER1])
        pred = predict(model, scaler, fold_eval)
        true = fold_eval["label"].map(LABEL_TO_ID).to_numpy()

        accs.append(accuracy_score(true, pred))
        f1s.append(f1_score(true, pred, average="macro"))
        all_true.append(true)
        all_pred.append(pred)
    return accs, f1s, np.concatenate(all_true), np.concatenate(all_pred)


if __name__ == "__main__":
    print("device:", DEVICE)
    df = load_main_dataset()

    for name, split_fn, grouped in [("RANDOM", random_split, False),
                                    ("ELEMENT", element_split, True)]:
        train, _ = split_fn(df)          # test set deliberately ignored
        for weighted in (True, False):
            accs, f1s, true, pred = cv_mlp(train, grouped, weighted)
            print(f"\n=== {name} split, MLP, class weights: {weighted} ===")
            print(f"accuracy {np.mean(accs):.3f} (+/- {np.std(accs):.3f})  "
                  f"macro-F1 {np.mean(f1s):.3f} (+/- {np.std(f1s):.3f})")
            print(classification_report(true, pred, labels=list(range(len(LABELS))),
                                        target_names=LABELS, digits=3))