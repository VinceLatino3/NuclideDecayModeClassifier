from sklearn.model_selection import train_test_split, StratifiedGroupKFold

try:
    from src.features import load_main_dataset
except ModuleNotFoundError:
    from features import load_main_dataset

SEED = 42
TEST_SIZE = .2


#IMPORTANT; We use two different splits, one traditional one that is completely random in splitting test and train data. 
#The other is a stratified split that ensures all isotopes with the same Z are together, so the model cannot train and test on near identical isotopes.

def random_split(df):
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"])
    return train, test

def element_split(df):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, testidx = next(sgkf.split(df, df["label"], groups=df["z"]))
    return df.iloc[train_idx], df.iloc[testidx]

if __name__ == "__main__":
    df = load_main_dataset()

    for name, split_fn in [("RANDOM", random_split), ("ELEMENT", element_split)]:
        train, test = split_fn(df)
        print(f"\n=== {name} split ===")
        print("train:", len(train), " test:", len(test),
              f" test share: {len(test) / len(df):.2f}")
        counts = (
            train["label"].value_counts().rename("train").to_frame()
            .join(test["label"].value_counts().rename("test"))
        )
        counts["test_share"] = (counts["test"] / (counts["train"] + counts["test"])).round(2)
        print(counts)

        # sanity checks
        assert not set(train.index) & set(test.index), "row overlap between train and test!"
        if name == "ELEMENT":
            assert not set(train["z"]) & set(test["z"]), "an element appears in both!"
            