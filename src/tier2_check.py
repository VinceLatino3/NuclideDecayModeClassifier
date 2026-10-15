import pandas as pd

try:
    from src.features import load_main_dataset
except ModuleNotFoundError:
    from features import load_main_dataset

T2 = ["binding", "massexcess", "sn", "sp", "qa", "qec", "qbm", "qbm_n"]

if __name__ == "__main__":
    df = load_main_dataset()
    print("Rows:", len(df))

    print("\nShare missing, by column and label:")
    print(df[T2].isna().groupby(df["label"]).mean().round(2).T.to_string())

    complete = df[T2].notna().all(axis=1)
    print("\nRows complete on all 8 columns:", int(complete.sum()), "of", len(df))
    print("\nLabel counts among complete rows vs. all rows:")
    print(pd.DataFrame({"all": df["label"].value_counts(),
                        "complete": df.loc[complete, "label"].value_counts()}).to_string())

    print("\nComplete rows split by extrapolated masses:")
    print(pd.crosstab(df.loc[complete, "label"], df.loc[complete, "extrapolated"]).to_string())

    print("\nColumn summaries (check units and ranges):")
    print(df[T2].describe().round(1).to_string())