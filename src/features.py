import numpy as np
import pandas as pd

try:
    from src.get_data import load_ground_states, build_labeled_data
except ModuleNotFoundError:
    from get_data import load_ground_states, build_labeled_data

#Observed "magic" numbers in nuclides
MAGIC = np.array([2, 8, 20, 28, 50, 82, 126])

#Desired Tier 1 features
TIER1 = ["z", "n", "A", "n_minus_z", "n_frac", "z_even", "n_even", "dz_magic", "dn_magic"]

def dist_to_magic(values):
    return np.abs(values.to_numpy()[:, None] - MAGIC[None, :]).min(axis=1)

def tier1_features(df):
    df = df.copy()
    df["A"] = df["z"] + df["n"]
    df["n_minus_z"] = df["n"] - df["z"]
    df["n_frac"] = df["n"] / df["A"] #Divide by A to avoid dividing by zero
    df["z_even"] = (df["z"] % 2 == 0).astype(int)
    df["n_even"] = (df["n"] % 2 == 0).astype(int)
    df["dz_magic"] = dist_to_magic(df["z"])
    df["dn_magic"] = dist_to_magic(df["n"])
    return df


def load_main_dataset():
    df = load_ground_states()
    df_model = build_labeled_data(df)
    df_model = df_model[~df_model["no_half_life"]] #measured half-lives only
    return tier1_features(df_model)

if __name__ == "__main__":
    df_model = load_main_dataset()
    X = df_model[TIER1]
    print("X shape:", X.shape)
    print("Missing values in X:", int(X.isna().sum().sum()))
    print("\nLabel counts (main dataset):")
    print(df_model["label"].value_counts())