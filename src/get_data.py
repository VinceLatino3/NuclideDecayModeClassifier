import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RAW = Path("data/raw/ground_states.csv")

PLOT = True


def load_ground_states():
    return pd.read_csv(RAW)


def make_label(row):
    #Must check first since some nuclides do decay, but at such a slow rate they are listed as STABLE
    if str(row["half_life"]).strip() == "STABLE":
        return "stable"
    mode = row["decay_1"]
    if pd.isna(mode):
        #Now stable values are stripped, we drop values with unkown decay mode
        return None
    if mode in ("EC+B+", "EC", "B+"):
        return "beta_plus_ec"
    if mode == "B-":
        return "beta_minus"
    if mode == "A":
        return "alpha"
    return "other"  #Forms of decay we are ignoring in this model


def build_labeled_data(df):
    df = df.copy()
    df["label"] = df.apply(make_label, axis=1)
    df = df.dropna(subset=["label"]).copy()
    # Flags for evaluation/filtering only, never model input features
    df["extrapolated"] = df["me_systematics"] == "Y"
    df["no_half_life"] = df["half_life"].isna()
    return df

#Creates Chart of nuclides colored by decay mode
def plot_chart(df_model):
    colors = {
        "beta_minus": "tab:blue",
        "beta_plus_ec": "tab:red",
        "alpha": "tab:green",
        "other": "tab:orange",
        "stable": "black",
    }
    fig, ax = plt.subplots(figsize=(12, 8))
    for lab in ["beta_minus", "beta_plus_ec", "alpha", "other", "stable"]:
        g = df_model[df_model["label"] == lab]
        ax.scatter(g["n"], g["z"], s=8, c=colors[lab], label=lab)
    ax.set_xlabel("Neutrons (N)")
    ax.set_ylabel("Protons (Z)")
    ax.legend(markerscale=2)
    plt.show()


if __name__ == "__main__":
    df = load_ground_states()
    print("Raw shape:", df.shape)

    df_model = build_labeled_data(df)
    print("\nRows kept:", len(df_model))
    print(df_model["label"].value_counts())

    if PLOT:
        plot_chart(df_model)