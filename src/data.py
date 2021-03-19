import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer


def main():
    os.makedirs("data/out", exist_ok=True)
    os.makedirs("data/profiling", exist_ok=True)

    metadata = pd.read_csv(
        "data/raw/movie.metadata.tsv",
        converters={"genres": lambda x: list(eval(x).values())},
        delimiter="\t",
        header=None,
        index_col="id",
        names=["id", "genres"],
        usecols=[0, 8]
    )

    summaries = pd.read_csv(
        "data/raw/plot_summaries.txt",
        delimiter="\t",
        header=None,
        index_col="id",
        names=["id", "summary"]
    )

    df = summaries.merge(metadata, on="id")

    statistics = {
        "records": {
            "data/raw/movie.metadata.tsv": len(metadata),
            "data/raw/plot_summaries.txt": len(summaries),
            "merged": len(df)
        },
        "normalized_genres": ...,
        "genres": metadata.genres.explode().value_counts().to_dict()
    }

    normalized_genres = json.load(open("config/config.json"))["normalized_genres"]

    def clean_summary(summary):
        return (
            summary
            .str.replace(r'{{.*?}}', '')  # Remove Wikipedia tags
            .str.replace(r'http\S+', '')  # Remove URLs
            .str.replace(r'\s+', ' ')     # Combine whitespace
            .str.strip()                  # Strip whitespace
            .replace('', pd.NA)           # Replace empty strings with NA
        )

    def normalize_genres(genres):
        normalized = []
        for genre in genres:
            if genre in normalized_genres:
                normalized.extend(normalized_genres[genre])
        return list(np.unique(normalized)) if normalized else pd.NA

    df = df.assign(
        summary=clean_summary(df.summary),
        genres=df.genres.apply(normalize_genres)
    ).dropna().reset_index(drop=True)

    mlb = MultiLabelBinarizer()
    summaries, labels = df[["summary"]], pd.DataFrame(mlb.fit_transform(df.genres), columns=mlb.classes_)
    df = pd.concat([summaries, labels], axis=1)
    df.to_pickle("data/out/df.pkl")

    statistics["records"]["data/out/df.pkl"] = len(df)
    statistics["normalized_genres"] = labels.sum().sort_values(ascending=False).to_dict()
    with open('data/profiling/statistics.json', 'w') as f:
        json.dump(statistics, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        labels.corr(),
        annot=True,
        cbar=False,
        cmap="bwr",
        fmt=".1f",
        square=True,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.tick_params(left=False, bottom=False)
    fig.savefig("data/profiling/label_correlation.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
