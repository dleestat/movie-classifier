import json
import numpy as np
import os
import pandas as pd
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

    statistics["records"]["data/out/df.pkl"] = len(df)
    statistics["normalized_genres"] = df.genres.explode().value_counts().to_dict()

    with open('data/profiling/statistics.json', 'w') as f:
        json.dump(statistics, f, indent=2)
    df.to_pickle("data/out/df.pkl")


if __name__ == "__main__":
    main()
