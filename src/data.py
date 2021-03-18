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

    with open('data/profiling/genres.json', 'w') as f:
        json.dump(metadata.genres.explode().value_counts().to_dict(), f, indent=2)

    summaries = pd.read_csv(
        "data/raw/plot_summaries.txt",
        delimiter="\t",
        header=None,
        index_col="id",
        names=["id", "summary"]
    )

    df = summaries.merge(metadata, on="id")

    num_records = {
        "data/raw/movie.metadata.tsv": len(metadata),
        "data/raw/plot_summaries.txt": len(summaries),
        "merged": len(df)
    }

    def clean_summary(summary):
        return (
            summary
            .str.replace(r'{{.*?}}', '')  # Remove Wikipedia tags
            .str.replace(r'http\S+', '')  # Remove URLs
            .str.replace(r'\s+', ' ')     # Combine whitespace
            .str.strip()                  # Strip whitespace
            .replace('', pd.NA)           # Replace empty strings with NA
        )

    normalized_genres = json.load(open("config.json"))["normalized_genres"]

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
    num_records["data/out/df.pkl"] = len(df)
    with open('data/profiling/num_records.json', 'w') as f:
        json.dump(num_records, f, indent=2)
    df.to_pickle("data/out/df.pkl")


if __name__ == "__main__":
    main()
