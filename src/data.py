import os
import pandas as pd


def main():
    print("\n====== data ======\n")
    os.makedirs("data/out", exist_ok=True)
    os.makedirs("data/profiling", exist_ok=True)

    print("data".ljust(30), "records")

    genres = pd.read_csv(
        "data/raw/movie.metadata.tsv",
        converters={"genres": lambda x: list(eval(x).values())},
        delimiter="\t",
        header=None,
        index_col="id",
        names=["id", "genres"],
        usecols=[0, 8])
    print("data/raw/movie.metadata.tsv".ljust(30), len(genres))

    summaries = pd.read_csv(
        "data/raw/plot_summaries.txt",
        delimiter="\t",
        header=None,
        index_col="id",
        names=["id", "summary"])
    print("data/raw/plot_summaries.txt".ljust(30), len(summaries))

    df = summaries.merge(genres, on="id").reset_index(drop=True)
    print("merged".ljust(30), len(df))

    df.to_pickle("data/out/df.pkl")
    print("data/out/df.pkl".ljust(30), len(df))


if __name__ == "__main__":
    main()
