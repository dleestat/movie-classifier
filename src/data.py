import pandas as pd


def main():
    print("===== data =====")

    genres = pd.read_csv(
        "data/raw/movie.metadata.tsv",
        converters={'genres': lambda x: list(eval(x).values())},
        delimiter='\t',
        header=None,
        index_col='id',
        names=['id', 'genres'],
        usecols=[0, 8])

    summaries = pd.read_csv(
        "data/raw/plot_summaries.txt",
        delimiter='\t',
        header=None,
        index_col='id',
        names=['id', 'summary'])

    df = summaries.merge(genres, on='id').reset_index(drop=True)


if __name__ == "main":
    main()
