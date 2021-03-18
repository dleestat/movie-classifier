from src import data
import sys


def main(targets):
    if "data" in targets:
        data.main()

    if "train" in targets:
        ...


if __name__ == "main":
    main(sys.argv[1:])
