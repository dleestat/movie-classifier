import os
from src import data
import sys


def main(targets):
    if "download" in targets:
        os.system("src/download.sh")

    if "data" in targets:
        data.main()

    if "train" in targets:
        ...


if __name__ == "__main__":
    main(sys.argv[1:])
