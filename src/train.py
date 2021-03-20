import datetime
import joblib
import json
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from time import time


def main():
    config = json.load(open("config/config.json"))

    df = pd.read_pickle("data/out/df.pkl")
    X, Y = df[["summary"]], df.drop("summary", axis=1)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=10000, random_state=0)

    algorithm = config["algorithm"]
    if algorithm == "baseline":
        classifier = DummyClassifier()
    elif algorithm == "logistic":
        classifier = LogisticRegression(solver="liblinear", random_state=0)
    elif algorithm == "svm":
        classifier = LinearSVC(random_state=0)
    else:
        raise ValueError(f"{algorithm} is not a valid algorithm")

    model = make_pipeline(make_column_transformer((TfidfVectorizer(), 0)), MultiOutputClassifier(classifier))

    def evaluate(model, X, Y, train=True):
        metrics = {"# samples": len(X)}

        if train:
            start = time()
            model.fit(X, Y)
            end = time()
            metrics["train time"] = end - start

        start = time()
        Y_pred = model.predict(X)
        end = time()
        metrics["inference time"] = end - start
        metrics["exact match"] = accuracy_score(Y, Y_pred)
        metrics["Hamming similarity"] = 1 - hamming_loss(Y, Y_pred)
        metrics["Jaccard similarity"] = jaccard_score(Y, Y_pred, average="samples", zero_division=0)
        metrics["precision"] = precision_score(Y, Y_pred, average="samples", zero_division=0)
        metrics["recall"] = recall_score(Y, Y_pred, average="samples", zero_division=0)
        metrics["f1"] = f1_score(Y, Y_pred, average="samples", zero_division=0)
        return metrics

    metrics = {
        "train": evaluate(model, X_train, Y_train),
        "validation": evaluate(model, X_val, Y_val, train=False),
        "full_train": evaluate(model, X, Y)
    }

    joblib.dump(model, "model/model.joblib")
    with open(f"model/{algorithm} {datetime.datetime.today()}.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
