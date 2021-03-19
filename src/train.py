import joblib
import json
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline


def main():
    def evaluate(model, X, Y_true):
        Y_pred = model.predict(X)
        return {
            "# Samples": len(X),
            "Subset accuracy": accuracy_score(Y_true, Y_pred),
            "Accuracy": jaccard_score(Y_true, Y_pred, average="samples", zero_division=1),
            "Hamming similarity": 1 - hamming_loss(Y_true, Y_pred),
            "Precision": precision_score(Y_true, Y_pred, average="samples", zero_division=1),
            "Recall": recall_score(Y_true, Y_pred, average="samples", zero_division=1),
            "F1": f1_score(Y_true, Y_pred, average="samples", zero_division=1)
        }


    df = pd.read_pickle("data/out/df.pkl")
    X, Y = df[["summary"]], df.drop("summary", axis=1)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=10000, random_state=0)

    model = make_pipeline(
        make_column_transformer((TfidfVectorizer(min_df=10), 0)),
        MultiOutputClassifier(LogisticRegression(solver="liblinear", random_state=0), n_jobs=-1)
    )

    model.fit(X_train, Y_train)
    metrics = {"Train": evaluate(model, X_train, Y_train), "Val": evaluate(model, X_val, Y_val)}
    model.fit(X, Y)
    metrics["Final train"] = evaluate(model, X, Y)

    joblib.dump(model, "../model/model.joblib")
    with open("../model/metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
