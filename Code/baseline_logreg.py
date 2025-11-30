#!/usr/bin/env python3

import argparse
import os
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

RANDOM_STATE = 42

# We will use these columns from your Phase-1 output:
TEXT_COL = "lemma_text"
LABEL_COL = "label"


# ================================================
#   CONFUSION MATRIX PLOT SAVER
# ================================================
def save_confusion_matrix(y_true, y_pred, class_names, out_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved confusion matrix to {out_path}")


# ================================================
#   DATA LOADING
# ================================================
def load_data(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    assert TEXT_COL in train_df.columns and LABEL_COL in train_df.columns
    assert TEXT_COL in test_df.columns and LABEL_COL in test_df.columns

    train_df = train_df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    test_df = test_df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Train label distribution:\n", train_df[LABEL_COL].value_counts())

    return train_df, test_df


# ================================================
#   BASELINE BUILD + TRAIN
# ================================================
def build_and_train_baseline(
    X_train, y_train, X_val, y_val, output_dir: str
):
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
    )

    X_train_vec = tfidf.fit_transform(X_train)
    X_val_vec = tfidf.transform(X_val)

    print("TF–IDF shapes:", X_train_vec.shape, X_val_vec.shape)

    base_clf = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        max_iter=200,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    param_grid = {"C": [0.1, 1.0, 10.0]}

    grid = GridSearchCV(
        base_clf,
        param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=2,
    )

    print("Running GridSearchCV...")
    grid.fit(X_train_vec, y_train)

    print("Best params:", grid.best_params_)
    best_clf = grid.best_estimator_

    # ----------- VALIDATION METRICS -----------
    val_preds = best_clf.predict(X_val_vec)

    print("\n=== Validation metrics ===")
    print("Accuracy:", accuracy_score(y_val, val_preds))
    print("Macro F1:", f1_score(y_val, val_preds, average="macro"))
    print("\nClassification report:\n", classification_report(y_val, val_preds))

    # SAVE CONFUSION MATRIX (VALIDATION)
    class_names = [str(i) for i in np.unique(y_train)]
    os.makedirs(output_dir, exist_ok=True)
    save_confusion_matrix(
        y_val,
        val_preds,
        class_names,
        out_path=os.path.join(output_dir, "confusion_val.png"),
        title="Confusion Matrix (Validation)"
    )

    # SAVE MODEL ARTIFACTS
    joblib.dump(tfidf, os.path.join(output_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(best_clf, os.path.join(output_dir, "logreg_baseline.joblib"))

    print(f"\nSaved TF–IDF and Logistic Regression model to {output_dir}")

    return best_clf, tfidf


# ================================================
#   TEST EVALUATION
# ================================================
def eval_on_test(clf, tfidf, X_test, y_test, output_dir):
    X_test_vec = tfidf.transform(X_test)
    preds = clf.predict(X_test_vec)

    print("\n=== Test metrics ===")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Macro F1:", f1_score(y_test, preds, average="macro"))
    print("\nClassification report:\n", classification_report(y_test, preds))

    # Save test confusion matrix
    class_names = [str(i) for i in np.unique(y_test)]
    save_confusion_matrix(
        y_test,
        preds,
        class_names,
        out_path=os.path.join(output_dir, "confusion_test.png"),
        title="Confusion Matrix (Test)"
    )


# ================================================
#   MAIN
# ================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="train_clean.csv")
    parser.add_argument("--test_csv", type=str, default="test_clean.csv")
    parser.add_argument("--output_dir", type=str, default="models/baseline")
    args = parser.parse_args()

    train_df, test_df = load_data(args.train_csv, args.test_csv)

    le = LabelEncoder()
    y_train_all = le.fit_transform(train_df[LABEL_COL])
    y_test = le.transform(test_df[LABEL_COL])

    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(le, os.path.join(args.output_dir, "label_encoder.joblib"))

    X_train, X_val, y_train, y_val = train_test_split(
        train_df[TEXT_COL],
        y_train_all,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_train_all,
    )

    clf, tfidf = build_and_train_baseline(
        X_train, y_train, X_val, y_val, args.output_dir
    )

    eval_on_test(
        clf,
        tfidf,
        test_df[TEXT_COL],
        y_test,
        args.output_dir
    )


if __name__ == "__main__":
    main()
