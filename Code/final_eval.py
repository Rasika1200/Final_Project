#!/usr/bin/env python3


import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import joblib

TEXT_COL = "clean_text"
LABEL_COL = "label"
MAX_LEN_LSTM = 50  # must match your BiLSTM script


# -------------------------------------------------------------------
# Helper: per-class F1 extractor
# -------------------------------------------------------------------
def get_per_class_f1(y_true, y_pred, labels):
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return [report[label]["f1-score"] for label in labels]


# -------------------------------------------------------------------
# Load test data
# -------------------------------------------------------------------
def load_test_data(path="test_clean.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    print("[DATA] Test shape:", df.shape)
    print("[DATA] Test label distribution:\n", df[LABEL_COL].value_counts())
    return df


# -------------------------------------------------------------------
# Baseline TF-IDF + Logistic Regression
# -------------------------------------------------------------------
def evaluate_baseline(test_df, model_dir="models/baseline"):
    print("\n=== Evaluating Baseline (TFIDF+LogReg) ===")

    tfidf = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.joblib"))
    clf = joblib.load(os.path.join(model_dir, "logreg_baseline.joblib"))
    le = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))

    X = tfidf.transform(test_df[TEXT_COL].astype(str))
    y_true = le.transform(test_df[LABEL_COL].astype(str))
    y_pred = clf.predict(X)

    labels = le.classes_.tolist()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"[BASELINE] Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")

    return y_true, y_pred, labels, acc, macro_f1


# -------------------------------------------------------------------
# BiLSTM evaluation
# -------------------------------------------------------------------
def evaluate_bilstm(test_df, model_dir="models/bilstm"):
    print("\n=== Evaluating BiLSTM ===")

    model = load_model(os.path.join(model_dir, "final_lstm_model.h5"))
    with open(os.path.join(model_dir, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    seqs = pad_sequences(
        tokenizer.texts_to_sequences(test_df[TEXT_COL].astype(str)),
        maxlen=MAX_LEN_LSTM,
        padding="post",
        truncating="post",
    )

    y_prob = model.predict(seqs, batch_size=256, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = le.transform(test_df[LABEL_COL].astype(str))
    labels = le.classes_.tolist()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"[BILSTM] Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")

    return y_true, y_pred, labels, acc, macro_f1


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    test_df = load_test_data("test_clean.csv")

    # 1) Evaluate baseline & BiLSTM
    y_lr, pred_lr, labels_lr, acc_lr, f1_lr = evaluate_baseline(test_df)
    y_lstm, pred_lstm, labels_lstm, acc_lstm, f1_lstm = evaluate_bilstm(test_df)

    # Sanity: same label order
    assert labels_lr == labels_lstm, "Label order mismatch between LR and BiLSTM"
    labels = labels_lr  # e.g. ['figurative', 'irony', 'regular', 'sarcasm']

    # 2) DistilBERT metrics (from your earlier run)
    # Test Accuracy: 0.7479985219854662
    # Test Macro F1: 0.6522444498455375
    acc_bert = 0.7479985219854662
    f1_bert = 0.6522444498455375

    # 3) Build summary table
    results = {
        "Model": ["TFIDF+LogReg", "BiLSTM", "DistilBERT"],
        "Accuracy": [acc_lr, acc_lstm, acc_bert],
        "Macro_F1": [f1_lr, f1_lstm, f1_bert],
    }
    df_results = pd.DataFrame(results)
    print("\n=== Summary Table (Test set) ===")
    print(df_results.to_string(index=False))
    df_results.to_csv("results_summary.csv", index=False)

    # 4) Heatmap of results (Accuracy & Macro F1)
    plt.figure(figsize=(5, 3))
    sns.heatmap(
        df_results.set_index("Model"),
        annot=True,
        fmt=".4f",
        cmap="Blues",
    )
    plt.title("Model Comparison (Accuracy & Macro F1)")
    plt.tight_layout()
    plt.savefig("results_heatmap.png", dpi=200)
    plt.close()
    print("[INFO] Saved: results_heatmap.png")

    # 5) Scatter plot: Accuracy vs Macro F1
    plt.figure(figsize=(5, 4))
    sns.scatterplot(
        data=df_results,
        x="Accuracy",
        y="Macro_F1",
        hue="Model",
        s=120,
    )
    for _, row in df_results.iterrows():
        plt.text(row["Accuracy"] + 0.0002, row["Macro_F1"] + 0.0002, row["Model"], fontsize=9)
    plt.title("Accuracy vs Macro F1")
    plt.tight_layout()
    plt.savefig("results_scatter.png", dpi=200)
    plt.close()
    print("[INFO] Saved: results_scatter.png")

    # 6) Per-class F1 for LR & BiLSTM (computed), DistilBERT (hard-coded from its test report)
    f1_lr_per_class = get_per_class_f1(y_lr, pred_lr, labels)
    f1_lstm_per_class = get_per_class_f1(y_lstm, pred_lstm, labels)

    # From your DistilBERT test classification_report:
    # figurative: f1 = 0.00
    # irony:      f1 = 0.80
    # regular:    f1 = 1.00
    # sarcasm:    f1 = 0.81
    f1_bert_per_class = [0.00, 0.80, 1.00, 0.81]

    per_class_f1 = pd.DataFrame({
        "Class": labels,
        "TFIDF+LogReg": f1_lr_per_class,
        "BiLSTM": f1_lstm_per_class,
        "DistilBERT": f1_bert_per_class,
    })
    print("\n=== Per-Class F1 (Test set) ===")
    print(per_class_f1.to_string(index=False))
    per_class_f1.to_csv("per_class_f1.csv", index=False)

    # Heatmap for per-class F1
    plt.figure(figsize=(6, 3))
    sns.heatmap(
        per_class_f1.set_index("Class"),
        annot=True,
        fmt=".3f",
        cmap="Reds",
    )
    plt.title("Per-Class F1 Across Models (Test set)")
    plt.tight_layout()
    plt.savefig("per_class_f1_heatmap.png", dpi=200)
    plt.close()
    print("[INFO] Saved: per_class_f1_heatmap.png")

    # 7) Extra metrics for LR & BiLSTM (no DistilBERT since we don't have raw preds)
    extra = {
        "Model": ["TFIDF+LogReg", "BiLSTM"],
        "Micro_F1": [
            f1_score(y_lr, pred_lr, average="micro"),
            f1_score(y_lstm, pred_lstm, average="micro"),
        ],
        "Weighted_F1": [
            f1_score(y_lr, pred_lr, average="weighted"),
            f1_score(y_lstm, pred_lstm, average="weighted"),
        ],
        "Cohen_Kappa": [
            cohen_kappa_score(y_lr, pred_lr),
            cohen_kappa_score(y_lstm, pred_lstm),
        ],
        "MCC": [
            matthews_corrcoef(y_lr, pred_lr),
            matthews_corrcoef(y_lstm, pred_lstm),
        ],
    }
    df_extra = pd.DataFrame(extra)
    print("\n=== Extra Metrics (Test set) ===")
    print(df_extra.to_string(index=False))
    df_extra.to_csv("extra_metrics.csv", index=False)

    print("\n[DONE] All evaluations & plots generated successfully.")


if __name__ == "__main__":
    main()
