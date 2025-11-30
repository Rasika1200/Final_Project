#!/usr/bin/env python3
"""
04_distilbert_model.py
Fine-tune DistilBERT for Figurative Language Detection on Tweets

Data: train_clean.csv, test_clean.csv
Columns (per row):
  raw_text,label,clean_text,lemma_text,char_len,word_len,num_exclam,num_question,
  num_hashtags,num_mentions,num_caps,cap_ratio

We use:
  - clean_text as input text
  - label as target (figurative, irony, regular, sarcasm)

Run:
    python bert_finetune.py
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

RANDOM_STATE = 42

TEXT_COL = "lemma_text"
LABEL_COL = "label"


# ============================================================
# Config + Dataset
# ============================================================
@dataclass
class TweetConfig:
    max_length: int = 64


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, cfg: TweetConfig):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# ============================================================
# Utility: confusion matrix plotting
# ============================================================
def save_confusion_matrix(y_true, y_pred, class_names, out_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved confusion matrix to {out_path}")


# ============================================================
# Data loading / preprocessing
# ============================================================
def load_data(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Expect cleaned columns
    assert TEXT_COL in train_df.columns and LABEL_COL in train_df.columns, \
        f"Expected columns {TEXT_COL}, {LABEL_COL} in train_clean.csv"
    assert TEXT_COL in test_df.columns and LABEL_COL in test_df.columns, \
        f"Expected columns {TEXT_COL}, {LABEL_COL} in test_clean.csv"

    before_train = train_df.shape[0]
    before_test = test_df.shape[0]

    train_df = train_df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    test_df = test_df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)

    print(f"Dropped {before_train - train_df.shape[0]} train rows with NaN labels/text")
    print(f"Dropped {before_test - test_df.shape[0]} test rows with NaN labels/text")

    print("Train shape:", train_df.shape)
    print("Test shape :", test_df.shape)
    print("Train label distribution:")
    print(train_df[LABEL_COL].value_counts())

    return train_df, test_df


def encode_labels(train_df, test_df):
    """
    Create a stable mapping label -> id based on sorted unique labels.
    """
    all_labels = sorted(train_df[LABEL_COL].unique())
    label2id = {lab: i for i, lab in enumerate(all_labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    train_ids = train_df[LABEL_COL].map(label2id).astype(int)
    test_ids = test_df[LABEL_COL].map(label2id).astype(int)

    print("Label2id mapping:", label2id)
    return train_ids, test_ids, label2id, id2label


def make_splits(train_df, train_ids, val_frac=0.15):
    X_train, X_val, y_train, y_val = train_test_split(
        train_df[TEXT_COL],
        train_ids,
        test_size=val_frac,
        random_state=RANDOM_STATE,
        stratify=train_ids,
    )
    return X_train, X_val, y_train, y_val


# ============================================================
# Metrics for Trainer
# ============================================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on figurative/sarcasm/irony tweets (cleaned data)"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="train_clean.csv",
        help="Path to train_clean.csv",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="test_clean.csv",
        help="Path to test_clean.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/distilbert",
        help="Directory to save the fine-tuned model and outputs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data
    train_df, test_df = load_data(args.train_csv, args.test_csv)
    train_ids, test_ids, label2id, id2label = encode_labels(train_df, test_df)

    # 2. Train/val split
    X_train, X_val, y_train, y_val = make_splits(train_df, train_ids)

    # 3. Tokenizer & datasets
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    cfg = TweetConfig(max_length=64)

    train_dataset = TweetDataset(X_train, y_train, tokenizer, cfg)
    val_dataset = TweetDataset(X_val, y_val, tokenizer, cfg)
    test_dataset = TweetDataset(test_df[TEXT_COL], test_ids, tokenizer, cfg)

    # 4. Model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    # 5. Training arguments (minimal, compatible with older transformers)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        seed=RANDOM_STATE,
        logging_steps=200,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 6. Train
    print("Starting DistilBERT training...")
    trainer.train()

    # 7. Evaluate on validation set
    print("Evaluating on validation set...")
    val_output = trainer.predict(val_dataset)
    val_preds = np.argmax(val_output.predictions, axis=-1)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    print("Validation Accuracy:", val_acc, "Macro F1:", val_f1)
    print("\nValidation classification report:")
    print(
        classification_report(
            y_val,
            val_preds,
            target_names=[id2label[i] for i in range(len(id2label))],
        )
    )

    save_confusion_matrix(
        y_val,
        val_preds,
        class_names=[id2label[i] for i in range(len(id2label))],
        out_path=os.path.join(args.output_dir, "confusion_val_distilbert.png"),
        title="DistilBERT Confusion Matrix (Validation)",
    )

    # 8. Evaluate on test set
    print("Evaluating on TEST set...")
    test_output = trainer.predict(test_dataset)
    test_preds = np.argmax(test_output.predictions, axis=-1)
    test_acc = accuracy_score(test_ids, test_preds)
    test_f1 = f1_score(test_ids, test_preds, average="macro")
    print("Test Accuracy:", test_acc, "Macro F1:", test_f1)
    print("\nClassification report on test:")
    print(
        classification_report(
            test_ids,
            test_preds,
            target_names=[id2label[i] for i in range(len(id2label))],
        )
    )

    save_confusion_matrix(
        test_ids,
        test_preds,
        class_names=[id2label[i] for i in range(len(id2label))],
        out_path=os.path.join(args.output_dir, "confusion_test_distilbert.png"),
        title="DistilBERT Confusion Matrix (Test)",
    )

    # 9. Save model & tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned DistilBERT model + tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
