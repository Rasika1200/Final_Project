#!/usr/bin/env python3


# ======================= IMPORTS =====================================
import os
import pickle
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    GlobalMaxPool1D,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical


# ======================= CONFIG ======================================
TEXT_COL = "clean_text"
LABEL_COL = "label"

MAX_VOCAB = 20000
MAX_LEN = 50
EMBEDDING_DIM = 128  # learned embeddings (no GloVe)

RANDOM_STATE = 42


# ======================= UTILS =======================================
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_confusion_matrix(y_true, y_pred, class_names, out_path, title="Confusion Matrix"):
    from sklearn.metrics import confusion_matrix

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


# ======================= DATA LOADING =================================
def load_data(train_path: str, test_path: str):
    print("[LOAD] Reading cleaned train & test...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Sanity checks
    assert TEXT_COL in train_df.columns and LABEL_COL in train_df.columns, \
        f"Expected columns {TEXT_COL}, {LABEL_COL} in train CSV"
    assert TEXT_COL in test_df.columns and LABEL_COL in test_df.columns, \
        f"Expected columns {TEXT_COL}, {LABEL_COL} in test CSV"

    before_train = train_df.shape[0]
    before_test = test_df.shape[0]

    train_df = train_df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    test_df = test_df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)

    print(f"Dropped {before_train - len(train_df)} train rows with NaN text/label")
    print(f"Dropped {before_test - len(test_df)} test rows with NaN text/label")

    print("Train shape:", train_df.shape)
    print("Test shape :", test_df.shape)
    print("Train label distribution:\n", train_df[LABEL_COL].value_counts())

    return train_df, test_df


# ======================= MAIN ========================================
def main():
    parser = argparse.ArgumentParser(
        description="BiLSTM model for sarcasm/irony/figurative detection (cleaned data)"
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
        default="models/bilstm",
        help="Directory to save model, tokenizer, and plots",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    # -------- 1. Load data --------
    train_df, test_df = load_data(args.train_csv, args.test_csv)

    # -------- 2. Encode labels with LabelEncoder (like LR baseline) --------
    le = LabelEncoder()
    y_all = le.fit_transform(train_df[LABEL_COL])
    y_test = le.transform(test_df[LABEL_COL])

    # Save label encoder for later use
    with open(os.path.join(args.output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    print("Label classes:", list(le.classes_))

    # Split train into train / validation
    train_split, val_split, y_train, y_val = train_test_split(
        train_df,
        y_all,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    X_train = train_split[TEXT_COL].astype(str)
    X_val = val_split[TEXT_COL].astype(str)
    X_test = test_df[TEXT_COL].astype(str)

    # -------- 3. Tokenizer & sequences --------
    print("[STEP] Fitting tokenizer on training text...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    def encode(texts):
        return pad_sequences(
            tokenizer.texts_to_sequences(texts),
            maxlen=MAX_LEN,
            padding="post",
            truncating="post",
        )

    X_train_seq = encode(X_train)
    X_val_seq = encode(X_val)
    X_test_seq = encode(X_test)

    num_classes = len(le.classes_)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # -------- 4. Build BiLSTM model --------
    print("[STEP] Building BiLSTM model...")
    model = Sequential(
        [
            Embedding(input_dim=MAX_VOCAB, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
            Bidirectional(LSTM(128, return_sequences=True)),
            GlobalMaxPool1D(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    # -------- 5. Train model --------
    checkpoint_path = os.path.join(args.output_dir, "best_lstm_model.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ]

    print("[STEP] Training BiLSTM...")
    history = model.fit(
        X_train_seq,
        y_train_cat,
        validation_data=(X_val_seq, y_val_cat),
        epochs=4,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # -------- 6. Evaluate on validation & test --------
    print("\n[STEP] Evaluating on validation set...")
    y_val_pred = np.argmax(model.predict(X_val_seq), axis=1)
    print("Val Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Val Macro F1:", f1_score(y_val, y_val_pred, average="macro"))
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=le.classes_))

    save_confusion_matrix(
        y_val,
        y_val_pred,
        class_names=le.classes_,
        out_path=os.path.join(args.output_dir, "confusion_val_lstm.png"),
        title="BiLSTM Confusion Matrix (Validation)",
    )

    print("\n[STEP] Evaluating on TEST set...")
    y_test_pred = np.argmax(model.predict(X_test_seq), axis=1)
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Test Macro F1:", f1_score(y_test, y_test_pred, average="macro"))
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))

    save_confusion_matrix(
        y_test,
        y_test_pred,
        class_names=le.classes_,
        out_path=os.path.join(args.output_dir, "confusion_test_lstm.png"),
        title="BiLSTM Confusion Matrix (Test)",
    )

    # -------- 7. Save final model & tokenizer --------
    final_model_path = os.path.join(args.output_dir, "final_lstm_model.h5")
    model.save(final_model_path)
    print(f"[INFO] Saved final BiLSTM model to {final_model_path}")

    tokenizer_path = os.path.join(args.output_dir, "tokenizer.pkl")
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[INFO] Saved tokenizer to {tokenizer_path}")

    print("\n[DONE] BiLSTM training + evaluation complete.")


if __name__ == "__main__":
    main()
