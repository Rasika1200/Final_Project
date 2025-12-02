#!/usr/bin/env python3


import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Layer,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ======================= CONFIG =======================
TEXT_COL = "clean_text"     # column with cleaned tweet text
LABEL_COL = "label"

MAX_VOCAB = 20000
MAX_LEN = 40                # shorter – avoids too much padding for tweets
EMBEDDING_DIM = 256         # capacity for semantics
RANDOM_STATE = 42


# ======================= UTILS ========================
def ensure_dir(path: str):
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
    print(f"[INFO] Saved confusion matrix -> {out_path}")


# ======================= ATTENTION ====================
class BahdanauAttention(Layer):
    """
    Bahdanau-style attention over time steps.
    Input:  (batch, timesteps, features)
    Output: (batch, features)
    """
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.W1 = Dense(units)
        self.V = Dense(1)

    def call(self, values):
        # values: (batch, timesteps, features)
        score = self.V(tf.nn.tanh(self.W1(values)))  # (batch, timesteps, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch, timesteps, 1)
        context_vector = attention_weights * values       # (batch, timesteps, features)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, features)
        return context_vector


# ======================= MAIN =========================
def main():
    parser = argparse.ArgumentParser(description="BiLSTM + Attention Training Script")
    parser.add_argument("--train_csv", type=str, default="train_clean.csv")
    parser.add_argument("--test_csv", type=str, default="test_clean.csv")
    parser.add_argument("--output_dir", type=str, default="models/bilstm")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    # ---------- 1. LOAD DATA ----------
    print("[LOAD] reading CSVs...")
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    train_df = train_df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    test_df = test_df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)

    print("Train shape:", train_df.shape)
    print("Test shape :", test_df.shape)
    print("Train label distribution:\n", train_df[LABEL_COL].value_counts())

    # ---------- 2. LABEL ENCODING ----------
    le = LabelEncoder()
    y_all = le.fit_transform(train_df[LABEL_COL])
    y_test = le.transform(test_df[LABEL_COL])

    with open(os.path.join(args.output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    print("[INFO] Label classes:", list(le.classes_))

    # ---------- 3. TRAIN / VAL SPLIT ----------
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        train_df[TEXT_COL].astype(str),
        y_all,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    X_test_text = test_df[TEXT_COL].astype(str)

    # ---------- 4. TOKENIZER ----------
    print("[STEP] Fitting tokenizer on training text...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)

    def encode(texts):
        return pad_sequences(
            tokenizer.texts_to_sequences(texts),
            maxlen=MAX_LEN,
            padding="post",
            truncating="post",
        )

    X_train = encode(X_train_text)
    X_val = encode(X_val_text)
    X_test = encode(X_test_text)

    num_classes = len(le.classes_)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # ---------- 5. CLASS WEIGHTS ----------
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_all),
        y=y_all,
    )
    class_weights = dict(enumerate(class_weights_arr))

    # extra upweight figurative / sarcasm / irony
    classes = list(le.classes_)
    print("[INFO] Base class weights:", class_weights)
    if "figurative" in classes:
        fig_idx = classes.index("figurative")
        class_weights[fig_idx] *= 3.0  # stronger boost
    if "sarcasm" in classes:
        sar_idx = classes.index("sarcasm")
        class_weights[sar_idx] *= 1.8
    if "irony" in classes:
        ir_idx = classes.index("irony")
        class_weights[ir_idx] *= 1.5

    print("[INFO] Adjusted class weights:", class_weights)

    # ---------- 6. BUILD BiLSTM + ATTENTION MODEL ----------
    print("[STEP] Building BiLSTM + Attention model...")

    inputs = Input(shape=(MAX_LEN,), name="input_ids")
    x = Embedding(input_dim=MAX_VOCAB, output_dim=EMBEDDING_DIM, input_length=MAX_LEN)(inputs)

    # Simpler BiLSTM stack to avoid overfitting
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    # Bahdanau Attention
    x = BahdanauAttention(64)(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    model.summary()

    # ---------- 7. TRAIN ----------
    checkpoint_path = os.path.join(args.output_dir, "best_lstm_model.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ]

    print("[STEP] Training...")
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=7,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ---------- 8. EVALUATE ----------
    print("\n[STEP] Evaluating on validation set...")
    y_val_pred = np.argmax(model.predict(X_val), axis=1)
    print("Val Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Val Macro F1:", f1_score(y_val, y_val_pred, average="macro"))
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=le.classes_))

    save_confusion_matrix(
        y_val,
        y_val_pred,
        class_names=le.classes_,
        out_path=os.path.join(args.output_dir, "confusion_val_lstm.png"),
        title="BiLSTM+Attention Confusion Matrix (Val)",
    )

    print("\n[STEP] Evaluating on TEST set...")
    y_test_pred = np.argmax(model.predict(X_test), axis=1)
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Test Macro F1:", f1_score(y_test, y_test_pred, average="macro"))
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))

    save_confusion_matrix(
        y_test,
        y_test_pred,
        class_names=le.classes_,
        out_path=os.path.join(args.output_dir, "confusion_test_lstm.png"),
        title="BiLSTM+Attention Confusion Matrix (Test)",
    )

    # ---------- 9. SAVE FINAL MODEL & TOKENIZER ----------
    final_model_path = os.path.join(args.output_dir, "final_lstm_model_02.h5")
    model.save(final_model_path)
    print(f"[INFO] Saved final model -> {final_model_path}")

    tokenizer_path = os.path.join(args.output_dir, "tokenizer.pkl")
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[INFO] Saved tokenizer -> {tokenizer_path}")

    print("\n[DONE] BiLSTM + Attention training complete.")


if __name__ == "__main__":
    main()
