#**********************************
#   DATS6312.11_Final_Project_FA25
#   Sarcasm & Figurative Tweets
#   PHASE 1: PREPROCESSING + CORPUS EDA
#**********************************

#%%
#==============================IMPORTS=========================================
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

try:
    import spacy
except ImportError:
    spacy = None

# NLTK downloads
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

#%%
#==============================CONFIG=========================================
"""
You MUST change the 4 paths below according to your folder structure.
Ensure train.csv and test.csv contain the text column and label column.
"""

TRAIN_CSV   = r"train.csv"
TEST_CSV    = r"test.csv"

TEXT_COL    = "tweets"      # <--- change if needed
LABEL_COL   = "class"      # <--- change if needed

OUT_TRAIN   = r"train_clean.csv"
OUT_TEST    = r"test_clean.csv"

FIG_DIR     = r"figures"

STOPWORDS = set(stopwords.words("english"))
LEMM      = WordNetLemmatizer()

# Try spaCy
if spacy is not None:
    try:
        NLP = spacy.load("en_core_web_sm")
    except:
        NLP = None
else:
    NLP = None


# Regex patterns
URL_PATTERN     = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
EMOJI_PATTERN   = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE
)
HASHTAG_PATTERN = re.compile(r"#\w+")


#%%
#==============================CLEANING========================================
def basic_clean(text: str) -> str:
    """Cleans tweets while keeping punctuation like ? and ! which indicate sarcasm."""
    if not isinstance(text, str):
        return ""

    t = text.strip()
    t = URL_PATTERN.sub(" ", t)
    t = MENTION_PATTERN.sub(" ", t)
    t = EMOJI_PATTERN.sub(" ", t)
    t = HASHTAG_PATTERN.sub(lambda m: " " + m.group(0).replace("#", " "), t)

    # Keep punctuation but remove weird symbols
    t = re.sub(r"[^a-zA-Z0-9\s\.\!\?]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def _wn(tag: str) -> str:
    tag = tag.lower()
    if tag.startswith("j"): return "a"
    if tag.startswith("v"): return "v"
    if tag.startswith("n"): return "n"
    if tag.startswith("r"): return "r"
    return "n"


def lemmatize(text: str) -> str:
    """Lemmatize cleaned tweet."""
    tokens = [w for w in word_tokenize(text) if w.isalpha() and w not in STOPWORDS]
    if NLP is not None:
        doc = NLP(" ".join(tokens))
        return " ".join([t.lemma_ for t in doc if t.lemma_.isalpha()])
    tagged = nltk.pos_tag(tokens)
    return " ".join([LEMM.lemmatize(tok, _wn(pos)) for tok, pos in tagged])


#%%
#==============================FEATURES========================================
def add_features(df, col):
    t = df[col].astype(str)
    df["char_len"]     = t.apply(len)
    df["word_len"]     = t.apply(lambda x: len(x.split()))
    df["num_exclam"]   = t.str.count("!")
    df["num_question"] = t.str.count(r"\?")
    df["num_hashtags"] = t.str.count("#")
    df["num_mentions"] = t.str.count("@")
    df["num_caps"]     = t.apply(lambda x: sum(1 for ch in x if ch.isupper()))
    df["cap_ratio"]    = df.apply(
        lambda row: row["num_caps"]/row["char_len"] if row["char_len"]>0 else 0, axis=1
    )
    return df


#%%
#==============================EDA==============================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_label_distribution(df):
    ensure_dir(FIG_DIR)
    plt.figure(figsize=(6,4))
    sns.countplot(x="label", data=df)
    plt.title("Label Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "label_distribution.png"), dpi=200)
    plt.close()


def plot_length(df):
    ensure_dir(FIG_DIR)
    plt.figure(figsize=(6,4))
    sns.histplot(df["char_len"], bins=40)
    plt.title("Tweet Length (Characters)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "char_length.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(6,4))
    sns.histplot(df["word_len"], bins=40)
    plt.title("Tweet Length (Words)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "word_length.png"), dpi=200)
    plt.close()


def plot_wordclouds(df):
    ensure_dir(FIG_DIR)
    for lab in df["label"].unique():
        subset = df[df["label"] == lab]["clean_text"].dropna()
        text = " ".join(subset)

        tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
        if len(tokens) == 0:
            continue

        wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(tokens))
        plt.figure(figsize=(8,4))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"WordCloud – {lab}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"wc_{lab}.png"), dpi=200)
        plt.close()


def summary(df):
    print("\n===== CORPUS SUMMARY =====")
    print("Rows:", len(df))
    print("Label counts:\n", df["label"].value_counts())
    print("Avg chars:", df["char_len"].mean())
    print("Avg words:", df["word_len"].mean())

    for lab in df["label"].unique():
        subset = df[df["label"] == lab]["clean_text"].tolist()
        tokens = []
        for t in subset:
            tokens.extend([w for w in word_tokenize(t) if w.isalpha() and w not in STOPWORDS])
        top = Counter(tokens).most_common(10)
        print(f"\nTop tokens for {lab}:")
        for w, c in top:
            print(f"  {w}: {c}")


#%%
#==============================PROCESS FUNCTION================================
def preprocess(df, name="train"):
    print(f"\n[STEP] Cleaning {name}...")
    df["clean_text"]  = df["raw_text"].apply(basic_clean)

    print(f"[STEP] Lemmatizing {name}...")
    df["lemma_text"] = df["clean_text"].apply(lemmatize)

    print(f"[STEP] Adding features to {name}...")
    df = add_features(df, "clean_text")
    return df


#%%
#==============================MAIN============================================
if __name__ == "__main__":
    print("[LOAD] Reading train.csv and test.csv ...")

    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)

    # Rename columns for uniform handling
    train = train[[TEXT_COL, LABEL_COL]].rename(columns={TEXT_COL: "raw_text", LABEL_COL: "label"})
    test  = test[[TEXT_COL, LABEL_COL]].rename(columns={TEXT_COL: "raw_text", LABEL_COL: "label"})

    # Drop missing rows
    train = train.dropna(subset=["raw_text", "label"]).reset_index(drop=True)
    test  = test.dropna(subset=["raw_text", "label"]).reset_index(drop=True)

    # Preprocess both
    train_clean = preprocess(train, "train")
    test_clean  = preprocess(test, "test")

    # Save cleaned CSVs
    ensure_dir("data/processed")
    train_clean.to_csv(OUT_TRAIN, index=False)
    test_clean.to_csv(OUT_TEST, index=False)

    print(f"\n[SAVED] Clean train → {OUT_TRAIN}")
    print(f"[SAVED] Clean test → {OUT_TEST}")

    # Run EDA on TRAIN ONLY
    print("\n[EDA] Running corpus EDA on train set...")
    plot_label_distribution(train_clean)
    plot_length(train_clean)
    plot_wordclouds(train_clean)
    summary(train_clean)

    print("\n[DONE] Phase 1 finished successfully.")
