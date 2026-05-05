"""
Text Preprocessing Pipeline for Political Social Media Data
Handles: URL removal, hashtag/mention normalization, emoji handling,
         stopword removal, tokenization, lemmatization, and political keyword normalization.
"""

import re
import os
import string
import logging
import unicodedata
from typing import Optional

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

try:
    import emoji
except ImportError:
    emoji = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Ensure NLTK data ────────────────────────────────────────────────────────
def download_nltk_data():
    resources = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger",
                 "punkt_tab", "omw-1.4"]
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

download_nltk_data()

# ─── Political Keyword Normalization Map ──────────────────────────────────────
POLITICAL_NORM_MAP = {
    # US Politicians
    r"\bjoe biden\b|\bpresident biden\b|\bjbiden\b": "Biden",
    r"\bdonald trump\b|\bpresident trump\b|\btrump45\b|\bdtrump\b": "Trump",
    r"\bkamala harris\b|\bvp harris\b": "Harris",
    r"\bron desantis\b|\bdesantis\b": "DeSantis",
    r"\bbarack obama\b|\bobama\b": "Obama",
    # Indian Politicians
    r"\bnarendra modi\b|\bpm modi\b|\bnamo\b": "Modi",
    r"\brahul gandhi\b|\bpappu\b": "RahulGandhi",
    r"\baravind kejriwal\b|\bkejriwal\b": "Kejriwal",
    r"\byogi adityanath\b|\byogi\b": "Yogi",
    r"\bmamata banerjee\b|\bdidi\b": "Mamata",
    # UK/Global
    r"\bboris johnson\b": "Johnson",
    r"\brishi sunak\b|\bsunak\b": "Sunak",
    r"\bemmanuel macron\b|\bmacron\b": "Macron",
    r"\bjustin trudeau\b|\btrudeau\b": "Trudeau",
    # Parties
    r"\bdemocratic party\b|\bdemocrats\b|\bdem\b": "Democrats",
    r"\brepublican party\b|\brepublicans\b|\bgop\b": "Republicans",
    r"\bbharatiya janata party\b|\bbjp\b": "BJP",
    r"\bindian national congress\b|\bcongress party\b": "Congress",
    r"\baam aadmi party\b|\baap\b": "AAP",
    r"\blabour party\b|\blabour\b": "LabourParty",
    r"\bconservative party\b|\btories\b|\btory\b": "ConservativeParty",
}


class TextPreprocessor:
    def __init__(self, language: str = "english"):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        # Keep political/sentiment-meaningful words
        self.keep_words = {
            "not", "no", "never", "nor", "won", "isn", "aren",
            "wasn", "weren", "hasn", "haven", "hadn", "doesn",
            "didn", "don", "cannot", "against", "but", "however"
        }
        self.stop_words -= self.keep_words

    def handle_emojis(self, text: str) -> str:
        """Convert emojis to text tokens to preserve sentiment signal."""
        if emoji:
            return emoji.demojize(text, delimiters=(" ", " "))
        return text

    def robust_clean(self, text: str) -> str:
        """Removes URLs, mentions, hashtags, and standardizes spacing."""
        # 1. URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # 2. Mentions
        text = re.sub(r"@\w+", " ", text)
        # 3. Hashtags (remove entirely to avoid leakage)
        text = re.sub(r"#\w+", " ", text)
        # 4. HTML
        text = re.sub(r"<[^>]+>", " ", text)
        # 5. Non-printable chars
        text = "".join(c for c in text if unicodedata.category(c) not in ("So", "Cs", "Co", "Cn"))
        # 6. Punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # 7. Spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    def normalize_political_keywords(self, text: str) -> str:
        for pattern, replacement in POLITICAL_NORM_MAP.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def preprocess(self, text: str, return_tokens: bool = False) -> str | list:
        if not isinstance(text, str) or not text.strip():
            return [] if return_tokens else ""

        # 1. Emoji handle
        text = self.handle_emojis(text)
        # 2. Robust clean
        text = self.robust_clean(text)
        # 3. Political normalization
        text = self.normalize_political_keywords(text)

        # 4. Tokenize
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
            
        # 5. Stopwords & Length
        tokens = [t for t in tokens if t.lower() not in self.stop_words and len(t) > 1]
        
        # 6. Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        if return_tokens:
            return tokens
        return " ".join(tokens)

    def preprocess_batch(self, texts: pd.Series, return_tokens: bool = False) -> list:
        results = []
        for text in tqdm(texts, desc="Preprocessing texts"):
            results.append(self.preprocess(text, return_tokens=return_tokens))
        return results


def run_preprocessing(input_path: str = "data/political_social_media.csv",
                      output_path: str = "data/preprocessed.csv") -> pd.DataFrame:
    logger.info(f"Loading data from {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return pd.DataFrame()
        
    df = pd.read_csv(input_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    preprocessor = TextPreprocessor()

    logger.info("Preprocessing text...")
    df["cleaned_text"]  = preprocessor.preprocess_batch(df["text"])
    df["tokens"]        = preprocessor.preprocess_batch(df["text"], return_tokens=True)
    df["token_count"]   = df["tokens"].apply(len)
    df["char_count"]    = df["cleaned_text"].apply(len)

    # Drop empty rows after preprocessing
    initial_len = len(df)
    df = df[df["cleaned_text"].str.strip().str.len() > 5].reset_index(drop=True)
    logger.info(f"Dropped {initial_len - len(df)} empty/short rows after preprocessing")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Preprocessed dataset saved to {output_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    df = run_preprocessing()
    if not df.empty:
        print(df[["text", "cleaned_text", "token_count"]].head(5).to_string())
