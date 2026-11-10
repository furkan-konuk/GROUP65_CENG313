"""Generic helpers: hand-crafted text features, seeding, submission writing."""
from __future__ import annotations

import random
import re
import string
import unicodedata

import numpy as np
import pandas as pd

ASCII_LETTERS = set(string.ascii_letters)
WORD_RE = re.compile(r"[A-Za-z']+")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def latin_char_ratio(text: str) -> float:
    """Share of alphabetic characters that are plain Latin (a-z, A-Z).

    This is the organizers' character-level baseline idea: corrupted LLM outputs
    often contain letters from other scripts (Cyrillic, CJK, accented words...).
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in ASCII_LETTERS for c in letters) / len(letters)


def text_features(text: str) -> dict:
    """Simple, interpretable statistics of one text."""
    words = text.split()
    n_chars = len(text)
    n_words = len(words)
    alpha_words = WORD_RE.findall(text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    non_ascii = sum(ord(c) > 127 for c in text)
    return {
        "n_chars": n_chars,
        "n_words": n_words,
        "n_sentences": len(sentences),
        "avg_word_len": np.mean([len(w) for w in alpha_words]) if alpha_words else 0.0,
        "avg_sent_len": n_words / max(len(sentences), 1),
        "type_token_ratio": len({w.lower() for w in alpha_words}) / max(len(alpha_words), 1),
        "latin_ratio": latin_char_ratio(text),
        "non_ascii_ratio": non_ascii / max(n_chars, 1),
        "digit_ratio": sum(c.isdigit() for c in text) / max(n_chars, 1),
        "punct_ratio": sum(c in string.punctuation for c in text) / max(n_chars, 1),
        "upper_ratio": sum(c.isupper() for c in text) / max(n_chars, 1),
        "exclam_count": text.count("!"),
        "newline_count": text.count("\n"),
        "symbol_ratio": sum(unicodedata.category(c).startswith("S") for c in text) / max(n_chars, 1),
    }


def features_frame(texts) -> pd.DataFrame:
    return pd.DataFrame([text_features(t) for t in texts])


def save_submission(ids, real_text_id, path: str = "submission.csv") -> pd.DataFrame:
    """Write a Kaggle submission file with columns: id, real_text_id (1 or 2)."""
    sub = pd.DataFrame({"id": ids, "real_text_id": np.asarray(real_text_id, dtype=int)})
    sub.to_csv(path, index=False)
    return sub
