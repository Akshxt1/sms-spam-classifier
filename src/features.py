import re
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

# Relative import — works correctly on Streamlit Cloud
from .preprocess import extract_signals

URGENCY_WORDS = {
    "urgent", "immediately", "expire", "expires", "limited", "hurry",
    "act now", "last chance", "today only", "don't miss", "offer ends",
    "deadline", "final", "respond", "verify", "confirm", "suspended",
}


class MetadataExtractor(BaseEstimator, TransformerMixin):
    NAMES = [
        "char_count", "word_count", "caps_ratio", "exclamation_count",
        "has_url", "has_phone", "has_currency",
        "digit_ratio", "avg_word_length", "urgency_score",
        "twitter_signal", "link_density",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return csr_matrix(np.array([self._row(t) for t in X], dtype=np.float32))

    def _row(self, text):
        if not isinstance(text, str):
            text = ""
        s      = extract_signals(text)
        words  = text.split()
        n_words = max(len(words), 1)
        digits  = sum(c.isdigit() for c in text)
        tl      = text.lower()
        return [
            min(s["char_count"] / 200.0, 1.0),
            min(s["word_count"]  / 40.0,  1.0),
            s["caps_ratio"],
            min(s["exclamation_count"] / 5.0, 1.0),
            float(s["has_url"]),
            float(s["has_phone"]),
            float(s["has_currency"]),
            digits / max(len(text), 1),
            min(sum(len(w) for w in words) / n_words / 10.0, 1.0),
            min(sum(1 for u in URGENCY_WORDS if u in tl) / 3.0, 1.0),
            float(bool(re.search(r"@\w+|#\w+|\bRT\b", text))),
            min(len(re.findall(r"https?://\S+|www\.\S+", text, re.I)) / n_words, 1.0),
        ]


class SoftEnsemble:
    """Soft-voting average. Defined here so joblib can always find it."""
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        return sum(m.predict_proba(X) for m in self.models) / len(self.models)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.45).astype(int)