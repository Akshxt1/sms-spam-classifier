import re
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import extract_signals

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
        s  = extract_signals(text)
        words = text.split()
        n_words = max(len(words), 1)

        digits = sum(c.isdigit() for c in text)
        digit_ratio = digits / max(len(text), 1)

        avg_wl = sum(len(w) for w in words) / n_words

        tl = text.lower()
        urgency_score = min(
            sum(1 for u in URGENCY_WORDS if u in tl) / 3.0, 1.0
        )

        twitter_signal = float(bool(re.search(r"@\w+|#\w+|\bRT\b", text)))

        url_count = len(re.findall(r"https?://\S+|www\.\S+", text, re.I))
        link_density = min(url_count / n_words, 1.0)

        return [
            min(s["char_count"] / 200.0, 1.0),
            min(s["word_count"]  / 40.0,  1.0),
            s["caps_ratio"],
            min(s["exclamation_count"] / 5.0, 1.0),
            float(s["has_url"]),
            float(s["has_phone"]),
            float(s["has_currency"]),
            digit_ratio,
            min(avg_wl / 10.0, 1.0),
            urgency_score,
            twitter_signal,
            link_density,
        ]


class SoftEnsemble:
    """Soft-voting average over constituent models. Kept here so joblib
    always finds it regardless of which module triggers the load."""
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        return sum(m.predict_proba(X) for m in self.models) / len(self.models)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.45).astype(int)