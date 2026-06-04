"""
Shared feature engineering — imported by both train.py and predict.py
so the MetadataExtractor class can be pickled/unpickled correctly.
"""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import extract_signals


class MetadataExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts 7 normalised numeric handcrafted features from raw message text.
    Designed to complement TF-IDF: captures structural signals that word-bag
    models miss (ALL-CAPS, URL presence, message length, etc.).
    """
    NAMES = [
        "char_count",
        "word_count",
        "caps_ratio",
        "exclamation_count",
        "has_url",
        "has_phone",
        "has_currency",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            s = extract_signals(text)
            rows.append([
                min(s["char_count"] / 200.0, 1.0),       # normalised length
                min(s["word_count"]  / 40.0,  1.0),       # normalised word count
                s["caps_ratio"],                           # fraction uppercase
                min(s["exclamation_count"] / 5.0, 1.0),   # urgency punctuation
                float(s["has_url"]),                       # binary URL flag
                float(s["has_phone"]),                     # binary phone flag
                float(s["has_currency"]),                  # binary currency flag
            ])
        return csr_matrix(np.array(rows, dtype=np.float32))