"""
Prediction module.
- Lazy model loading (won't crash on fresh clone)
- Tuned threshold from training
- Spam anatomy: top contributing word-level features
"""
import os
import json
import joblib
import numpy as np
from scipy.sparse import hstack

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import clean_text, extract_signals

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEFAULT_THRESHOLD = 0.45  # fallback if threshold.json missing

# Lazy singletons
_model = None
_vectorizer = None
_meta_extractor = None
_feature_names = None
_coefs = None
_threshold = None


def _load_models():
    global _model, _vectorizer, _meta_extractor, _feature_names, _coefs, _threshold
    if _model is not None:
        return

    required = ["model.pkl", "vectorizer.pkl", "metadata_extractor.pkl"]
    for f in required:
        path = os.path.join(MODELS_DIR, f)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run: python -m src.train  to train the model first."
            )

    _model           = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
    _vectorizer      = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))
    _meta_extractor  = joblib.load(os.path.join(MODELS_DIR, "metadata_extractor.pkl"))

    try:
        _feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
        _coefs         = joblib.load(os.path.join(MODELS_DIR, "coefficients.pkl"))
    except FileNotFoundError:
        _feature_names = None
        _coefs = None

    t_path = os.path.join(MODELS_DIR, "threshold.json")
    if os.path.exists(t_path):
        with open(t_path) as f:
            _threshold = json.load(f)["threshold"]
    else:
        _threshold = DEFAULT_THRESHOLD


def predict_sms(message: str) -> dict:
    """
    Returns:
        {
          "label":      "SPAM" | "HAM",
          "confidence": float (0-1),
          "threshold":  float,
          "signals":    dict from extract_signals(),
          "anatomy":    [{"token": str, "weight": float}, ...] top spam triggers,
        }
    """
    _load_models()

    cleaned = clean_text(message)
    vec_tfidf = _vectorizer.transform([cleaned])
    vec_meta  = _meta_extractor.transform([message])
    vec       = hstack([vec_tfidf, vec_meta])

    spam_prob = float(_model.predict_proba(vec)[0, 1])
    label     = "SPAM" if spam_prob >= _threshold else "HAM"
    confidence = spam_prob if label == "SPAM" else 1.0 - spam_prob

    anatomy = _get_anatomy(vec, vec_tfidf, spam_prob)

    return {
        "label":      label,
        "confidence": round(confidence, 4),
        "spam_prob":  round(spam_prob, 4),
        "threshold":  round(_threshold, 4),
        "signals":    extract_signals(message),
        "anatomy":    anatomy,
    }


def _get_anatomy(vec, vec_tfidf, spam_prob: float, top_n: int = 8) -> list:
    """Top word-level tokens that pushed toward the spam prediction."""
    if _feature_names is None or _coefs is None:
        return []

    arr = vec.toarray()[0]
    contributions = arr * _coefs

    anatomy = []
    for idx in np.argsort(contributions)[::-1]:
        name = str(_feature_names[idx])
        weight = float(contributions[idx])
        if weight <= 0:
            break
        # Only show word-level features (skip char n-grams and metadata)
        if name.startswith("word_tfidf__"):
            token = name.replace("word_tfidf__", "")
            if len(token) > 1:
                anatomy.append({"token": token, "weight": round(weight, 4)})
        elif name.startswith("META:"):
            # Only show metadata if it has meaningful contribution
            if weight > 0.02:
                label = name.replace("META:", "").replace("_", " ")
                anatomy.append({"token": f"[{label}]", "weight": round(weight, 4)})
        if len(anatomy) >= top_n:
            break

    return anatomy


def models_loaded() -> bool:
    try:
        _load_models()
        return True
    except FileNotFoundError:
        return False