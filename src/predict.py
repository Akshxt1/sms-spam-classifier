import os
import json
import joblib
import numpy as np
from scipy.sparse import hstack

# Relative imports — avoids the /mount/src/ path conflict on Streamlit Cloud
from .preprocess import clean_text, extract_signals
from .features import SoftEnsemble  # noqa: F401 — needed for joblib unpickling

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

MODEL_INFO = {
    "ensemble": {
        "label":   "Ensemble",
        "tagline": "Recommended — best F1 + recall",
    },
    "linear_svc": {
        "label":   "Linear SVC",
        "tagline": "Best accuracy, high precision",
    },
    "logistic_reg": {
        "label":   "Logistic Reg.",
        "tagline": "Well-calibrated probabilities",
    },
    "naive_bayes": {
        "label":   "Naive Bayes",
        "tagline": "Fastest, highest spam recall",
    },
}

_cache: dict = {}


def _load(model_id: str):
    if model_id in _cache:
        return _cache[model_id]

    pkl       = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    meta_path = os.path.join(MODELS_DIR, f"{model_id}_meta.json")

    if not os.path.exists(pkl):
        raise FileNotFoundError(
            f"Model not found: {pkl}\nRun: python -m src.train_all"
        )

    model = joblib.load(pkl)
    with open(meta_path) as f:
        meta = json.load(f)

    if "_shared" not in _cache:
        _cache["_shared"] = {
            "vectorizer":     joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl")),
            "meta_extractor": joblib.load(os.path.join(MODELS_DIR, "metadata_extractor.pkl")),
            "feature_names":  _try_load(os.path.join(MODELS_DIR, "feature_names.pkl")),
            "coefficients":   _try_load(os.path.join(MODELS_DIR, "coefficients.pkl")),
        }

    _cache[model_id] = {"model": model, "meta": meta}
    return _cache[model_id]


def _try_load(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


def predict_sms(message: str, model_id: str = "ensemble") -> dict:
    entry     = _load(model_id)
    shared    = _cache["_shared"]
    model     = entry["model"]
    threshold = entry["meta"]["threshold"]

    cleaned   = clean_text(message)
    vec_tfidf = shared["vectorizer"].transform([cleaned])
    vec_meta  = shared["meta_extractor"].transform([message])
    vec       = hstack([vec_tfidf, vec_meta])

    spam_prob  = float(model.predict_proba(vec)[0, 1])
    label      = "SPAM" if spam_prob >= threshold else "HAM"
    confidence = spam_prob if label == "SPAM" else 1.0 - spam_prob

    return {
        "label":      label,
        "confidence": round(confidence, 4),
        "spam_prob":  round(spam_prob, 4),
        "threshold":  round(threshold, 4),
        "model_id":   model_id,
        "signals":    extract_signals(message),
        "anatomy":    _anatomy(vec, shared),
        "metrics":    entry["meta"],
    }


def predict_all(message: str) -> dict:
    return {mid: predict_sms(message, mid) for mid in MODEL_INFO}


def get_all_metrics() -> dict:
    out = {}
    for mid in MODEL_INFO:
        path = os.path.join(MODELS_DIR, f"{mid}_meta.json")
        if os.path.exists(path):
            with open(path) as f:
                out[mid] = json.load(f)
    return out


def models_loaded() -> bool:
    try:
        _load("ensemble")
        return True
    except FileNotFoundError:
        return False


def _anatomy(vec, shared, top_n: int = 8) -> list:
    names = shared["feature_names"]
    coefs = shared["coefficients"]
    if names is None or coefs is None:
        return []
    arr      = vec.toarray()[0]
    contribs = arr * coefs
    anatomy  = []
    for idx in np.argsort(contribs)[::-1]:
        name   = str(names[idx])
        weight = float(contribs[idx])
        if weight <= 0:
            break
        if name.startswith("word_tfidf__"):
            tok = name.replace("word_tfidf__", "")
            if len(tok) > 1:
                anatomy.append({"token": tok, "weight": round(weight, 4)})
        elif name.startswith("META:") and weight > 0.02:
            anatomy.append({"token": f'[{name.replace("META:", "").replace("_", " ")}]',
                            "weight": round(weight, 4)})
        if len(anatomy) >= top_n:
            break
    return anatomy