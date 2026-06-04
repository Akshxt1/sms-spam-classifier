"""
Train the revamped SMS spam classifier.

Improvements over v1:
  - 138k messages (was 5k), near-balanced classes (1.3:1, was 6.9:1)
  - Metadata features: char_count, word_count, caps_ratio, has_url, has_phone,
    has_currency, exclamation_count
  - PorterStemmer preprocessing + protected spam keywords
  - min_df=2 on both TF-IDF vectorizers (was missing from char_tfidf)
  - Threshold tuned to maximise F1-spam on validation set
  - Feature names + coefficients exported for spam anatomy panel
  - Lazy model loading in predict.py (no crash on fresh clone)
"""
import os, json, sys
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import clean_text, extract_signals
from src.features import MetadataExtractor

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sms_large.csv")
    if not os.path.exists(data_path):
        print("Dataset not found — run:  python -m data.download")
        sys.exit(1)
    df = pd.read_csv(data_path, low_memory=False)
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df = df.rename(columns={"text": "message"})
    df = df[["label", "message"]].dropna(subset=["message"])
    df = df[df["message"].str.strip().ne("")].reset_index(drop=True)
    return df


# ── Vectorizer ────────────────────────────────────────────────────────────────

def build_vectorizer():
    return FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=15_000,
            min_df=2,
            sublinear_tf=True,
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=8_000,
            min_df=2,
            sublinear_tf=True,
        )),
    ])


# ── Threshold tuning ──────────────────────────────────────────────────────────

def tune_threshold(model, X_val, y_val) -> float:
    probs = model.predict_proba(X_val)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = int(np.argmax(f1s[:-1]))
    best_t   = float(thresholds[best_idx])
    print(f"  Tuned threshold: {best_t:.4f}  (F1-spam on val: {f1s[best_idx]:.4f})")
    return best_t


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    print("Loading data …")
    df = load_data()
    counts = df["label"].value_counts().to_dict()
    print(f"  {len(df):,} messages — ham: {counts['ham']:,}  spam: {counts['spam']:,}")

    print("Preprocessing text …")
    df["processed"] = df["message"].apply(clean_text)

    y          = (df["label"] == "spam").astype(int).values
    X_raw      = df["message"].values
    X_proc     = df["processed"].values

    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.20, random_state=42, stratify=y)
    val, tst = train_test_split(tmp, test_size=0.50, random_state=42, stratify=y[tmp])

    print("Vectorising …")
    tfidf     = build_vectorizer()
    Xtr_tfidf = tfidf.fit_transform(X_proc[tr])
    Xval_tfidf = tfidf.transform(X_proc[val])
    Xtst_tfidf = tfidf.transform(X_proc[tst])

    print("Extracting metadata features …")
    meta      = MetadataExtractor()
    Xtr_meta  = meta.fit_transform(X_raw[tr])
    Xval_meta = meta.transform(X_raw[val])
    Xtst_meta = meta.transform(X_raw[tst])

    Xtr  = hstack([Xtr_tfidf,  Xtr_meta])
    Xval = hstack([Xval_tfidf, Xval_meta])
    Xtst = hstack([Xtst_tfidf, Xtst_meta])

    print("Training model …")
    # C=1.0 tuned by cross-validation; class_weight='balanced' handles residual imbalance
    base_svc = LinearSVC(class_weight="balanced", C=1.0, max_iter=3000)
    model    = CalibratedClassifierCV(base_svc, cv=5, method="sigmoid")
    model.fit(Xtr, y[tr])

    print("Tuning decision threshold …")
    threshold = tune_threshold(model, Xval, y[val])

    # ── Evaluate ──────────────────────────────────────────────────────────────
    probs_tst = model.predict_proba(Xtst)[:, 1]
    y_pred    = (probs_tst >= threshold).astype(int)

    print("\n" + "=" * 55)
    print("TEST SET RESULTS")
    print("=" * 55)
    print(f"Accuracy   : {accuracy_score(y[tst], y_pred):.4f}")
    print(f"F1 (spam)  : {f1_score(y[tst], y_pred):.4f}")
    print(classification_report(y[tst], y_pred, target_names=["ham", "spam"]))
    print("Confusion matrix:")
    print(confusion_matrix(y[tst], y_pred))

    # ── Feature coefficients for spam anatomy ─────────────────────────────────
    coefs = np.mean(
        [c.estimator.coef_[0] for c in model.calibrated_classifiers_],
        axis=0,
    )
    tfidf_names = np.array(tfidf.get_feature_names_out())
    meta_names  = np.array([f"META:{n}" for n in MetadataExtractor.NAMES])
    all_names   = np.concatenate([tfidf_names, meta_names])

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model,      os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(tfidf,      os.path.join(MODELS_DIR, "vectorizer.pkl"))
    joblib.dump(meta,       os.path.join(MODELS_DIR, "metadata_extractor.pkl"))
    joblib.dump(all_names,  os.path.join(MODELS_DIR, "feature_names.pkl"))
    joblib.dump(coefs,      os.path.join(MODELS_DIR, "coefficients.pkl"))
    with open(os.path.join(MODELS_DIR, "threshold.json"), "w") as f:
        json.dump({"threshold": threshold}, f)

    print(f"\n✓ All artefacts saved to {MODELS_DIR}/")
    return model


if __name__ == "__main__":
    train()