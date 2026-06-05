"""
Train all 4 models with the upgraded 12-feature metadata extractor.
Each model gets its own threshold.json tuned on the val set.

Models:
  naive_bayes  — ComplementNB (fastest, highest recall)
  logistic_reg — Logistic Regression (best calibration)
  linear_svc   — LinearSVC + CalibratedClassifierCV (best accuracy)
  ensemble     — Soft voting avg of all three (recommended: best F1 + recall)
"""
import os, sys, json
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              f1_score, precision_recall_curve,
                              precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import clean_text, extract_signals
from src.features import MetadataExtractor, SoftEnsemble



MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_data():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "sms_large.csv")
    df = pd.read_csv(path, low_memory=False)
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df = df.rename(columns={"text": "message"})
    df = df[["label", "message"]].dropna().reset_index(drop=True)
    df = df[df["message"].str.strip().ne("")]
    return df


def build_vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 3),
            max_features=20_000, min_df=2,
            sublinear_tf=True, strip_accents="unicode", max_df=0.95,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            max_features=8_000, min_df=2,
            sublinear_tf=True,
        )),
    ])


def tune_threshold(model, Xval, yval):
    probs = model.predict_proba(Xval)[:, 1]
    prec, rec, thresh = precision_recall_curve(yval, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best = int(np.argmax(f1s[:-1]))
    return float(thresh[best]), float(f1s[best])


def train():
    print("Loading data …")
    df = load_data()
    y = (df["label"] == "spam").astype(int).values
    df["proc"] = df["message"].apply(clean_text)
    print(f"  {len(df):,} messages  ham={int((y==0).sum()):,}  spam={int((y==1).sum()):,}")

    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.20, random_state=42, stratify=y)
    val, tst = train_test_split(tmp, test_size=0.50, random_state=42, stratify=y[tmp])

    print("Vectorising …")
    tfidf = build_vectorizer()
    meta  = MetadataExtractor()

    Xtr_t = tfidf.fit_transform(df["proc"].values[tr])
    Xtr_m = meta.fit_transform(df["message"].values[tr])
    Xtr   = hstack([Xtr_t, Xtr_m])

    def _X(split_idx):
        return hstack([tfidf.transform(df["proc"].values[split_idx]),
                       meta.transform(df["message"].values[split_idx])])
    Xval, Xtst = _X(val), _X(tst)
    ytr, yval, ytst = y[tr], y[val], y[tst]

    print("Training models …\n")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 1. Naive Bayes ────────────────────────────────────────────────────────
    print("  [1/4] Naive Bayes …")
    nb = ComplementNB(alpha=0.1)
    nb.fit(Xtr, ytr)
    nb_t, _ = tune_threshold(nb, Xval, yval)
    _save_model("naive_bayes", nb, nb_t, Xtst, ytst)

    # ── 2. Logistic Regression ────────────────────────────────────────────────
    print("  [2/4] Logistic Regression …")
    lr = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", solver="saga")
    lr.fit(Xtr, ytr)
    lr_t, _ = tune_threshold(lr, Xval, yval)
    _save_model("logistic_reg", lr, lr_t, Xtst, ytst)

    # ── 3. Linear SVC ─────────────────────────────────────────────────────────
    print("  [3/4] Linear SVC …")
    svc = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", max_iter=3000), cv=5)
    svc.fit(Xtr, ytr)
    svc_t, _ = tune_threshold(svc, Xval, yval)
    _save_model("linear_svc", svc, svc_t, Xtst, ytst)

    # ── Ensemble coefficients (from SVC folds for anatomy) ───────────────────
    coefs = np.mean([c.estimator.coef_[0] for c in svc.calibrated_classifiers_], axis=0)
    tfidf_names = np.array(tfidf.get_feature_names_out())
    meta_names  = np.array([f"META:{n}" for n in MetadataExtractor.NAMES])
    all_names   = np.concatenate([tfidf_names, meta_names])
    joblib.dump(all_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
    joblib.dump(coefs,     os.path.join(MODELS_DIR, "coefficients.pkl"))

    # ── 4. Soft Ensemble ──────────────────────────────────────────────────────
    print("  [4/4] Soft Ensemble …")
    ens = SoftEnsemble([nb, lr, svc])
    ens_t, _ = tune_threshold(ens, Xval, yval)
    _save_model("ensemble", ens, ens_t, Xtst, ytst)

    # ── Save shared artefacts ─────────────────────────────────────────────────
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "vectorizer.pkl"))
    joblib.dump(meta,  os.path.join(MODELS_DIR, "metadata_extractor.pkl"))

    print("\n✓ All models saved to models/")


def _save_model(name, model, threshold, Xtst, ytst):
    probs = model.predict_proba(Xtst)[:, 1]
    yp    = (probs >= threshold).astype(int)
    acc   = accuracy_score(ytst, yp)
    f1    = f1_score(ytst, yp)
    prec  = precision_score(ytst, yp)
    rec   = recall_score(ytst, yp)

    print(f"       acc={acc:.3f}  f1={f1:.3f}  prec={prec:.3f}  rec={rec:.3f}  thresh={threshold:.3f}")

    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))
    with open(os.path.join(MODELS_DIR, f"{name}_meta.json"), "w") as f:
        json.dump(dict(threshold=threshold, accuracy=round(acc,4),
                       f1=round(f1,4), precision=round(prec,4), recall=round(rec,4)), f)


if __name__ == "__main__":
    train()