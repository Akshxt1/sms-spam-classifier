import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.preprocess import clean_text


def load_data():
    df = pd.read_csv("data/clean_spam.csv")

    df["processed_message"] = df["message"].apply(clean_text)

    df["label_encoded"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    return df


def train_model():
    df = load_data()

    X = df["processed_message"]
    y = df["label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Hybrid vectorizer
    vectorizer = FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                max_features=8000,
                min_df=2,
            )
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                max_features=5000,
            )
        ),
    ])

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    base_model = LinearSVC(class_weight="balanced")

    model = CalibratedClassifierCV(base_model)

    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    print("\nMODEL PERFORMANCE")
    print("=" * 50)

    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("\nSaved successfully!")
    print("models/model.pkl")
    print("models/vectorizer.pkl")


if __name__ == "__main__":
    train_model()