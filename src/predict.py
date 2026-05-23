import joblib
from src.preprocess import clean_text

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


def predict_sms(message: str):
    cleaned = clean_text(message)

    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]

    probabilities = model.predict_proba(vectorized)[0]

    spam_confidence = float(probabilities[1])

    if prediction == 1:
        return "SPAM", spam_confidence
    else:
        return "HAM", 1 - spam_confidence