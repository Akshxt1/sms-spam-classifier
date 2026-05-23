import re
import nltk
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Smart preprocessing for spam/scam detection
    """

    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # normalize urls
    text = re.sub(r"http\S+|www\S+", " URL ", text)

    # normalize emails
    text = re.sub(r"\S+@\S+", " EMAIL ", text)

    # normalize phone numbers
    text = re.sub(r"\b\d{10,15}\b", " PHONE ", text)

    # normalize money values
    text = re.sub(r"[$₹€£]\s?\d+", " MONEY ", text)

    # normalize standalone numbers
    text = re.sub(r"\d+", " NUMBER ", text)

    # keep only text + placeholders
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    tokens = text.split()

    # keep meaningful tokens
    cleaned_tokens = [
        token
        for token in tokens
        if token not in stop_words and len(token) > 1
    ]

    return " ".join(cleaned_tokens)