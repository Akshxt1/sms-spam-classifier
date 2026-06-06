import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

for resource in ["stopwords", "punkt"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

_stop_words = set(stopwords.words("english"))
_stemmer    = PorterStemmer()

SPAM_KEYWORDS = {
    "free", "win", "won", "prize", "cash", "claim", "urgent", "reward",
    "offer", "credit", "loan", "deal", "save", "cheap", "bonus", "gift",
    "click", "now", "call", "reply", "stop", "cancel", "verify", "alert",
    "congrat", "selected", "chosen", "limited", "expire", "guaranteed",
}

_safe_stops = _stop_words - SPAM_KEYWORDS


def extract_signals(text: str) -> dict:
    if not isinstance(text, str):
        return {"has_url": False, "has_phone": False, "has_currency": False,
                "has_email": False, "caps_ratio": 0.0, "exclamation_count": 0,
                "char_count": 0, "word_count": 0}
    letters = [c for c in text if c.isalpha()]
    caps    = sum(1 for c in letters if c.isupper())
    return {
        "has_url":          bool(re.search(r"https?://|www\.", text, re.I)),
        "has_phone":        bool(re.search(r"\b(\+?\d[\d\s\-]{8,14}\d)\b", text)),
        "has_currency":     bool(re.search(r"[$₹€£¥]|\b(rs|usd|gbp)\b", text, re.I)),
        "has_email":        bool(re.search(r"\S+@\S+\.\S+", text)),
        "caps_ratio":       round(caps / max(len(letters), 1), 3),
        "exclamation_count": text.count("!"),
        "char_count":       len(text),
        "word_count":       len(text.split()),
    }


def clean_text(text: str, stem: bool = True) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+",             " URLTOKEN ",   text)
    text = re.sub(r"\S+@\S+\.\S+",                      " EMAILTOKEN ", text)
    text = re.sub(r"\b(\+?\d[\d\s\-]{8,14}\d)\b",       " PHONETOKEN ", text)
    text = re.sub(r"[$₹€£¥]\s?\d+[\d,]*|\b\d+[\d,]*\s*(?:rs|usd|gbp|inr)\b",
                  " MONEYTOKEN ", text, flags=re.I)
    text = re.sub(r"\d+", " NUMTOKEN ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens  = text.split()
    cleaned = []
    for tok in tokens:
        if tok.upper().endswith("TOKEN"):
            cleaned.append(tok.upper())
            continue
        if tok in _safe_stops or len(tok) <= 1:
            continue
        cleaned.append(_stemmer.stem(tok) if stem else tok)
    return " ".join(cleaned)