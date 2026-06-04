# SpamDetect

SMS spam classifier retrained on 138k real-world messages with word-level explanation, live signal detection, and a dark terminal UI.

## Features

- **138k training messages** from the Salman et al. 2022 dataset (arxiv:2210.10451), sourced from ScamWatch & Action Fraud
- **Spam anatomy** — highlights the exact tokens that triggered the classification
- **Live signal chips** — detects URLs, phone numbers, currency symbols, and ALL-CAPS before you even classify
- **Batch mode** — classify up to 200 messages at once and export as CSV
- **Model stats** — visualises top spam/ham feature weights
- **Tuned threshold** — optimised for spam F1, not just accuracy

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset and train model (takes ~5 min)
python -m data.download          # download the 138k dataset
python -m src.train              # train, tune, and save models

# 3. Launch the app
streamlit run app.py
```

## Model architecture

| Component          | Detail                                          |
|--------------------|-------------------------------------------------|
| Classifier         | LinearSVC wrapped in CalibratedClassifierCV     |
| Text features      | TF-IDF word 1–3g + char_wb 3–5g (FeatureUnion) |
| Metadata features  | char_count, word_count, caps_ratio, exclamation_count, has_url, has_phone, has_currency |
| Balancing          | SMOTE + class_weight=balanced                   |
| Threshold          | Tuned on held-out validation set (F1-spam)      |
| Dataset            | 138,813 messages (ham: 78k, spam: 60k)          |

## Project structure

```
sms-spam-classifier/
├── data/
│   ├── download.py          # Dataset download + preparation
│   └── sms_large.csv        # Downloaded dataset (git-ignored)
├── models/                  # Saved model artifacts (git-ignored)
│   ├── model.pkl
│   ├── vectorizer.pkl
│   ├── metadata_extractor.pkl
│   ├── feature_names.pkl
│   ├── coefficients.pkl
│   └── threshold.json
├── src/
│   ├── preprocess.py        # Text cleaning, stemming, signal extraction
│   ├── train.py             # Full training pipeline
│   └── predict.py           # Inference + spam anatomy
├── .streamlit/
│   └── config.toml          # Dark terminal theme
├── app.py                   # Streamlit UI
└── requirements.txt
```

## Dataset

The training data is the merged dataset from:
- **Salman et al. (2022)** — *An Empirical Analysis of SMS Scam Detection Systems* ([arxiv:2210.10451](https://arxiv.org/abs/2210.10451))
- Aggregated from ScamWatch (Australian Competition & Consumer Commission) and Action Fraud (UK)

This replaces the original UCI SMS Spam Collection (5,574 messages, 2012) with modern, real-world smishing patterns including OTP fraud, delivery scams, and financial phishing.
