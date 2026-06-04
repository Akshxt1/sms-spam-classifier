"""
Download and prepare the large-scale SMS scam dataset.
Source: github.com/vinit9638/SMS-scam-detection-dataset
Based on: Salman et al. 2022 (arxiv:2210.10451) — 153k messages
"""
import os
import urllib.request
import pandas as pd

RAW_URL = (
    "https://raw.githubusercontent.com/vinit9638/"
    "SMS-scam-detection-dataset/main/"
    "sms_scam_detection_dataset_merged_with_lang.csv"
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_PATH = os.path.join(DATA_DIR, "sms_large.csv")


def download(force: bool = False) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(OUT_PATH) and not force:
        print(f"Dataset already exists at {OUT_PATH}")
        return OUT_PATH

    print("Downloading large-scale SMS dataset (~30 MB)...")
    urllib.request.urlretrieve(RAW_URL, OUT_PATH)
    print(f"Saved to {OUT_PATH}")
    return OUT_PATH


def load(path: str = OUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # Normalise label
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].copy()

    # Use the text column
    df = df.rename(columns={"text": "message"})
    df = df[["label", "message"]].dropna(subset=["message"])
    df = df[df["message"].str.strip().ne("")].reset_index(drop=True)

    print(f"Loaded {len(df):,} messages")
    print(df["label"].value_counts().to_string())
    return df


if __name__ == "__main__":
    path = download()
    df = load(path)