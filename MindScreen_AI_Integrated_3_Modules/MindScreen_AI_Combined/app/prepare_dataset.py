from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = DATA / "MindScreen_NLP_Dataset.csv"

text_df = pd.read_csv(DATA / "text analysis.csv")
mental_df = pd.read_csv(DATA / "Combined Data.csv")
emotion_df = pd.read_csv(DATA / "emotions.csv")

# Normalize text-analysis dataset
if "clean_text" in text_df.columns:
    text_col = "clean_text"
elif "text" in text_df.columns:
    text_col = "text"
else:
    raise ValueError(f"Could not find text column in text analysis.csv: {list(text_df.columns)}")

if "is_depression" not in text_df.columns:
    raise ValueError("text analysis.csv must contain 'is_depression'.")

a = pd.DataFrame({
    "text": text_df[text_col].fillna("").astype(str).str.strip(),
    "source": "depression_text",
    "depression_label": pd.to_numeric(text_df["is_depression"], errors="coerce"),
    "mental_health_label": pd.NA,
    "emotion_label": pd.NA,
})

# Normalize mental-health dataset
if "statement" not in mental_df.columns or "status" not in mental_df.columns:
    raise ValueError(f"Combined Data.csv must contain statement/status. Found: {list(mental_df.columns)}")

b = pd.DataFrame({
    "text": mental_df["statement"].fillna("").astype(str).str.strip(),
    "source": "mental_health",
    "depression_label": pd.NA,
    "mental_health_label": mental_df["status"].fillna("").astype(str).str.strip(),
    "emotion_label": pd.NA,
})

# Normalize emotion dataset
if "text" not in emotion_df.columns or "label" not in emotion_df.columns:
    raise ValueError(f"emotions.csv must contain text/label. Found: {list(emotion_df.columns)}")

c = pd.DataFrame({
    "text": emotion_df["text"].fillna("").astype(str).str.strip(),
    "source": "emotion",
    "depression_label": pd.NA,
    "mental_health_label": pd.NA,
    "emotion_label": emotion_df["label"],
})

df = pd.concat([a, b, c], ignore_index=True)
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
df = df[df["text"].str.len() > 2]
df = df.drop_duplicates(subset=["text", "source"])

df.to_csv(OUT, index=False)

print("=" * 70)
print("MindScreen AI - Unified NLP Dataset")
print("=" * 70)
print(f"Total records: {len(df):,}")
print("\nRecords by source:")
print(df["source"].value_counts())
print(f"\nSaved: {OUT}")
