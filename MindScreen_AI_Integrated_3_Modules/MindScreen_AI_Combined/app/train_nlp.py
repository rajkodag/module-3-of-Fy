from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE = Path(__file__).resolve().parent.parent
DATA_FILE = BASE / "data" / "MindScreen_NLP_Dataset.csv"
MODEL_DIR = BASE / "models" / "nlp"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)
df["text"] = df["text"].fillna("").astype(str)

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    strip_accents="unicode"
)
X_all = vectorizer.fit_transform(df["text"])
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")

def train_head(frame, label_col, filename, title):
    frame = frame.dropna(subset=[label_col]).copy()
    X = vectorizer.transform(frame["text"])
    y = frame[label_col]
    if label_col == "depression_label":
        y = y.astype(int)
    if y.nunique() < 2:
        raise ValueError(f"{title}: fewer than two classes.")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    print(f"\n{title}: {acc*100:.2f}% accuracy")
    print(classification_report(yte, pred, zero_division=0))
    joblib.dump(model, MODEL_DIR / filename)

train_head(df, "depression_label", "depression_model.joblib", "Depression")
train_head(df, "mental_health_label", "mental_health_model.joblib", "Mental Health")
train_head(df, "emotion_label", "emotion_model.joblib", "Emotion")

print("\nNLP ENGINE TRAINING COMPLETE.")
