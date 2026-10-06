from pathlib import Path
import joblib

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models" / "nlp"

vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
depression_model = joblib.load(MODEL_DIR / "depression_model.joblib")
mental_model = joblib.load(MODEL_DIR / "mental_health_model.joblib")
emotion_model = joblib.load(MODEL_DIR / "emotion_model.joblib")

def analyze_text(text):
    text = str(text).strip()
    if not text:
        raise ValueError("Text cannot be empty.")
    X = vectorizer.transform([text])

    dp = depression_model.predict_proba(X)[0]
    dc = depression_model.classes_
    mp = mental_model.predict_proba(X)[0]
    mc = mental_model.classes_
    ep = emotion_model.predict_proba(X)[0]
    ec = emotion_model.classes_

    return {
        "depression_prediction": int(depression_model.predict(X)[0]),
        "depression_probabilities": {str(k): float(v) for k, v in zip(dc, dp)},
        "mental_health_prediction": str(mental_model.predict(X)[0]),
        "mental_health_probabilities": {str(k): float(v) for k, v in zip(mc, mp)},
        "emotion_prediction": str(emotion_model.predict(X)[0]),
        "emotion_probabilities": {str(k): float(v) for k, v in zip(ec, ep)},
    }

if __name__ == "__main__":
    print("=" * 70)
    print("MindScreen AI - Unified NLP Engine")
    print("=" * 70)
    text = input("\nEnter text:\n> ")
    result = analyze_text(text)
    print("\nDepression:", result["depression_prediction"])
    print("Mental-health:", result["mental_health_prediction"])
    print("Emotion:", result["emotion_prediction"])
    print("\nMental-health probabilities:")
    for k, v in sorted(result["mental_health_probabilities"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v*100:.2f}%")
    print("\nEmotion probabilities:")
    for k, v in sorted(result["emotion_probabilities"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v*100:.2f}%")
    print("\nResearch/model output only; not a medical diagnosis.")
