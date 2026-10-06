from pathlib import Path
import sys
import subprocess
import importlib.util

import pandas as pd
import joblib

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MODELS = BASE / "models"

D2X_DATASET = DATA / "D2X_Dataset.csv"
D2X_MODEL = MODELS / "d2x" / "d2x_model.joblib"
NLP_MODEL_DIR = MODELS / "nlp"

D2X_FEATURES = [
    "PosTweets", "NegTweets", "DepressTweets",
    "PosRetweets", "NegRetweets", "DepressRetweets",
    "PosHashtags", "NegHashtags", "DepressHashtags",
    "TextScore", "EmojiScore",
    "PositiveEmoji", "NegativeEmoji",
    "NoPersonImage", "OnePersonImage", "ManyPersonImage",
    "DullImage", "ColorFullImage"
]

def header(title):
    print("\n" + "=" * 75)
    print(f"{title:^75}")
    print("=" * 75)

def phq9_module():
    from app.phq9 import QUESTIONS, ask_question, calculate_score
    header("PHQ-9 DEPRESSION SCREENING")
    print("Answer each question for the past 2 weeks.")
    print("0 = Not at all | 1 = Several days | 2 = More than half the days | 3 = Nearly every day")
    answers = []
    for number, question in enumerate(QUESTIONS, 1):
        answers.append(ask_question(number, question))
    score, severity = calculate_score(answers)
    print("\n" + "-" * 75)
    print(f"PHQ-9 Score : {score}/27")
    print(f"Severity    : {severity}")
    if answers[8] > 0:
        print("\nIMPORTANT: Item 9 was answered above 0.")
        print("This requires appropriate professional/safety follow-up.")
    print("PHQ-9 is a screening tool, not a standalone diagnosis.")

def train_d2x():
    from app.twitter_module import train
    header("D2X MODEL TRAINING")
    train(D2X_DATASET, MODELS / "d2x")

def d2x_predict():
    from app.twitter_module import predict
    if not D2X_MODEL.exists():
        print("\nD2X model not found. Train it first.")
        return
    header("D2X / TWITTER ANALYSIS")
    features = []
    for feature in D2X_FEATURES:
        while True:
            try:
                v = float(input(f"{feature}: ").strip())
                if v < 0: raise ValueError
                features.append(v)
                break
            except ValueError:
                print("Enter a valid non-negative number.")
    result = predict(features, D2X_MODEL)
    print("\nDepression Level :", result["level"])
    print(f"Confidence       : {result['confidence']:.2f}%")
    print("\nClass probabilities:")
    for level, p in result["probabilities"].items():
        print(f"  {level}: {p:.2f}%")

def train_nlp():
    header("NLP ENGINE TRAINING")
    prep = BASE / "app" / "prepare_dataset.py"
    trainer = BASE / "app" / "train_nlp.py"
    print("\nStep 1/2: Preparing unified NLP dataset...")
    subprocess.run([sys.executable, str(prep)], check=True)
    print("\nStep 2/2: Training NLP models...")
    subprocess.run([sys.executable, str(trainer)], check=True)

def nlp_predict():
    if not all((NLP_MODEL_DIR / f).exists() for f in [
        "tfidf_vectorizer.joblib", "depression_model.joblib",
        "mental_health_model.joblib", "emotion_model.joblib"
    ]):
        print("\nNLP models are not trained.")
        print("Use option 7 first.")
        return
    from app.predict_nlp import analyze_text
    header("NLP ENGINE")
    text = input("\nEnter text/journal/comment:\n> ").strip()
    if not text:
        print("Text cannot be empty.")
        return
    r = analyze_text(text)
    print("\nDepression prediction :", r["depression_prediction"])
    print("\nDepression probabilities:")
    for k, v in r["depression_probabilities"].items():
        print(f"  {k}: {v*100:.2f}%")
    print("\nMental-health context :", r["mental_health_prediction"])
    print("\nMental-health probabilities:")
    for k, v in sorted(r["mental_health_probabilities"].items(), key=lambda x:x[1], reverse=True):
        print(f"  {k}: {v*100:.2f}%")
    print("\nEmotion prediction    :", r["emotion_prediction"])
    print("\nEmotion probabilities:")
    for k, v in sorted(r["emotion_probabilities"].items(), key=lambda x:x[1], reverse=True):
        print(f"  {k}: {v*100:.2f}%")

def combined_screening():
    header("MINDSCREEN AI - COMBINED SCREENING")
    print("This runs the three modules independently and displays their outputs.")
    print("It does NOT invent a combined medical diagnosis or arbitrary weighted score.")

    # PHQ-9
    from app.phq9 import QUESTIONS, ask_question, calculate_score
    print("\n[1/3] PHQ-9")
    answers = []
    for number, question in enumerate(QUESTIONS, 1):
        answers.append(ask_question(number, question))
    phq_score, phq_severity = calculate_score(answers)

    # NLP
    nlp_result = None
    if all((NLP_MODEL_DIR / f).exists() for f in [
        "tfidf_vectorizer.joblib", "depression_model.joblib",
        "mental_health_model.joblib", "emotion_model.joblib"
    ]):
        from app.predict_nlp import analyze_text
        print("\n[2/3] NLP")
        text = input("Enter text/journal/comment: ").strip()
        if text:
            nlp_result = analyze_text(text)
    else:
        print("\n[2/3] NLP: models not trained; skipped.")

    # D2X
    d2x_result = None
    if D2X_MODEL.exists():
        from app.twitter_module import predict
        print("\n[3/3] D2X")
        print("Enter D2X/Twitter features:")
        features = []
        for feature in D2X_FEATURES:
            while True:
                try:
                    v = float(input(f"{feature}: ").strip())
                    if v < 0: raise ValueError
                    features.append(v)
                    break
                except ValueError:
                    print("Enter a valid non-negative number.")
        d2x_result = predict(features, D2X_MODEL)
    else:
        print("\n[3/3] D2X: model not found; skipped.")

    header("COMBINED SCREENING SUMMARY")
    print(f"PHQ-9       : {phq_score}/27 ({phq_severity})")
    if nlp_result:
        print(f"NLP         : depression={nlp_result['depression_prediction']}, "
              f"mental-health={nlp_result['mental_health_prediction']}, "
              f"emotion={nlp_result['emotion_prediction']}")
    else:
        print("NLP         : Not available")
    if d2x_result:
        print(f"D2X         : {d2x_result['level']} ({d2x_result['confidence']:.2f}% confidence)")
    else:
        print("D2X         : Not available")
    print("\nThese are separate screening/model outputs, not a medical diagnosis.")

def d2x_evaluation():
    if not D2X_MODEL.exists():
        print("\nD2X model not found. Train it first.")
        return
    package = joblib.load(D2X_MODEL)
    header("D2X MODEL EVALUATION")
    for label, key in [
        ("Accuracy","accuracy"), ("Precision","precision"),
        ("Recall","recall"), ("Macro F1","macro_f1")
    ]:
        if key in package:
            print(f"{label:<12}: {package[key]*100:.2f}%")
    print("\nClasses:")
    for c in package.get("classes", []):
        print(" ", c)

def dataset_info():
    header("MINDSCREEN DATASETS")
    if D2X_DATASET.exists():
        df = pd.read_csv(D2X_DATASET)
        print(f"\nD2X records: {len(df)}")
        print(df["DepressionLevel"].value_counts().sort_index())
    unified = DATA / "MindScreen_NLP_Dataset.csv"
    if unified.exists():
        ndf = pd.read_csv(unified)
        print(f"\nUnified NLP records: {len(ndf)}")
        print(ndf["source"].value_counts())
    else:
        print("\nUnified NLP dataset: not prepared yet.")

def main():
    while True:
        header("MINDSCREEN AI - INTEGRATED SYSTEM")
        print("1. PHQ-9 Screening")
        print("2. NLP Engine - Analyze Text")
        print("3. D2X/Twitter - Predict")
        print("4. Combined Screening (PHQ-9 + NLP + D2X)")
        print("5. Train D2X Model")
        print("6. Evaluate D2X Model")
        print("7. Prepare + Train NLP Engine")
        print("8. Dataset Information")
        print("9. Exit")
        print("-" * 75)
        choice = input("Choose an option (1-9): ").strip()
        try:
            if choice == "1": phq9_module()
            elif choice == "2": nlp_predict()
            elif choice == "3": d2x_predict()
            elif choice == "4": combined_screening()
            elif choice == "5": train_d2x()
            elif choice == "6": d2x_evaluation()
            elif choice == "7": train_nlp()
            elif choice == "8": dataset_info()
            elif choice == "9":
                print("\nExiting MindScreen AI...")
                break
            else: print("\nInvalid option.")
        except Exception as e:
            print("\nERROR:", e)

if __name__ == "__main__":
    main()
