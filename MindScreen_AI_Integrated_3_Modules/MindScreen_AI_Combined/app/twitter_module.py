from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# D2X FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "PosTweets",
    "NegTweets",
    "DepressTweets",
    "PosRetweets",
    "NegRetweets",
    "DepressRetweets",
    "PosHashtags",
    "NegHashtags",
    "DepressHashtags",
    "TextScore",
    "EmojiScore",
    "PositiveEmoji",
    "NegativeEmoji",
    "NoPersonImage",
    "OnePersonImage",
    "ManyPersonImage",
    "DullImage",
    "ColorFullImage"
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(dataset_path):

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{dataset_path}"
        )

    df = pd.read_csv(dataset_path)

    # Remove accidental spaces in column names
    df.columns = df.columns.str.strip()

    # --------------------------------------------------------
    # Required target column
    # --------------------------------------------------------

    if "DepressionLevel" not in df.columns:

        raise ValueError(
            "Target column 'DepressionLevel' "
            "was not found in the dataset."
        )

    # --------------------------------------------------------
    # Check all 18 features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "The following D2X features are missing:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
        )

    # --------------------------------------------------------
    # Convert features to numeric
    # --------------------------------------------------------

    for feature in FEATURE_NAMES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    # Remove records containing missing feature/label values
    before = len(df)

    df = df.dropna(
        subset=FEATURE_NAMES + ["DepressionLevel"]
    ).copy()

    removed = before - len(df)

    if removed > 0:

        print(
            f"\nRemoved {removed} incomplete records."
        )

    # --------------------------------------------------------
    # Feature matrix
    # --------------------------------------------------------

    X = df[
        FEATURE_NAMES
    ].astype(float)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = (
        df["DepressionLevel"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print("\nDataset loaded successfully.")

    print(
        f"Records : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    print(
        "\nDepression Level Distribution:"
    )

    print(
        y.value_counts()
        .sort_index()
    )

    print(
        "\nDetected classes:"
    )

    for cls in sorted(y.unique()):

        print(
            f"  {cls}"
        )

    return df, X, y


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_models():

    models = {

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            ),

        "SVM":
            Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ]),

        "Logistic Regression":
            Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ]),

        "Gradient Boosting":
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
    }

    return models


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train(dataset_path, models_path):

    models_path = Path(models_path)

    models_path.mkdir(
        parents=True,
        exist_ok=True
    )

    df, X, y = load_dataset(
        dataset_path
    )

    print("\n" + "=" * 70)
    print("                    MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    print(
        "\nRunning 5-fold stratified validation..."
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    y_pred = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y,
        y_pred
    )

    precision = precision_score(
        y,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y,
        y_pred,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Display metrics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("              5-FOLD TRAINING EVALUATION")
    print("=" * 70)

    print(
        f"\nAccuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"Macro F1  : {macro_f1 * 100:.2f}%"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y,
            y_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Train final model using ALL records
    # --------------------------------------------------------

    print(
        "\nTraining final Random Forest "
        "using complete dataset..."
    )

    model.fit(
        X,
        y
    )

    # --------------------------------------------------------
    # Save model package
    # --------------------------------------------------------

    model_file = (
        models_path /
        "d2x_model.joblib"
    )

    package = {

        "model":
            model,

        "features":
            FEATURE_NAMES,

        "classes":
            sorted(y.unique()),

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "macro_f1":
            macro_f1,

        "records":
            len(df)
    }

    joblib.dump(
        package,
        model_file
    )

    print(
        "\nModel successfully saved:"
    )

    print(
        model_file
    )

    print("\nSaved metrics:")

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1 : {macro_f1 * 100:.2f}%"
    )

    print("\n" + "=" * 70)
    print("                  TRAINING COMPLETE")
    print("=" * 70)


# ============================================================
# PREDICT NEW RECORD
# ============================================================

def predict(features, model_file):

    package = joblib.load(
        model_file
    )

    model = package["model"]

    feature_names = package["features"]

    if len(features) != len(
        feature_names
    ):

        raise ValueError(
            f"Expected {len(feature_names)} "
            f"features but received {len(features)}."
        )

    X = pd.DataFrame(
        [features],
        columns=feature_names
    )

    prediction = model.predict(
        X
    )[0]

    probabilities = (
        model.predict_proba(X)[0]
    )

    classes = model.classes_

    probability_dict = {

        str(cls):
            float(probability * 100)

        for cls, probability
        in zip(
            classes,
            probabilities
        )
    }

    confidence = max(
        probability_dict.values()
    )

    return {

        "level":
            str(prediction),

        "confidence":
            confidence,

        "probabilities":
            probability_dict
    }


# ============================================================
# TEST DATASET RECORD
# ============================================================

def test_dataset_record(
    dataset_path,
    model_file,
    record_number
):

    df, X, y = load_dataset(
        dataset_path
    )

    index = record_number - 1

    if (
        index < 0
        or
        index >= len(df)
    ):

        raise ValueError(
            f"Record must be between 1 and {len(df)}."
        )

    features = (
        X.iloc[index]
        .tolist()
    )

    actual = str(
        y.iloc[index]
    )

    result = predict(
        features,
        model_file
    )

    print("\n" + "=" * 70)

    print(
        f"              DATASET RECORD #{record_number}"
    )

    print("=" * 70)

    print(
        f"\nActual Depression Level    : "
        f"{actual}"
    )

    print(
        f"Predicted Depression Level : "
        f"{result['level']}"
    )

    print(
        f"Confidence                 : "
        f"{result['confidence']:.2f}%"
    )

    print(
        "\nClass probabilities:"
    )

    for level in [
        "level0",
        "level1",
        "level2",
        "level3"
    ]:

        probability = (
            result[
                "probabilities"
            ].get(
                level,
                0
            )
        )

        print(
            f"  {level}: "
            f"{probability:.2f}%"
        )

    if actual == result["level"]:

        print(
            "\nResult                    : CORRECT"
        )

    else:

        print(
            "\nResult                    : INCORRECT"
        )

    print("=" * 70)


# ============================================================
# TEST ONE RECORD FROM EACH LEVEL
# ============================================================

def test_all_four_levels(
    dataset_path,
    model_file
):

    df, X, y = load_dataset(
        dataset_path
    )

    print("\n" + "=" * 70)
    print("             TESTING ALL FOUR LEVELS")
    print("=" * 70)

    correct = 0
    total = 0

    for level in [
        "level0",
        "level1",
        "level2",
        "level3"
    ]:

        indexes = np.where(
            y.values == level
        )[0]

        if len(indexes) == 0:

            print(
                f"\nNo records found for {level}."
            )

            continue

        # First available record
        index = indexes[0]

        features = (
            X.iloc[index]
            .tolist()
        )

        actual = str(
            y.iloc[index]
        )

        result = predict(
            features,
            model_file
        )

        predicted = result["level"]

        print("\n" + "-" * 60)

        print(
            f"Level requested : {level}"
        )

        print(
            f"Record number   : {index + 1}"
        )

        print(
            f"Actual          : {actual}"
        )

        print(
            f"Predicted       : {predicted}"
        )

        print(
            f"Probability     : "
            f"{result['confidence']:.2f}%"
        )

        if actual == predicted:

            print(
                "Result          : CORRECT"
            )

            correct += 1

        else:

            print(
                "Result          : INCORRECT"
            )

        total += 1

    print("\n" + "=" * 70)

    print(
        f"Correct : {correct} / {total}"
    )

    print("=" * 70)


# ============================================================
# OUT-OF-FOLD VALIDATION
# ============================================================

def cross_validate_dataset(
    dataset_path
):

    df, X, y = load_dataset(
        dataset_path
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    y_pred = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict"
    )

    accuracy = accuracy_score(
        y,
        y_pred
    )

    macro_f1 = f1_score(
        y,
        y_pred,
        average="macro",
        zero_division=0
    )

    precision = precision_score(
        y,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y,
        y_pred,
        average="macro",
        zero_division=0
    )

    labels = sorted(
        y.unique()
    )

    print("\n" + "=" * 70)
    print("             OUT-OF-FOLD VALIDATION")
    print("=" * 70)

    print(
        f"\n5-Fold Accuracy  : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"5-Fold Precision : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"5-Fold Recall    : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"5-Fold Macro F1  : "
        f"{macro_f1 * 100:.2f}%"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y,
            y_pred,
            labels=labels,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y,
        y_pred,
        labels=labels
    )

    print(
        "Confusion Matrix:"
    )

    print(
        "        "
        + " ".join(
            f"{label:>8}"
            for label in labels
        )
    )

    for i, label in enumerate(labels):

        print(
            f"{label:>8}"
            + "".join(
                f"{value:>8}"
                for value in cm[i]
            )
        )

    print("=" * 70)


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_models(
    dataset_path
):

    df, X, y = load_dataset(
        dataset_path
    )

    models = get_models()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    labels = sorted(
        y.unique()
    )

    results = []

    print("\n" + "=" * 70)
    print("                  MODEL COMPARISON")
    print("=" * 70)

    for model_name, model in models.items():

        print(
            f"\nEvaluating {model_name}..."
        )

        y_pred = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict"
        )

        accuracy = accuracy_score(
            y,
            y_pred
        )

        precision = precision_score(
            y,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y,
            y_pred,
            average="macro",
            zero_division=0
        )

        macro_f1 = f1_score(
            y,
            y_pred,
            average="macro",
            zero_division=0
        )

        results.append({

            "Model":
                model_name,

            "Accuracy":
                accuracy * 100,

            "Precision":
                precision * 100,

            "Recall":
                recall * 100,

            "Macro F1":
                macro_f1 * 100
        })

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 70)
    print("             5-FOLD MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Accuracy":
                    lambda x:
                    f"{x:.2f}%",

                "Precision":
                    lambda x:
                    f"{x:.2f}%",

                "Recall":
                    lambda x:
                    f"{x:.2f}%",

                "Macro F1":
                    lambda x:
                    f"{x:.2f}%"
            }
        )
    )

    # --------------------------------------------------------
    # Best model based on Macro F1
    # --------------------------------------------------------

    best_index = (
        results_df["Macro F1"]
        .idxmax()
    )

    best_model = (
        results_df
        .loc[
            best_index,
            "Model"
        ]
    )

    print(
        "\nHighest Macro F1 model:"
    )

    print(
        best_model
    )

    # --------------------------------------------------------
    # Detailed classification report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("          DETAILED MODEL REPORTS")
    print("=" * 70)

    for model_name, model in models.items():

        y_pred = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict"
        )

        print("\n" + "-" * 70)

        print(
            model_name
        )

        print("-" * 70)

        print(
            classification_report(
                y,
                y_pred,
                labels=labels,
                zero_division=0
            )
        )

    print("\n" + "=" * 70)
    print("              MODEL COMPARISON COMPLETE")
    print("=" * 70)

    return results_df


# ============================================================
# HIGH CONFIDENCE RECORDS
# ============================================================

def find_high_confidence_examples(
    dataset_path,
    model_file,
    threshold=60
):

    df, X, y = load_dataset(
        dataset_path
    )

    package = joblib.load(
        model_file
    )

    model = package["model"]

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )

    print("\n" + "=" * 70)

    print(
        f"       RECORDS WITH PROBABILITY >= "
        f"{threshold:.2f}%"
    )

    print("=" * 70)

    count = 0

    for i in range(
        len(df)
    ):

        max_probability = (
            max(
                probabilities[i]
            ) * 100
        )

        if max_probability >= threshold:

            actual = str(
                y.iloc[i]
            )

            predicted = str(
                predictions[i]
            )

            result = (
                "CORRECT"
                if actual == predicted
                else "INCORRECT"
            )

            print(
                f"\nRecord #{i + 1}"
            )

            print(
                f"Actual       : {actual}"
            )

            print(
                f"Predicted    : {predicted}"
            )

            print(
                f"Probability  : "
                f"{max_probability:.2f}%"
            )

            print(
                f"Result       : {result}"
            )

            count += 1

            if count >= 20:

                break

    if count == 0:

        print(
            "\nNo records found."
        )

    print("=" * 70)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(
    model_file
):

    package = joblib.load(
        model_file
    )

    model = package["model"]

    features = package["features"]

    if not hasattr(
        model,
        "feature_importances_"
    ):

        print(
            "\nFeature importance "
            "is not available for this model."
        )

        return

    importance = (
        model.feature_importances_
    )

    data = pd.DataFrame({

        "Feature":
            features,

        "Importance":
            importance

    })

    data = data.sort_values(
        "Importance",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("                 FEATURE IMPORTANCE")
    print("=" * 70)

    for _, row in data.iterrows():

        print(
            f"{row['Feature']:<25}"
            f"{row['Importance'] * 100:>8.2f}%"
        )

    print("=" * 70)