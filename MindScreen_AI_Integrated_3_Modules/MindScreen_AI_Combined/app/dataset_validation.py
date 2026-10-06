"""
Validate the PHQ-9 implementation against the supplied CSV dataset.

The script tries to identify the nine PHQ question columns and the dataset's
existing PHQ_Total / PHQ_Severity columns. It does not train an ML model.
"""

from pathlib import Path
import re
import pandas as pd

from phq9 import get_severity

DATASET = Path(__file__).resolve().parent.parent / "data" / "PHQ-9_Dataset_5th Edition.csv"

def normalize(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())

def find_column(columns, candidates):
    normalized = {normalize(c): c for c in columns}
    for candidate in candidates:
        key = normalize(candidate)
        if key in normalized:
            return normalized[key]
    return None

def find_q_columns(columns):
    found = {}
    for i in range(1, 10):
        candidates = [
            f"Q{i}", f"Q{i}_", f"Question{i}", f"PHQ{i}",
            f"PHQ_{i}", f"Item{i}", f"Item_{i}"
        ]
        col = find_column(columns, candidates)
        if col:
            found[i] = col

    # Fallback: columns whose names start with q + number.
    if len(found) < 9:
        for col in columns:
            m = re.match(r"^\s*q\s*([1-9])\b", str(col), re.I)
            if m and int(m.group(1)) not in found:
                found[int(m.group(1))] = col

    return found

def to_score(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        mapping = {
            "not at all": 0,
            "several days": 1,
            "more than half the days": 2,
            "nearly every day": 3,
        }
        if text in mapping:
            return mapping[text]
        try:
            value = int(float(text))
        except ValueError:
            return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value in (0, 1, 2, 3) else None

def main():
    print("=" * 60)
    print("             PHQ-9 DATASET VALIDATION")
    print("=" * 60)

    if not DATASET.exists():
        print("\nDataset not found:")
        print(DATASET)
        print("\nCopy your CSV into the data folder with this exact name:")
        print("PHQ-9_Dataset_5th Edition.csv")
        return

    df = pd.read_csv(DATASET)
    print(f"\nDataset: {DATASET.name}")
    print(f"Records: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    q_cols = find_q_columns(df.columns)
    total_col = find_column(df.columns, ["PHQ_Total", "PHQ Total", "PHQTotal"])
    severity_col = find_column(df.columns, ["PHQ_Severity", "PHQ Severity", "PHQSeverity"])

    print("\nDetected PHQ question columns:")
    for i in range(1, 10):
        print(f"  Q{i}: {q_cols.get(i, 'NOT FOUND')}")

    print(f"\nDetected total column: {total_col or 'NOT FOUND'}")
    print(f"Detected severity column: {severity_col or 'NOT FOUND'}")

    if len(q_cols) != 9:
        print("\nERROR: Could not identify all 9 PHQ question columns.")
        print("Actual columns in the CSV:")
        for col in df.columns:
            print(f"  - {col}")
        return

    if not total_col:
        print("\nERROR: PHQ_Total column was not found.")
        return

    score_matches = 0
    score_comparable = 0
    severity_matches = 0
    severity_comparable = 0
    invalid_rows = []

    for idx, row in df.iterrows():
        answers = [to_score(row[q_cols[i]]) for i in range(1, 10)]

        if any(a is None for a in answers):
            invalid_rows.append(idx + 2)
            continue

        calculated = sum(answers)
        dataset_total = to_score(row[total_col])

        # PHQ_Total can be 0..27, so to_score() is suitable here.
        if dataset_total is not None:
            score_comparable += 1
            if calculated == dataset_total:
                score_matches += 1

        if severity_col:
            dataset_severity = str(row[severity_col]).strip().lower()
            calculated_severity = get_severity(calculated).lower()
            if dataset_severity:
                severity_comparable += 1
                if dataset_severity == calculated_severity:
                    severity_matches += 1

    print("\n" + "-" * 60)
    print("VALIDATION RESULT")
    print("-" * 60)

    if score_comparable:
        score_accuracy = score_matches / score_comparable * 100
        print(f"Score matches       : {score_matches}")
        print(f"Score mismatches    : {score_comparable - score_matches}")
        print(f"Score accuracy      : {score_accuracy:.2f}%")
    else:
        print("No comparable PHQ_Total records found.")

    if severity_col and severity_comparable:
        severity_accuracy = severity_matches / severity_comparable * 100
        print(f"\nSeverity matches    : {severity_matches}")
        print(f"Severity mismatches : {severity_comparable - severity_matches}")
        print(f"Severity accuracy   : {severity_accuracy:.2f}%")
    elif severity_col:
        print("\nNo comparable PHQ_Severity records found.")

    print(f"\nRows with invalid/missing Q1-Q9: {len(invalid_rows)}")

    if invalid_rows:
        print("First invalid CSV row numbers:", invalid_rows[:10])

    print("\nValidation complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
