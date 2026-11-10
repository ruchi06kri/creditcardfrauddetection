from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from sklearn.preprocessing import FunctionTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "creditcard.csv" / "AIML Dataset.csv"
MODEL_PATH = BASE_DIR / "fraud_detection_pipeline.pkl"

TARGET = "isFraud"
NUMERIC_FEATURES = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest"
]
CATEGORICAL_FEATURES = ["type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Check that the CSV is inside the creditcard.csv folder."
        )

    print(f"Reading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])]
    df[TARGET] = df[TARGET].astype(int)

    # Keep only supported transaction types and convert numeric columns safely.
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["type"] = df["type"].fillna("UNKNOWN").astype(str)

    X = df[FEATURES]
    y = df[TARGET]

    if y.nunique() < 2:
        raise ValueError("The target is missing one of its classes (0 or 1).")

    print(f"Rows: {len(df):,}")
    print("Class counts:")
    print(y.value_counts().sort_index().rename(index={0: "Non-fraud", 1: "Fraud"}))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES)
    ])

    # Balanced random forest is a reasonable baseline for highly imbalanced fraud data.
    model = RandomForestClassifier(
        n_estimators=160,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    print("\nClassification report (threshold 0.50):")
    print(classification_report(y_test, predictions, digits=4, zero_division=0))
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, predictions))
    print(f"Average precision (PR-AUC summary): {average_precision_score(y_test, probabilities):.6f}")

    # Save the complete preprocessing + model pipeline so the app uses identical transformations.
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    print(f"\nSaved trained pipeline to: {MODEL_PATH}")
    print("Note: this is a learning/demo model. Validate thresholds and performance before real use.")

if __name__ == "__main__":
    main()
