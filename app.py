"""
app.py

Predictive Risk Modeling Framework for Microplastic Pollution
- Loads MicroPlastic.csv
- Preprocesses data (outlier handling, skewness correction, encoding, scaling, splitting)
- Trains three classification models:
    * Logistic Regression
    * Random Forest Classifier
    * Gradient Boosting Classifier
- Validates models (accuracy, precision, recall, f1)
- Performs K-Fold cross-validation (k=5)
- Modular, documented, research-friendly

Usage:
    python app.py

Requirements:
    pandas, numpy, scikit-learn, seaborn (optional), matplotlib (optional)
"""

import os
import sys
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# Optional plotting imports
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def load_and_preprocess_data(
    csv_path: str = "MicroPlastic.csv",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict]:
    """
    Load dataset and perform preprocessing:
      - Load CSV
      - Handle missing values
      - Handle outliers for specified numeric columns using IQR (winsorize/cap)
      - Compute skewness and apply log1p to skewed numeric columns (|skew| > 0.5)
      - Encode specified categorical columns (One-Hot for low-cardinality, LabelEncoder otherwise)
      - Scale numerical columns with StandardScaler
      - Split into train/test sets (80/20)
    Returns:
      X_train, X_test, y_train, y_test, meta_info (dict with scalers/encoders/feature names)
    """

    print("\n[1] Loading dataset...")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found in the current directory ({os.getcwd()}). Exiting.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f" - Loaded dataset with shape: {df.shape}")
    print(" - Columns:", list(df.columns))

    # Decide target variable automatically
    if "Risk_Level" in df.columns:
        target_col = "Risk_Level"
        print(" - Using 'Risk_Level' as the target variable.")
    elif "Risk_Type" in df.columns:
        target_col = "Risk_Type"
        print(" - 'Risk_Level' not found; using 'Risk_Type' as the target variable.")
    else:
        # Fallback: last column as target
        target_col = df.columns[-1]
        print(
            f" - Neither 'Risk_Level' nor 'Risk_Type' found. Falling back to last column as target: '{target_col}'."
        )

    # Make a working copy
    df_original = df.copy()

    # Fill missing values for numerical and categorical separately
    print("\n[2] Handling missing values...")
    num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_all = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f" - Numerical columns detected: {num_cols_all}")
    print(f" - Categorical/object columns detected: {cat_cols_all}")

    # Pre-specified numerical columns where we must handle outliers
    outlier_cols = [
        "MP_Count_per_L",
        "Risk_Score",
        "Microplastic_Size_mm_midpoint",
        "Density_midpoint",
    ]

    # Ensure missing-value handling
    for c in num_cols_all:
        median = df[c].median()
        df[c] = df[c].fillna(median)
    for c in cat_cols_all:
        df[c] = df[c].fillna("Missing")

    print(" - Missing numerical values filled with median; categorical with 'Missing'.")

    # OUTLIER HANDLING using IQR on specified columns
    print("\n[3] Outlier detection & handling (IQR capping) on specific columns:")
    for col in outlier_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            n_outliers_low = (df[col] < lower_bound).sum()
            n_outliers_high = (df[col] > upper_bound).sum()
            total_outliers = n_outliers_low + n_outliers_high
            print(
                f" - {col}: Q1={q1:.4g}, Q3={q3:.4g}, IQR={iqr:.4g}, lower={lower_bound:.4g}, upper={upper_bound:.4g}"
            )
            print(f"   -> Outliers below: {n_outliers_low}, above: {n_outliers_high} (total {total_outliers})")
            # Cap values (winsorization)
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            print(f"   -> Values capped to IQR bounds for {col}.")
        else:
            print(f" - {col}: NOT FOUND or NOT NUMERIC in dataset. Skipping outlier handling for this column.")

    # Recompute numerical columns list (some columns might be encoded later)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target if numeric to avoid transforming it
    if target_col in num_cols:
        num_cols.remove(target_col)

    # Transform skewed numerical columns
    print("\n[4] Skewness analysis and transformation (log1p) on numerical columns:")
    skew_info = {}
    for col in num_cols:
        skew_val = float(df[col].skew())
        skew_info[col] = {"original_skew": skew_val}
    skew_df = pd.DataFrame(skew_info).T
    skew_df = skew_df.rename_axis("feature").reset_index()
    print(" - Skewness BEFORE transformation:")
    for feature, info in skew_info.items():
        print(f"   {feature}: skew = {info['original_skew']:.4f}")

    # Identify skewed columns: |skew| > 0.5
    skewed_cols = [c for c, v in skew_info.items() if abs(v["original_skew"]) > 0.5]
    print(f" - Columns identified as skewed (|skew| > 0.5): {skewed_cols}")

    # Apply log1p to skewed columns (with shift if necessary)
    transforms_applied = {}
    for col in skewed_cols:
        col_min = df[col].min()
        shift = 0.0
        if col_min <= -1e-9:
            # need to shift so values > -1 for log1p
            shift = abs(col_min) + 1e-6
            print(f"   -> Column {col} has min {col_min:.4g}; will shift by {shift:.6g} before log1p.")
        df[col] = np.log1p(df[col] + shift)
        transforms_applied[col] = {"shift": shift}
    # Recalculate skewness
    print("\n - Skewness AFTER transformation (for transformed columns):")
    for col in skewed_cols:
        new_skew = float(df[col].skew())
        print(f"   {col}: new skew = {new_skew:.4f}")

    # CATEGORICAL ENCODING
    # Columns required by the user to encode (treat them as categorical even if numeric-looking)
    cols_to_encode = [
        "Location",
        "Shape",
        "Polymer_Type",
        "pH",
        "Salinity",
        "Industrial_Activity",
        "Population_Density",
        "Risk_Type",
        "Risk_Level",
        "Author",
    ]
    print("\n[5] Encoding categorical variables (specified list):")
    # Make sure columns exist in df
    cols_to_encode = [c for c in cols_to_encode if c in df.columns]
    print(f" - Columns to encode (present in dataset): {cols_to_encode}")

    df_encoded = df.copy()
    label_encoders = {}
    onehot_columns = []

    for col in cols_to_encode:
        # Convert to string categories to avoid accidental numeric issues:
        df_encoded[col] = df_encoded[col].astype(str).fillna("Missing")

        nunique = df_encoded[col].nunique()
        print(f"   Encoding {col}: unique values = {nunique}")
        if nunique <= 10:
            # One-hot encode using pandas.get_dummies (drop_first to avoid multicollinearity)
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
            onehot_columns.extend(list(dummies.columns))
            print(f"    -> One-Hot encoded {col} into {len(dummies.columns)} columns.")
        else:
            # Label encode
            le = LabelEncoder()
            df_encoded[col + "_LE"] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
            df_encoded = df_encoded.drop(columns=[col])
            print(f"    -> LabelEncoded {col} into column {col + '_LE'}.")

    # Ensure target variable is numeric (LabelEncode if necessary)
    print("\n[6] Preparing target variable:")
    y = df_encoded[target_col] if target_col in df_encoded.columns else df[target_col]
    if y.dtype == "O" or not pd.api.types.is_numeric_dtype(y):
        le_target = LabelEncoder()
        y_enc = le_target.fit_transform(y.astype(str))
        y = pd.Series(y_enc, name=target_col)
        label_encoders[target_col + "_target"] = le_target
        print(f" - Target '{target_col}' label-encoded (classes: {list(le_target.classes_)})")
    else:
        y = pd.Series(y, name=target_col)

    # Drop target from feature DataFrame if present
    if target_col in df_encoded.columns:
        X_df = df_encoded.drop(columns=[target_col])
    else:
        X_df = df_encoded.copy()

    # Re-evaluate numerical columns after encoding (we only scale the numeric features)
    numeric_features = X_df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n[7] Numerical features to be scaled: {numeric_features}")

    # IMPORTANT: The user requested to apply StandardScaler to the numerical columns and display first rows
    scaler = StandardScaler()
    X_numeric = pd.DataFrame(
        scaler.fit_transform(X_df[numeric_features]),
        columns=numeric_features,
        index=X_df.index,
    )
    print("\n - First few rows of scaled numerical features:")
    print(X_numeric.head())

    # Combine scaled numeric with any remaining non-numeric features (should be none)
    non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f" - Note: Non-numeric features still present (unexpected): {non_numeric}")
        X_final = pd.concat([X_numeric, X_df[non_numeric].reset_index(drop=True)], axis=1)
    else:
        X_final = X_numeric

    # Final features and shapes
    print(f"\n - Final feature matrix shape: {X_final.shape}")
    print(f" - Target vector shape: {y.shape}")

    # Train-test split
    print("\n[8] Splitting data into training and testing sets (80% train / 20% test)")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
    except Exception:
        # If stratify fails (e.g., single class), fallback without stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y, test_size=0.2, random_state=RANDOM_STATE, stratify=None
        )

    print(f" - X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f" - y_train: {y_train.shape}, y_test: {y_test.shape}")

    meta = {
        "scaler": scaler,
        "label_encoders": label_encoders,
        "onehot_columns": onehot_columns,
        "numeric_features": numeric_features,
        "target_col": target_col,
        "transforms_applied": transforms_applied,
    }

    return X_train, X_test, y_train, y_test, meta


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
    """
    Instantiate and train three classification models on the provided training data:
      - Logistic Regression
      - Random Forest Classifier
      - Gradient Boosting Classifier

    Returns dictionary of trained models.
    """
    print("\n[9] Training models...")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }

    trained_models = {}
    for name, model in models.items():
        print(f" - Training {name} ...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"   -> {name} trained. Model configuration / params:")
        # Print a compact model config summary
        params = model.get_params()
        # Print only the top-level keys to keep output readable
        top_params = {k: params[k] for k in sorted(params.keys())[:10]}
        print(f"      {top_params} ... (total params: {len(params)})")

    return trained_models


def validate_models(
    trained_models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate each trained model on test set and compute:
      - Accuracy, Precision (weighted), Recall (weighted), F1-score (weighted)
    Print classification report for each model and return summary metrics.
    """
    print("\n[10] Validating models on the test set...")

    results = {}
    for name, model in trained_models.items():
        print(f"\n--- Evaluating {name} ---")
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # Compare models summary
    print("\n[Summary] Model performance comparison (test set):")
    for name, metrics in results.items():
        print(
            f" - {name:15s} | Acc: {metrics['accuracy']:.4f} | Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}"
        )

    return results


def cross_validate_models(
    trained_models: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 5,
) -> Dict[str, Tuple[float, float]]:
    """
    Perform K-Fold cross-validation (with StratifiedKFold) for each model using accuracy as scoring.
    Returns average accuracy and standard deviation for each model.
    """
    print(f"\n[11] Cross-Validation using StratifiedKFold (k={k}) on full dataset...")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    cv_results = {}
    for name, model in trained_models.items():
        print(f" - Cross-validating {name} ...")
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            print(f"   -> Accuracy scores: {scores}")
            print(f"   -> Mean accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
            cv_results[name] = (mean_score, std_score)
        except Exception as e:
            print(f"   -> Cross-validation failed for {name}: {e}")
            cv_results[name] = (np.nan, np.nan)

    print("\n[Cross-Validation Summary]")
    for name, (mean_score, std_score) in cv_results.items():
        print(f" - {name:15s} | CV Mean Acc: {mean_score:.4f} | Std: {std_score:.4f}")

    return cv_results


def main():
    print("=== Predictive Risk Modeling Framework for Microplastic Pollution ===")
    X_train, X_test, y_train, y_test, meta = load_and_preprocess_data(csv_path="MicroPlastic.csv")

    # Train models
    trained_models = train_models(X_train, y_train)

    # Validate on test set
    test_results = validate_models(trained_models, X_test, y_test)

    # For cross-validation, combine train+test back to full
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)

    # Cross-validate models
    cv_results = cross_validate_models(trained_models, X_full, y_full, k=5)

    print("\n=== Completed end-to-end run ===")
    print(" - Final Test Results Summary:")
    for name, metrics in test_results.items():
        print(
            f"   {name:15s}: Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
        )

    print("\n - Cross-Validation Summary (accuracy):")
    for name, (mean_acc, std_acc) in cv_results.items():
        print(f"   {name:15s}: Mean Acc={mean_acc:.4f}, Std={std_acc:.4f}")

    print("\nYou can inspect 'meta' dict in the code for encoding/scaling artifacts if needed.")


if __name__ == "__main__":
    main()
