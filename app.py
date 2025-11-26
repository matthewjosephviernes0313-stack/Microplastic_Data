"""
app.py

Predictive Risk Modeling Framework for Microplastic Pollution (v2)
- Adds flexible CSV upload/selection handling:
    * Accepts --csv <path_or_url> command-line argument
    * Interactive prompt to type a local path or URL
    * Optional GUI file picker (tkinter) when available and user accepts
- All other processing (preprocessing, visualizations, modeling, validation, CV)
  retained from the previous version with plots saved to the outputs/ folder.

Usage:
    python app.py --csv path/to/MicroPlastic.csv
    python app.py          # will prompt for path/URL or open file dialog

Requirements:
    pandas, numpy, scikit-learn, matplotlib, seaborn
"""

import os
import sys
import warnings
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

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
    confusion_matrix,
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# Output directory for saved plots
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set(style="whitegrid")


# ----------------------------
# CSV upload / selection helpers
# ----------------------------
def parse_cli_csv_arg() -> Optional[str]:
    """Parse command-line args for --csv <path_or_url> or first positional arg."""
    args = sys.argv[1:]
    if not args:
        return None
    # Look for --csv flag
    if "--csv" in args:
        i = args.index("--csv")
        if i + 1 < len(args):
            return args[i + 1]
    # Accept first positional argument if it looks like a path/URL
    first = args[0]
    if not first.startswith("-"):
        return first
    return None


def try_gui_filepicker() -> Optional[str]:
    """
    Try to open a file picker using tkinter.
    Returns a selected file path or None if not available / cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        print("Opening file picker dialog...")
        file_path = filedialog.askopenfilename(
            title="Select MicroPlastic CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        if file_path:
            return file_path
        return None
    except Exception:
        return None


def get_csv_path_interactive() -> str:
    """
    Interactive routine to obtain CSV path or URL from the user.
    Tries CLI arg first, then prompt input, then optional GUI picker.
    Raises SystemExit if the user does not provide a valid path.
    """
    # 1) CLI arg
    cli_path = parse_cli_csv_arg()
    if cli_path:
        print(f"Using CSV from command-line argument: {cli_path}")
        return cli_path

    # 2) Interactive prompt
    print("\nNo --csv argument provided.")
    print("You can provide a local path or a URL to the CSV file (e.g., https://.../MicroPlastic.csv).")
    user_input = input("Enter path or URL to CSV (leave blank to open file dialog if available): ").strip()
    if user_input:
        print(f"Using CSV path/URL entered: {user_input}")
        return user_input

    # 3) Try GUI file picker
    print("Attempting to open a file picker dialog...")
    picked = try_gui_filepicker()
    if picked:
        print(f"Selected file: {picked}")
        return picked

    # 4) If nothing provided, fail with guidance
    print("No CSV file selected. Exiting. Provide a CSV file via one of these options:")
    print(" - Command line: python app.py --csv path/to/MicroPlastic.csv")
    print(" - Interactive: type or paste a path/URL when prompted")
    print(" - Use a GUI file picker (if your environment permits)")
    sys.exit(1)


# ----------------------------
# Helper plotting functions
# ----------------------------
def save_and_show(fig, filename: str, dpi: int = 150, tight: bool = True, show: bool = False):
    """Save figure to outputs directory and optionally show it."""
    path = os.path.join(OUTPUT_DIR, filename)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    print(f"   -> Plot saved to {path}")


def plot_missing_values_heatmap(df: pd.DataFrame, fname: str = "missing_heatmap.png"):
    """Plot heatmap of missing values (boolean)."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(df.columns) * 0.2)))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax)
    ax.set_title("Missing Values Heatmap (True = missing)")
    save_and_show(fig, fname)


def plot_boxplot_before_after(before: pd.DataFrame, after: pd.DataFrame, cols: List[str], prefix: str = "outlier"):
    """
    Plot boxplots before and after for columns in 'cols'. Saves one figure per column.
    """
    for col in cols:
        if col not in before.columns or col not in after.columns:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.boxplot(x=before[col], ax=axes[0], color="skyblue")
        axes[0].set_title(f"{col} (Before)")
        sns.boxplot(x=after[col], ax=axes[1], color="lightgreen")
        axes[1].set_title(f"{col} (After - capped)")
        fig.suptitle(f"Outlier handling (IQR cap) for {col}")
        save_and_show(fig, f"{prefix}_boxplot_{col}.png")


def plot_histogram_comparison(df_before: pd.DataFrame, df_after: pd.DataFrame, cols: List[str], prefix: str = "skew"):
    """
    For each column, plot histogram/KDE before and after transformation (log1p).
    """
    for col in cols:
        if col not in df_before.columns or col not in df_after.columns:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df_before[col].dropna(), kde=True, ax=axes[0], color="cornflowerblue")
        axes[0].set_title(f"{col} BEFORE transformation\nskew={df_before[col].skew():.3f}")
        sns.histplot(df_after[col].dropna(), kde=True, ax=axes[1], color="seagreen")
        axes[1].set_title(f"{col} AFTER transformation\nskew={df_after[col].skew():.3f}")
        fig.suptitle(f"Skewness correction for {col}")
        save_and_show(fig, f"{prefix}_hist_{col}.png")


def plot_categorical_counts(df: pd.DataFrame, cols: List[str], top_n: int = 10, prefix: str = "cat_counts"):
    """
    Plot bar charts of categorical counts for the provided columns (before encoding).
    """
    for col in cols:
        if col not in df.columns:
            continue
        vc = df[col].astype(str).value_counts().nlargest(top_n)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=vc.values, y=vc.index, palette="viridis", ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
        ax.set_title(f"Top {len(vc)} value counts for {col}")
        save_and_show(fig, f"{prefix}_{col}.png")


def plot_scaled_feature_distributions(X_scaled: pd.DataFrame, prefix: str = "scaled"):
    """Plot histograms for scaled numerical features (first up to 12 features)."""
    cols = X_scaled.columns.tolist()
    n = min(len(cols), 12)
    if n == 0:
        return
    sample_cols = cols[:n]
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 2 * n))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, sample_cols):
        sns.histplot(X_scaled[col], kde=True, ax=ax, color="orchid")
        ax.set_title(f"Scaled Distribution: {col}")
    fig.suptitle("Scaled numerical feature distributions (sample)")
    save_and_show(fig, f"{prefix}_feature_dists_sample.png")


def plot_feature_importances(model, feature_names: List[str], name: str, top_n: int = 10):
    """
    Plot top_n feature importances.
    Supports tree-based models with feature_importances_ and linear models with coef_.
    """
    importances = None
    title = ""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = f"{name} Feature Importances"
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            importances = np.mean(np.abs(coef), axis=0)
        title = f"{name} Coefficient magnitudes (mean abs)"
    else:
        print(f"   -> No importances/coefficients available for {name}")
        return

    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    sns.barplot(x=fi.values, y=fi.index, ax=ax, palette="magma")
    ax.set_title(title)
    ax.set_xlabel("Importance / |Coefficient|")
    save_and_show(fig, f"feature_importances_{name}.png")


def plot_confusion_matrix_heatmap(y_true, y_pred, classes: List[str], name: str):
    """Plot confusion matrix heatmap and save."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {name}")
    if classes is not None:
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes, rotation=0)
    save_and_show(fig, f"confusion_matrix_{name}.png")


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], prefix: str = "metrics"):
    """Plot bar chart comparing accuracy/precision/recall/f1 across models."""
    df = pd.DataFrame(metrics_dict).T  # models x metrics
    metrics = ["accuracy", "precision", "recall", "f1"]
    df = df[metrics]
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on test set")
    ax.set_ylabel("Score")
    save_and_show(fig, f"{prefix}_comparison.png")


def plot_cv_boxplot(cv_scores: Dict[str, np.ndarray], prefix: str = "cv"):
    """Plot boxplots of cross-validation accuracies for each model."""
    data = []
    labels = []
    for name, scores in cv_scores.items():
        data.append(scores)
        labels.append(name)
    if not any(len(arr) for arr in data):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=labels)
    ax.set_title("Cross-validation accuracy distribution")
    ax.set_ylabel("Accuracy")
    save_and_show(fig, f"{prefix}_boxplot.png")


# ----------------------------
# Core processing functions
# ----------------------------
def load_and_preprocess_data(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
    """
    Load dataset and perform preprocessing with visualizations at each step.
    csv_path may be a local path or URL readable by pandas.
    Returns:
      X_train, X_test, y_train, y_test, meta_info
    """
    print("\n[1] Loading dataset...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"ERROR: Failed to read CSV from '{csv_path}': {e}")
        sys.exit(1)

    print(f" - Loaded dataset with shape: {df.shape}")

    # Visualize missing values
    print("\n[2] Visualizing missing values...")
    try:
        plot_missing_values_heatmap(df, fname="missing_heatmap.png")
    except Exception as e:
        print(f"   -> Could not plot missing heatmap: {e}")

    # Determine target column
    if "Risk_Level" in df.columns:
        target_col = "Risk_Level"
        print(" - Using 'Risk_Level' as the target variable.")
    elif "Risk_Type" in df.columns:
        target_col = "Risk_Type"
        print(" - Using 'Risk_Type' as the target variable.")
    else:
        target_col = df.columns[-1]
        print(f" - Falling back to last column as target: '{target_col}'.")

    # Keep copy before modifications for plots
    df_before_outliers = df.copy()

    # Handle missing values
    print("\n[3] Handling missing values...")
    num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_all = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f" - Numerical columns detected: {num_cols_all}")
    print(f" - Categorical columns detected: {cat_cols_all}")

    for c in num_cols_all:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols_all:
        df[c] = df[c].fillna("Missing")

    print("   -> Filled numeric with median and categorical with 'Missing'.")

    # OUTLIER HANDLING using IQR for specified columns
    outlier_cols = [
        "MP_Count_per_L",
        "Risk_Score",
        "Microplastic_Size_mm_midpoint",
        "Density_midpoint",
    ]
    print("\n[4] Outlier detection & handling (IQR capping) on specific columns:")
    outlier_cols_present = [c for c in outlier_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not outlier_cols_present:
        print("   -> None of the specified outlier columns were present and numeric. Skipping outlier visuals.")
    else:
        print(f"   -> Columns to process for outliers: {outlier_cols_present}")

    # Create a before-capping snapshot for plotting
    df_snapshot_before = df.copy()

    # Cap extremes
    for col in outlier_cols_present:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_low = (df[col] < lower).sum()
        n_high = (df[col] > upper).sum()
        print(f"    - {col}: IQR bounds ({lower:.4g}, {upper:.4g}), below={n_low}, above={n_high}")
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    # Plot boxplots before/after
    try:
        plot_boxplot_before_after(df_snapshot_before, df, outlier_cols_present, prefix="outlier")
    except Exception as e:
        print(f"   -> Could not plot outlier boxplots: {e}")

    # SKNEWNESS ANALYSIS
    print("\n[5] Skewness analysis and log1p transformation for skewed numeric columns:")
    # Recompute numeric columns and avoid transforming target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    skew_info = {}
    for col in numeric_cols:
        skew_val = float(df[col].skew())
        skew_info[col] = skew_val
    print(" - Skewness BEFORE transformation (sample):")
    for k, v in list(skew_info.items())[:10]:
        print(f"    {k}: {v:.4f}")

    skewed_cols = [c for c, s in skew_info.items() if abs(s) > 0.5]
    print(f" - Columns identified as skewed (|skew| > 0.5): {skewed_cols}")

    df_snapshot_before_skew = df.copy()
    transforms_applied = {}
    for col in skewed_cols:
        col_min = df[col].min()
        shift = 0.0
        if col_min <= -1e-9:
            shift = abs(col_min) + 1e-6
            print(f"    -> Shifting {col} by {shift:.6g} before log1p due to negative min {col_min:.4g}")
        df[col] = np.log1p(df[col] + shift)
        transforms_applied[col] = {"shift": shift}

    # Plot histogram comparisons for skewed columns
    try:
        plot_histogram_comparison(df_snapshot_before_skew, df, skewed_cols, prefix="skew")
    except Exception as e:
        print(f"   -> Could not plot skewness histograms: {e}")

    # CATEGORICAL ENCODING
    print("\n[6] Categorical encoding (specified columns):")
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
    # Keep only those present
    cols_to_encode_present = [c for c in cols_to_encode if c in df.columns]
    print(f"   -> Columns present for encoding: {cols_to_encode_present}")

    # Visualize categorical distributions BEFORE encoding
    try:
        plot_categorical_counts(df, cols_to_encode_present, top_n=8, prefix="cat_before_encoding")
    except Exception as e:
        print(f"   -> Could not plot categorical counts: {e}")

    df_encoded = df.copy()
    label_encoders = {}
    onehot_columns = []

    for col in cols_to_encode_present:
        # Ensure string dtype for consistent encoding
        df_encoded[col] = df_encoded[col].astype(str).fillna("Missing")
        nunique = df_encoded[col].nunique()
        if nunique <= 10:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
            onehot_columns.extend(list(dummies.columns))
            print(f"    - One-hot encoded {col} into {len(dummies.columns)} columns.")
        else:
            le = LabelEncoder()
            df_encoded[col + "_LE"] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
            df_encoded = df_encoded.drop(columns=[col])
            print(f"    - Label-encoded {col} into {col + '_LE'} (classes: {len(le.classes_)})")

    # Prepare target variable
    if target_col in df_encoded.columns:
        y_raw = df_encoded[target_col]
    else:
        y_raw = df[target_col]
    if y_raw.dtype == "O" or not pd.api.types.is_numeric_dtype(y_raw):
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y_raw.astype(str)), name=target_col)
        label_encoders[target_col + "_target"] = le_target
        print(f"   -> Target '{target_col}' label-encoded (classes: {list(le_target.classes_)})")
    else:
        y = pd.Series(y_raw, name=target_col)

    # Remove target from features if still present
    if target_col in df_encoded.columns:
        X_df = df_encoded.drop(columns=[target_col])
    else:
        X_df = df_encoded.copy()

    # Feature scaling of numeric features
    numeric_features = X_df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n[7] Scaling numerical features (StandardScaler). Numeric features: {len(numeric_features)}")
    scaler = StandardScaler()
    if numeric_features:
        X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_df[numeric_features]), columns=numeric_features, index=X_df.index)
    else:
        X_numeric_scaled = pd.DataFrame(index=X_df.index)

    # Visualize scaled feature distributions (sample)
    try:
        plot_scaled_feature_distributions(X_numeric_scaled, prefix="scaled")
    except Exception as e:
        print(f"   -> Could not plot scaled feature distributions: {e}")

    # Combine scaled numeric and any remaining non-numeric (should be none)
    non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"   -> Note: non-numeric features still present (unexpected): {non_numeric}")
        X_final = pd.concat([X_numeric_scaled, X_df[non_numeric].reset_index(drop=True)], axis=1)
    else:
        X_final = X_numeric_scaled

    print(f" - Final feature matrix shape: {X_final.shape}; target shape: {y.shape}")

    # Train-test split
    print("\n[8] Splitting data into training and testing sets (80% train / 20% test)")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y, test_size=0.2, random_state=RANDOM_STATE, stratify=None
        )
        print("   -> Stratified split failed; used regular split.")

    print(f" - X_train: {X_train.shape}, X_test: {X_test.shape}")
    meta = {
        "scaler": scaler,
        "label_encoders": label_encoders,
        "onehot_columns": onehot_columns,
        "numeric_features": numeric_features,
        "target_col": target_col,
        "transforms_applied": transforms_applied,
        "csv_path": csv_path,
    }
    return X_train, X_test, y_train, y_test, meta


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
    """
    Train 3 classification models and return trained models.
    Also saves simple textual model summaries and feature importance plots.
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
        print(f"   -> {name} trained. Example params: { {k: v for k, v in list(model.get_params().items())[:6]} }")
        # Plot feature importances / coefficients if possible
        try:
            plot_feature_importances(model, feature_names=X_train.columns.tolist(), name=name, top_n=12)
        except Exception as e:
            print(f"   -> Could not plot feature importances for {name}: {e}")

    return trained_models


def validate_models(trained_models: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series, meta: Dict) -> Dict[str, Dict[str, float]]:
    """
    Evaluate models on test set, print metrics and produce confusion matrices and a comparison plot.
    """
    print("\n[10] Validating models on the test set...")
    results = {}
    # If label encoder for target exists, get class names for plotting
    le_target = meta.get("label_encoders", {}).get(meta.get("target_col", "") + "_target", None)
    class_names = None
    if le_target is not None:
        class_names = list(le_target.classes_)

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

        # Confusion matrix plot
        try:
            plot_confusion_matrix_heatmap(y_test, y_pred, classes=class_names, name=name)
        except Exception as e:
            print(f"   -> Could not plot confusion matrix for {name}: {e}")

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # Aggregate comparison plot
    try:
        plot_metrics_comparison(results, prefix="metrics")
    except Exception as e:
        print(f"   -> Could not plot metrics comparison: {e}")

    # Print summary
    print("\n[Summary] Model performance comparison (test set):")
    for name, metrics in results.items():
        print(f" - {name:15s} | Acc: {metrics['accuracy']:.4f} | Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")

    return results


def cross_validate_models(trained_models: Dict[str, object], X: pd.DataFrame, y: pd.Series, k: int = 5) -> Dict[str, np.ndarray]:
    """
    Perform Stratified K-Fold cross-validation for each model.
    Returns dictionary of arrays (accuracy scores per fold) for each model and plots a boxplot.
    """
    print(f"\n[11] Cross-Validation using StratifiedKFold (k={k}) on full dataset...")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}
    for name, model in trained_models.items():
        print(f" - Cross-validating {name} ...")
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
            cv_scores[name] = scores
            print(f"   -> Scores: {scores}")
            print(f"   -> Mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        except Exception as e:
            print(f"   -> Cross-validation failed for {name}: {e}")
            cv_scores[name] = np.array([])

    # Plot boxplot of CV scores
    try:
        plot_cv_boxplot(cv_scores, prefix="cv")
    except Exception as e:
        print(f"   -> Could not plot CV boxplot: {e}")

    return cv_scores


def main():
    print("=== Predictive Risk Modeling Framework for Microplastic Pollution (with CSV upload support) ===")
    csv_path = get_csv_path_interactive()
    print(f"\nCSV source: {csv_path}")

    X_train, X_test, y_train, y_test, meta = load_and_preprocess_data(csv_path=csv_path)

    # Train models
    trained_models = train_models(X_train, y_train)

    # Validate on test set (produces confusion matrices + metrics plot)
    test_results = validate_models(trained_models, X_test, y_test, meta)

    # Combine train+test for cross-validation
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)

    # Cross-validate and plot CV accuracies
    cv_scores = cross_validate_models(trained_models, X_full, y_full, k=5)

    # Save summary CSVs of results and metadata
    try:
        pd.DataFrame(test_results).T.to_csv(os.path.join(OUTPUT_DIR, "test_results_summary.csv"))
        cv_summary = {k: (np.mean(v) if v.size else np.nan) for k, v in cv_scores.items()}
        pd.Series(cv_summary).to_csv(os.path.join(OUTPUT_DIR, "cv_mean_accuracy_summary.csv"))
        # Save metadata (what file was used)
        pd.Series(meta).to_csv(os.path.join(OUTPUT_DIR, "meta_summary.csv"))
        print(f"\n - Summary CSVs saved to {OUTPUT_DIR}")
    except Exception as e:
        print(f"   -> Could not save summary CSVs: {e}")

    print("\n=== Completed end-to-end run with visualizations ===")
    print(f"All plots and summaries saved in the folder: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
