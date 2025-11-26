# app.py
"""
Predictive Risk Modeling Framework for Microplastic Pollution (Streamlit UI, auto-run)

This corrected version:
- Robust CSV reading & preview
- Outlier handling (IQR capping)
- Skewness detection + log1p shifts
- Categorical handling (smart one-hot / label encode for high-cardinality)
- Ensures final X is numeric (no stray object columns) before scaling and modelling
- Feature ranking using RandomForest and SelectFromModel
- Trains LogisticRegression, RandomForest, GradientBoosting
- Evaluates on test set and performs Stratified K-Fold cross-validation
- Saves plots & CSV outputs to ./outputs/
"""

import os
import sys
import warnings
from typing import Dict, Tuple, List, Optional

import io
import glob
import zipfile
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
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

# Directories
OUTPUT_DIR = "outputs"
INPUT_DIR = "inputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
sns.set(style="whitegrid")

# Try Streamlit import
try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False


# ----------------------------
# Robust CSV preview helper
# ----------------------------
def read_csv_preview(path: str, nrows: int = 10) -> Tuple[pd.DataFrame, str]:
    encodings_to_try = [None, "utf-8", "latin1", "cp1252"]
    last_exception = None

    for enc in encodings_to_try:
        try:
            if enc is None:
                df = pd.read_csv(path, nrows=nrows)
            else:
                df = pd.read_csv(path, encoding=enc, nrows=nrows)
            return df, enc or "default"
        except UnicodeDecodeError as e:
            last_exception = e
            continue
        except Exception as e:
            last_exception = e
            continue

    # Final fallback: open with errors='replace'
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        df = pd.read_csv(io.StringIO(text), nrows=nrows)
        return df, "replaced-invalid-bytes"
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV for preview. Last error: {last_exception}; final error: {e}")


# ----------------------------
# Plot helpers (save to outputs and return path)
# ----------------------------
def save_and_show(fig, filename: str, dpi: int = 150, tight: bool = True):
    path = os.path.join(OUTPUT_DIR, filename)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_missing_values_heatmap(df: pd.DataFrame, fname: str = "missing_heatmap.png"):
    fig, ax = plt.subplots(figsize=(10, max(4, len(df.columns) * 0.2)))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax)
    ax.set_title("Missing Values Heatmap (True = missing)")
    return save_and_show(fig, fname)


def plot_boxplot_before_after(before: pd.DataFrame, after: pd.DataFrame, cols: List[str], prefix: str = "outlier"):
    saved = []
    for col in cols:
        if col not in before.columns or col not in after.columns:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.boxplot(x=before[col], ax=axes[0])
        axes[0].set_title(f"{col} (Before)")
        sns.boxplot(x=after[col], ax=axes[1])
        axes[1].set_title(f"{col} (After - capped)")
        fig.suptitle(f"Outlier handling (IQR cap) for {col}")
        saved.append(save_and_show(fig, f"{prefix}_boxplot_{col}.png"))
    return saved


def plot_histogram_comparison(df_before: pd.DataFrame, df_after: pd.DataFrame, cols: List[str], prefix: str = "skew"):
    saved = []
    for col in cols:
        if col not in df_before.columns or col not in df_after.columns:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df_before[col].dropna(), kde=True, ax=axes[0])
        axes[0].set_title(f"{col} BEFORE transformation\nskew={df_before[col].skew():.3f}")
        sns.histplot(df_after[col].dropna(), kde=True, ax=axes[1])
        axes[1].set_title(f"{col} AFTER transformation\nskew={df_after[col].skew():.3f}")
        fig.suptitle(f"Skewness correction for {col}")
        saved.append(save_and_show(fig, f"{prefix}_hist_{col}.png"))
    return saved


def plot_categorical_counts(df: pd.DataFrame, cols: List[str], top_n: int = 10, prefix: str = "cat_counts"):
    saved = []
    for col in cols:
        if col not in df.columns:
            continue
        vc = df[col].astype(str).value_counts().nlargest(top_n)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=vc.values, y=vc.index, ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
        ax.set_title(f"Top {len(vc)} value counts for {col}")
        saved.append(save_and_show(fig, f"{prefix}_{col}.png"))
    return saved


def plot_scaled_feature_distributions(X_scaled: pd.DataFrame, prefix: str = "scaled"):
    saved = []
    cols = X_scaled.columns.tolist()
    n = min(len(cols), 12)
    if n == 0:
        return saved
    sample_cols = cols[:n]
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 2 * n))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, sample_cols):
        sns.histplot(X_scaled[col], kde=True, ax=ax)
        ax.set_title(f"Scaled Distribution: {col}")
    fig.suptitle("Scaled numerical feature distributions (sample)")
    saved.append(save_and_show(fig, f"{prefix}_feature_dists_sample.png"))
    return saved


def plot_feature_importances(model, feature_names: List[str], name: str, top_n: int = 10):
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
        return None

    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    sns.barplot(x=fi.values, y=fi.index, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance / |Coefficient|")
    return save_and_show(fig, f"feature_importances_{name}.png")


def plot_confusion_matrix_heatmap(y_true, y_pred, classes: List[str], name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {name}")
    if classes is not None:
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes, rotation=0)
    return save_and_show(fig, f"confusion_matrix_{name}.png")


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], prefix: str = "metrics"):
    df = pd.DataFrame(metrics_dict).T  # models x metrics
    metrics = ["accuracy", "precision", "recall", "f1"]
    df = df.reindex(columns=[m for m in metrics if m in df.columns])
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on test set")
    ax.set_ylabel("Score")
    return save_and_show(fig, f"{prefix}_comparison.png")


def plot_cv_boxplot(cv_scores: Dict[str, np.ndarray], prefix: str = "cv"):
    data = []
    labels = []
    for name, scores in cv_scores.items():
        data.append(scores)
        labels.append(name)
    if not any(len(arr) for arr in data):
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=labels)
    ax.set_title("Cross-validation accuracy distribution")
    ax.set_ylabel("Accuracy")
    return save_and_show(fig, f"{prefix}_boxplot.png")


# ----------------------------
# Feature ranking / selection
# ----------------------------
def rank_and_select_features(X: pd.DataFrame, y: pd.Series, top_n: int = 20) -> Tuple[List[str], Dict]:
    """
    Use RandomForest to rank features, then SelectFromModel to pick important ones.
    Returns selected feature list and meta info.
    """
    meta = {}
    if X.shape[0] < 2 or y.nunique() < 2:
        # not enough data to rank
        return X.columns.tolist(), meta

    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = importances.head(min(top_n, len(importances))).index.tolist()
    # Use SelectFromModel with threshold mean importance
    selector = SelectFromModel(rf, threshold="mean", prefit=True)
    selected_mask = selector.get_support()
    selected = list(X.columns[selected_mask])
    if not selected:
        # fallback to top k
        selected = top_features[: min(10, len(top_features))]
    meta["feature_importances"] = importances.to_dict()
    meta["top_features"] = top_features
    meta["selected_features"] = selected
    return selected, meta


# ----------------------------
# Core processing (same pipeline)
# ----------------------------
def load_and_preprocess_data(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV from '{csv_path}': {e}")

    visuals = {"plots": []}

    # Missing values heatmap
    try:
        visuals["plots"].append(plot_missing_values_heatmap(df, fname="missing_heatmap.png"))
    except Exception:
        pass

    # Target selection
    if "Risk_Level" in df.columns:
        target_col = "Risk_Level"
    elif "Risk_Type" in df.columns:
        target_col = "Risk_Type"
    else:
        target_col = df.columns[-1]

    # Fill missing
    num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_all = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in num_cols_all:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols_all:
        df[c] = df[c].fillna("Missing")

    # Outlier handling (IQR) for some expected columns if present
    outlier_cols = [
        "MP_Count_per_L",
        "Risk_Score",
        "Microplastic_Size_mm_midpoint",
        "Density_midpoint",
    ]
    outlier_cols_present = [c for c in outlier_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    df_before_outliers = df.copy()
    for col in outlier_cols_present:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    try:
        visuals["plots"].extend(plot_boxplot_before_after(df_before_outliers, df, outlier_cols_present, prefix="outlier"))
    except Exception:
        pass

    # Skewness detection & log1p transform on numeric columns (but not target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    skew_info = {col: float(df[col].skew()) for col in numeric_cols}
    skewed_cols = [c for c, s in skew_info.items() if abs(s) > 0.5]
    df_before_skew = df.copy()
    transforms_applied = {}
    for col in skewed_cols:
        col_min = df[col].min()
        shift = 0.0
        if col_min <= -1e-9:
            shift = abs(col_min) + 1e-6
        # avoid taking log of zero/negatives
        df[col] = np.log1p(df[col] + shift)
        transforms_applied[col] = {"shift": shift}
    try:
        visuals["plots"].extend(plot_histogram_comparison(df_before_skew, df, skewed_cols, prefix="skew"))
    except Exception:
        pass

    # Categorical encoding for known columns (if present)
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
    cols_to_encode_present = [c for c in cols_to_encode if c in df.columns]
    try:
        visuals["plots"].extend(plot_categorical_counts(df, cols_to_encode_present, top_n=8, prefix="cat_before_encoding"))
    except Exception:
        pass

    df_encoded = df.copy()
    label_encoders = {}
    onehot_columns = []

    # Encode the specified categorical columns using either get_dummies (low cardinality) or LabelEncoder (high)
    for col in cols_to_encode_present:
        df_encoded[col] = df_encoded[col].astype(str).fillna("Missing")
        nunique = df_encoded[col].nunique()
        if nunique <= 10:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
            onehot_columns.extend(list(dummies.columns))
        else:
            le = LabelEncoder()
            df_encoded[col + "_LE"] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
            df_encoded = df_encoded.drop(columns=[col])

    # After user-specified encoding, ensure ALL remaining object columns are encoded so X is numeric
    remaining_obj_cols = df_encoded.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in remaining_obj_cols:
        df_encoded[col] = df_encoded[col].astype(str).fillna("Missing")
        nunique = df_encoded[col].nunique()
        # If small cardinality, one-hot; otherwise label-encode
        if nunique <= 12 and df_encoded.shape[0] >= nunique:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
            onehot_columns.extend(list(dummies.columns))
        else:
            le = LabelEncoder()
            df_encoded[col + "_LE"] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
            df_encoded = df_encoded.drop(columns=[col])

    # Prepare target (y)
    if target_col in df.columns:
        y_raw = df[target_col]
    else:
        # fallback if target not found in original (rare)
        raise RuntimeError(f"Target column '{target_col}' not found in dataset.")

    if y_raw.dtype == "O" or not pd.api.types.is_numeric_dtype(y_raw):
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y_raw.astype(str)), name=target_col)
        label_encoders[target_col + "_target"] = le_target
    else:
        y = pd.Series(y_raw, name=target_col)

    # Build X (drop original target if present in df_encoded)
    if target_col in df_encoded.columns:
        X_df = df_encoded.drop(columns=[target_col])
    else:
        X_df = df_encoded.copy()

    # Ensure all columns of X_df are numeric (they should be after encoding)
    non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        # last resort: try to coerce to numeric
        for c in non_numeric:
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce").fillna(0.0)

    # Feature scaling (fit on numeric features)
    numeric_features = X_df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler() if numeric_features else None
    if numeric_features:
        X_scaled = pd.DataFrame(scaler.fit_transform(X_df[numeric_features]), columns=numeric_features, index=X_df.index)
    else:
        X_scaled = pd.DataFrame(index=X_df.index)

    X_final = X_scaled  # X_scaled contains all features (we coerced non-numeric earlier)
    # If there are columns that were non-numeric but coerced into numeric with different names, they are in X_final already.

    try:
        visuals["plots"].extend(plot_scaled_feature_distributions(X_final, prefix="scaled"))
    except Exception:
        pass

    # Feature ranking & selection
    selected_features, fs_meta = rank_and_select_features(X_final, y, top_n=30)
    # Keep only selected features for modeling (makes model leaner). If selection returns empty, keep all.
    if selected_features:
        X_model = X_final[selected_features]
    else:
        X_model = X_final.copy()

    # Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_model, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X_model, y, test_size=0.2, random_state=RANDOM_STATE, stratify=None
        )

    # Save a small snapshot of preprocessed data
    try:
        df.head(20).to_csv(os.path.join(OUTPUT_DIR, "dataset_head_after_preprocessing.csv"), index=False)
    except Exception:
        pass

    meta = {
        "scaler": scaler,
        "label_encoders": label_encoders,
        "onehot_columns": onehot_columns,
        "numeric_features": numeric_features,
        "target_col": target_col,
        "transforms_applied": transforms_applied,
        "visuals": visuals,
        "csv_path": csv_path,
        "feature_selection": fs_meta,
    }

    return X_train, X_test, y_train, y_test, meta


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    trained_models = {}
    # If y has only one class, don't attempt to train classification models
    if y_train.nunique() < 2:
        raise RuntimeError("Target y contains only one class; cannot train classifiers.")

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        try:
            plot_feature_importances(model, feature_names=X_train.columns.tolist(), name=name, top_n=min(20, X_train.shape[1]))
        except Exception:
            pass
    return trained_models


def validate_models(trained_models: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series, meta: Dict) -> Dict[str, Dict[str, float]]:
    results = {}
    le_target = meta.get("label_encoders", {}).get(meta.get("target_col", "") + "_target", None)
    class_names = list(le_target.classes_) if le_target is not None else None

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        try:
            plot_confusion_matrix_heatmap(y_test, y_pred, classes=class_names, name=name)
        except Exception:
            pass

    try:
        plot_metrics_comparison(results, prefix="metrics")
    except Exception:
        pass

    try:
        pd.DataFrame(results).T.to_csv(os.path.join(OUTPUT_DIR, "test_results_summary.csv"))
    except Exception:
        pass

    return results


def cross_validate_models(trained_models: Dict[str, object], X: pd.DataFrame, y: pd.Series, k: int = 5) -> Dict[str, np.ndarray]:
    # If y has only one class, cannot cross-validate stratified - return empty arrays
    if y.nunique() < 2:
        return {name: np.array([]) for name in trained_models.keys()}

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}
    for name, model in trained_models.items():
        try:
            # cross_val_score will clone the estimator; safe to pass the fitted model
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
            cv_scores[name] = scores
        except Exception:
            cv_scores[name] = np.array([])

    try:
        plot_cv_boxplot(cv_scores, prefix="cv")
    except Exception:
        pass

    try:
        cv_summary = {k: (float(np.mean(v)) if v.size else np.nan) for k, v in cv_scores.items()}
        pd.Series(cv_summary).to_csv(os.path.join(OUTPUT_DIR, "cv_mean_accuracy_summary.csv"))
    except Exception:
        pass

    return cv_scores


# ----------------------------
# Streamlit UI (auto-run + sidebar navigation)
# ----------------------------
def run_streamlit_app():
    st.set_page_config(page_title="Microplastic Risk Modeling", layout="wide")
    st.sidebar.title("Navigation")
    nav_choice = st.sidebar.radio(
        "Go to",
        ("Upload & Preview", "Visualizations", "Model Results", "Download Outputs"),
    )

    st.title("Predictive Risk Modeling for Microplastic Pollution")
    st.write(
        "Upload a CSV file containing microplastic monitoring data. The pipeline will run automatically after upload. "
        "Outputs (plots & CSVs) are stored in the 'outputs' folder."
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="CSV file with microplastic monitoring data")
    # Track last processed file to avoid re-processing identical file repeatedly
    if "last_processed" not in st.session_state:
        st.session_state["last_processed"] = {"filename": None, "ts": None}

    preview_df = None
    encoding_used = None

    if uploaded_file is not None:
        save_path = os.path.join(INPUT_DIR, uploaded_file.name)
        # Save uploaded file
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved to: {save_path}")

        # Robust preview
        try:
            preview_df, encoding_used = read_csv_preview(save_path, nrows=10)
        except Exception as e:
            st.error(f"Could not preview uploaded CSV: {e}")
            preview_df = None

        # Determine whether to run pipeline: run if new file or not processed recently
        new_file = st.session_state["last_processed"]["filename"] != uploaded_file.name
        if new_file:
            st.session_state["last_processed"] = {"filename": uploaded_file.name, "ts": time.time()}
            # Automatically run pipeline
            with st.spinner("Running full pipeline automatically..."):
                try:
                    X_train, X_test, y_train, y_test, meta = load_and_preprocess_data(save_path)
                    trained_models = train_models(X_train, y_train)
                    test_results = validate_models(trained_models, X_test, y_test, meta)
                    X_full = pd.concat([X_train, X_test], axis=0)
                    y_full = pd.concat([y_train, y_test], axis=0)
                    cv_scores = cross_validate_models(trained_models, X_full, y_full, k=5)
                    # store results in session_state for interactive viewing
                    st.session_state["pipeline_result"] = {
                        "save_path": save_path,
                        "meta": meta,
                        "test_results": test_results,
                        "cv_scores": {k: v.tolist() for k, v in cv_scores.items()},
                    }
                    st.success("Pipeline completed successfully.")
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
                    st.session_state.pop("pipeline_result", None)

    # Sidebar navigation: content rendering
    if nav_choice == "Upload & Preview":
        st.header("Upload & Preview")
        st.info("Upload a CSV above to automatically run the pipeline. Preview shows first 10 rows (robustly read).")
        if preview_df is not None:
            st.write(f"Preview read using encoding: {encoding_used}")
            st.dataframe(preview_df)
        else:
            st.write("No preview available yet. Upload a CSV to see a preview.")
    elif nav_choice == "Visualizations":
        st.header("Visualizations (saved in outputs/)")
        if "pipeline_result" not in st.session_state:
            st.info("No pipeline run available. Upload a CSV to run the pipeline and generate visualizations.")
        else:
            # show saved images (sorted)
            image_paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.png")))
            if image_paths:
                for img_path in image_paths:
                    try:
                        st.image(img_path, caption=os.path.basename(img_path))
                    except Exception:
                        st.write(f"Could not load image {img_path}")
            else:
                st.info("No visualizations found in outputs/ yet.")
    elif nav_choice == "Model Results":
        st.header("Model Results & Metrics")
        if "pipeline_result" not in st.session_state:
            st.info("No pipeline run available. Upload a CSV to run the pipeline.")
        else:
            result = st.session_state["pipeline_result"]
            # Display test results table if exists
            test_results = result.get("test_results", {})
            if test_results:
                df_results = pd.DataFrame(test_results).T
                st.subheader("Test set performance (accuracy, precision, recall, f1)")
                st.dataframe(df_results)
            else:
                st.write("No test results saved.")

            # Display CV scores summary
            cv_scores = result.get("cv_scores", {})
            if cv_scores:
                st.subheader("Cross-validation accuracy per fold")
                for model_name, scores in cv_scores.items():
                    st.write(f"{model_name}: {scores}")
            else:
                st.write("No cross-validation results.")

            # Show FS summary if present
            fs_meta = result.get("meta", {}).get("feature_selection", None)
            if fs_meta:
                st.subheader("Feature selection summary")
                top = fs_meta.get("top_features", [])[:20]
                st.write("Top features (by RandomForest importance):")
                st.write(top)
                st.write("Selected features used for modelling:")
                st.write(fs_meta.get("selected_features", []))

            # Show list of saved CSV summaries
            summary_csv = os.path.join(OUTPUT_DIR, "test_results_summary.csv")
            if os.path.exists(summary_csv):
                with open(summary_csv, "rb") as f:
                    st.download_button("Download test results CSV", data=f, file_name="test_results_summary.csv")
    elif nav_choice == "Download Outputs":
        st.header("Download Outputs")
        st.write(f"All pipeline outputs are saved to: {os.path.abspath(OUTPUT_DIR)}")
        # Create zip and offer download
        zip_path = os.path.join(OUTPUT_DIR, "outputs_bundle.zip")
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in glob.glob(os.path.join(OUTPUT_DIR, "*")):
                    zf.write(file, arcname=os.path.basename(file))
            with open(zip_path, "rb") as f:
                st.download_button("Download all outputs (zip)", data=f, file_name="outputs_bundle.zip")
        except Exception as e:
            st.write(f"Could not create/download outputs zip: {e}")


# ----------------------------
# CLI fallback (unchanged minimal)
# ----------------------------
def parse_cli_csv_arg() -> Optional[str]:
    args = sys.argv[1:]
    if not args:
        return None
    if "--csv" in args:
        i = args.index("--csv")
        if i + 1 < len(args):
            return args[i + 1]
    first = args[0]
    if not first.startswith("-"):
        return first
    return None


def get_csv_path_interactive() -> str:
    cli = parse_cli_csv_arg()
    if cli:
        return cli
    csv_path = input("Enter path or URL to CSV file: ").strip()
    if not csv_path:
        print("No CSV provided. Exiting.")
        sys.exit(1)
    return csv_path


def main_cli():
    csv_path = get_csv_path_interactive()
    if os.path.isfile(csv_path):
        saved_path = csv_path
    else:
        try:
            df_tmp = pd.read_csv(csv_path)
            saved_path = os.path.join(INPUT_DIR, os.path.basename(csv_path) or "uploaded.csv")
            df_tmp.to_csv(saved_path, index=False)
        except Exception:
            print(f"Could not read CSV from {csv_path}. Exiting.")
            sys.exit(1)

    X_train, X_test, y_train, y_test, meta = load_and_preprocess_data(saved_path)
    trained_models = train_models(X_train, y_train)
    test_results = validate_models(trained_models, X_test, y_test, meta)
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)
    cv_scores = cross_validate_models(trained_models, X_full, y_full, k=5)

    print("\nPipeline finished. Outputs saved in:", os.path.abspath(OUTPUT_DIR))


# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        run_streamlit_app()
    else:
        print("Streamlit not available. Running CLI fallback.")
        main_cli()
