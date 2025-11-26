"""
app.py

Predictive Risk Modeling Framework for Microplastic Pollution (Streamlit-enabled)

This version fixes CSV preview decoding errors by using robust CSV-reading fallback:
- Tries pandas default read
- If UnicodeDecodeError occurs, tries common encodings (utf-8, latin1, cp1252)
- If still failing, opens file with errors='replace' into a StringIO and reads via pandas
- Displays the encoding (or 'replaced-invalid-bytes') used for preview in the Streamlit UI

Behavior:
- If Streamlit is available (and you run `streamlit run app.py`) a browser UI will appear.
  * User uploads a CSV file via the UI.
  * The uploaded file is saved to ./inputs/ and then the pipeline runs end-to-end.
  * All plots and summaries are stored in ./outputs/ and displayed in the Streamlit UI.
- If Streamlit is not available or you run the script directly (python app.py), the script falls back
  to a CLI flow that prompts for a CSV path.

Usage:
- Streamlit UI: streamlit run app.py
- CLI: python app.py --csv path/to/MicroPlastic.csv
"""

import os
import sys
import warnings
from typing import Dict, Tuple, List, Optional

import io
import glob
import zipfile

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

# Directories
OUTPUT_DIR = "outputs"
INPUT_DIR = "inputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
sns.set(style="whitegrid")

# Try optional Streamlit import
try:
    import streamlit as st  # type: ignore

    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False


# ----------------------------
# Robust CSV preview helper
# ----------------------------
def read_csv_preview(path: str, nrows: int = 10) -> Tuple[pd.DataFrame, str]:
    """
    Read a CSV file for preview robustly, returning (DataFrame, encoding_used).
    Strategy:
      1) Try pandas.read_csv with no encoding (uses default, typically 'utf-8')
      2) If UnicodeDecodeError, attempt common encodings: 'utf-8', 'latin1', 'cp1252'
      3) If still failing, open file in text mode with errors='replace' and feed to pandas via StringIO.
         In that final case encoding_used = 'replaced-invalid-bytes'
    """
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
            # Some files may raise other parse errors; for preview we still try the next strategy
            last_exception = e
            continue

    # Final fallback: open bytes and decode with errors='replace' to avoid decode failure
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        df = pd.read_csv(io.StringIO(text), nrows=nrows)
        return df, "replaced-invalid-bytes"
    except Exception as e:
        # As ultimate fallback, return an empty DataFrame and raise the last exception for visibility
        raise RuntimeError(f"Failed to read CSV for preview. Last error: {last_exception}; final error: {e}")


# ----------------------------
# Helper plotting functions (unchanged)
# ----------------------------
def save_and_show(fig, filename: str, dpi: int = 150, tight: bool = True):
    """Save figure to outputs directory and close it."""
    path = os.path.join(OUTPUT_DIR, filename)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    # Return saved path
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
        sns.boxplot(x=before[col], ax=axes[0], color="skyblue")
        axes[0].set_title(f"{col} (Before)")
        sns.boxplot(x=after[col], ax=axes[1], color="lightgreen")
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
        sns.histplot(df_before[col].dropna(), kde=True, ax=axes[0], color="cornflowerblue")
        axes[0].set_title(f"{col} BEFORE transformation\nskew={df_before[col].skew():.3f}")
        sns.histplot(df_after[col].dropna(), kde=True, ax=axes[1], color="seagreen")
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
        sns.barplot(x=vc.values, y=vc.index, palette="viridis", ax=ax)
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
        sns.histplot(X_scaled[col], kde=True, ax=ax, color="orchid")
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
    sns.barplot(x=fi.values, y=fi.index, ax=ax, palette="magma")
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
    df = df[metrics]
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
# Core processing functions
# ----------------------------
def load_and_preprocess_data(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
    """
    Load dataset and perform preprocessing (visualizations saved along the way).
    Returns X_train, X_test, y_train, y_test, meta dictionary.
    """
    print("\n[1] Loading dataset...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV from '{csv_path}': {e}")

    print(f" - Loaded dataset with shape: {df.shape}")

    visuals = {"plots": []}

    # Missing values heatmap
    try:
        path = plot_missing_values_heatmap(df, fname="missing_heatmap.png")
        visuals["plots"].append(path)
    except Exception as e:
        print(f"Could not save missing values plot: {e}")

    # Determine target
    if "Risk_Level" in df.columns:
        target_col = "Risk_Level"
    elif "Risk_Type" in df.columns:
        target_col = "Risk_Type"
    else:
        target_col = df.columns[-1]

    # Handle missing values
    num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_all = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in num_cols_all:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols_all:
        df[c] = df[c].fillna("Missing")

    # Outlier handling on specified columns
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
        saved = plot_boxplot_before_after(df_before_outliers, df, outlier_cols_present, prefix="outlier")
        visuals["plots"].extend(saved)
    except Exception as e:
        print(f"Could not save outlier boxplots: {e}")

    # Skewness analysis and log1p transform
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
        df[col] = np.log1p(df[col] + shift)
        transforms_applied[col] = {"shift": shift}
    try:
        saved = plot_histogram_comparison(df_before_skew, df, skewed_cols, prefix="skew")
        visuals["plots"].extend(saved)
    except Exception as e:
        print(f"Could not save skewness histograms: {e}")

    # Categorical encoding
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
        saved = plot_categorical_counts(df, cols_to_encode_present, top_n=8, prefix="cat_before_encoding")
        visuals["plots"].extend(saved)
    except Exception:
        pass

    df_encoded = df.copy()
    label_encoders = {}
    onehot_columns = []
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

    # Prepare target
    if target_col in df_encoded.columns:
        y_raw = df_encoded[target_col]
    else:
        y_raw = df[target_col]
    if y_raw.dtype == "O" or not pd.api.types.is_numeric_dtype(y_raw):
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y_raw.astype(str)), name=target_col)
        label_encoders[target_col + "_target"] = le_target
    else:
        y = pd.Series(y_raw, name=target_col)

    if target_col in df_encoded.columns:
        X_df = df_encoded.drop(columns=[target_col])
    else:
        X_df = df_encoded.copy()

    numeric_features = X_df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    if numeric_features:
        X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_df[numeric_features]), columns=numeric_features, index=X_df.index)
    else:
        X_numeric_scaled = pd.DataFrame(index=X_df.index)

    try:
        saved = plot_scaled_feature_distributions(X_numeric_scaled, prefix="scaled")
        visuals["plots"].extend(saved)
    except Exception:
        pass

    non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X_final = pd.concat([X_numeric_scaled, X_df[non_numeric].reset_index(drop=True)], axis=1)
    else:
        X_final = X_numeric_scaled

    # Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y, test_size=0.2, random_state=RANDOM_STATE, stratify=None
        )

    meta = {
        "scaler": scaler,
        "label_encoders": label_encoders,
        "onehot_columns": onehot_columns,
        "numeric_features": numeric_features,
        "target_col": target_col,
        "transforms_applied": transforms_applied,
        "visuals": visuals,
        "csv_path": csv_path,
    }

    # Save a small snapshot of data (heads) for reference
    try:
        df.head(20).to_csv(os.path.join(OUTPUT_DIR, "dataset_head_after_preprocessing.csv"), index=False)
    except Exception:
        pass

    return X_train, X_test, y_train, y_test, meta


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
    print("\n[TRAIN] Training models...")
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        try:
            plot_feature_importances(model, feature_names=X_train.columns.tolist(), name=name, top_n=12)
        except Exception:
            pass
    return trained_models


def validate_models(trained_models: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series, meta: Dict) -> Dict[str, Dict[str, float]]:
    print("\n[VALIDATE] Validating models...")
    results = {}
    le_target = meta.get("label_encoders", {}).get(meta.get("target_col", "") + "_target", None)
    class_names = list(le_target.classes_) if le_target is not None else None

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        # Save confusion matrix
        try:
            plot_confusion_matrix_heatmap(y_test, y_pred, classes=class_names, name=name)
        except Exception:
            pass
    # Save metrics comparison plot
    try:
        plot_metrics_comparison(results, prefix="metrics")
    except Exception:
        pass

    # Save summary to CSV
    try:
        pd.DataFrame(results).T.to_csv(os.path.join(OUTPUT_DIR, "test_results_summary.csv"))
    except Exception:
        pass

    return results


def cross_validate_models(trained_models: Dict[str, object], X: pd.DataFrame, y: pd.Series, k: int = 5) -> Dict[str, np.ndarray]:
    print("\n[CV] Cross-validating models...")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}
    for name, model in trained_models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
            cv_scores[name] = scores
        except Exception:
            cv_scores[name] = np.array([])
    # Save CV boxplot
    try:
        plot_cv_boxplot(cv_scores, prefix="cv")
    except Exception:
        pass
    # Save CV summary
    try:
        cv_summary = {k: (np.mean(v) if v.size else np.nan) for k, v in cv_scores.items()}
        pd.Series(cv_summary).to_csv(os.path.join(OUTPUT_DIR, "cv_mean_accuracy_summary.csv"))
    except Exception:
        pass
    return cv_scores


# ----------------------------
# UI: Streamlit (with robust preview)
# ----------------------------
def run_streamlit_app():
    st.set_page_config(page_title="Microplastic Risk Modeling", layout="wide")
    st.title("Predictive Risk Modeling for Microplastic Pollution")
    st.write("Upload a CSV file containing microplastic monitoring data. The pipeline will run end-to-end and save outputs to the 'outputs' folder.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload a CSV file to proceed.")
        return

    # Save uploaded file
    save_path = os.path.join(INPUT_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved to: {save_path}")

    # Robust preview
    try:
        df_preview, encoding_used = read_csv_preview(save_path, nrows=10)
        st.subheader("Preview (first 10 rows)")
        st.write(f"Preview read using encoding: {encoding_used}")
        st.dataframe(df_preview)
    except Exception as e:
        st.error(f"Could not preview uploaded CSV: {e}")

    if st.button("Run full pipeline"):
        with st.spinner("Running preprocessing, training, validation, and cross-validation..."):
            try:
                X_train, X_test, y_train, y_test, meta = load_and_preprocess_data(save_path)
                trained_models = train_models(X_train, y_train)
                test_results = validate_models(trained_models, X_test, y_test, meta)
                X_full = pd.concat([X_train, X_test], axis=0)
                y_full = pd.concat([y_train, y_test], axis=0)
                cv_scores = cross_validate_models(trained_models, X_full, y_full, k=5)
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                return

        st.success("Pipeline completed. Outputs saved to the 'outputs' folder.")

        # Show saved plots
        st.subheader("Saved visualizations")
        image_paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.png")))
        if image_paths:
            cols = st.columns(2)
            for i, img_path in enumerate(image_paths):
                try:
                    with open(img_path, "rb") as f:
                        img_bytes = f.read()
                    col = cols[i % 2]
                    col.image(img_bytes, caption=os.path.basename(img_path))
                except Exception:
                    st.write(f"Could not load image {img_path}")
        else:
            st.info("No images found in outputs folder.")

        # Show test results table if available
        result_csv = os.path.join(OUTPUT_DIR, "test_results_summary.csv")
        if os.path.exists(result_csv):
            st.subheader("Test results summary")
            try:
                rr = pd.read_csv(result_csv, index_col=0)
                st.dataframe(rr)
                st.download_button("Download test results CSV", data=open(result_csv, "rb").read(), file_name="test_results_summary.csv")
            except Exception:
                st.write("Could not load test results CSV.")
        else:
            st.info("Test results CSV not found.")

        # Provide a ZIP of outputs for download
        zip_path = os.path.join(OUTPUT_DIR, "outputs_bundle.zip")
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in glob.glob(os.path.join(OUTPUT_DIR, "*")):
                    zf.write(file, arcname=os.path.basename(file))
            with open(zip_path, "rb") as f:
                btn = st.download_button("Download all outputs (zip)", data=f, file_name="outputs_bundle.zip")
        except Exception as e:
            st.write(f"Could not create/download outputs zip: {e}")

        st.info(f"Outputs folder: {os.path.abspath(OUTPUT_DIR)}")


# ----------------------------
# CLI helpers (fallback)
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
        print(f"Using CSV from CLI: {cli}")
        return cli
    csv_path = input("Enter path or URL to CSV file: ").strip()
    if not csv_path:
        print("No CSV provided. Exiting.")
        sys.exit(1)
    return csv_path


# ----------------------------
# Entrypoint
# ----------------------------
def main_cli():
    csv_path = get_csv_path_interactive()
    # If path points to an uploaded file under inputs, leave it; else if it's a URL or remote, pandas will read it.
    if os.path.isfile(csv_path):
        saved_path = csv_path
    else:
        # try to download / save URL with pandas
        try:
            df_tmp = pd.read_csv(csv_path)
            saved_path = os.path.join(INPUT_DIR, os.path.basename(csv_path) or "uploaded.csv")
            df_tmp.to_csv(saved_path, index=False)
            print(f"Downloaded CSV saved to {saved_path}")
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


if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        # When Streamlit runs this script, it will execute top-level code. Use the Streamlit UI instead.
        run_streamlit_app()
    else:
        main_cli()
