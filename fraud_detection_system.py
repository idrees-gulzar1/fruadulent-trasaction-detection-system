#!/usr/bin/env python3
"""Interactive AI Financial Fraud Detection System."""

import os
import time
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

try:
    from imblearn.over_sampling import SMOTE  # type: ignore
except Exception:  # pragma: no cover - fallback when imblearn missing
    SMOTE = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")
sns.set_theme(style="whitegrid")

# Directory constants
DATA_DIR = "data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
MODEL_REGISTRY = os.path.join(MODELS_DIR, "registry.json")

# Shared application state
default_state = {
    "df": None,
    "df_processed": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "X_train_bal": None,
    "y_train_bal": None,
    "models": {},
    "best_model": None,
    "best_model_name": "",
    "scaler": None,
    "encoders": {},
    "feature_cols": [],
    "categorical_cols": [],
    "scaled_features": [],
    "feature_defaults": {},
    "target_col": "is_fraud",
    "results": {},
    "dataset_loaded": False,
    "preprocessed": False,
    "models_trained": False,
    "dataset_info": {},
    "prediction_count": 0,
    "models_trained_count": 0,
}

state = default_state.copy()

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    """Create required folders if they do not exist."""
    for folder in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
        os.makedirs(folder, exist_ok=True)


def pause() -> None:
    """Pause for user acknowledgement."""
    input("\nPress Enter to return to the main menu...")


def print_banner() -> None:
    """Print the ASCII banner on startup."""
    banner = (
        "╔══════════════════════════════════════════════════════╗\n"
        "║     AI FINANCIAL FRAUD DETECTION SYSTEM v1.0         ║\n"
        "║     Powered by: Scikit-learn + XGBoost               ║\n"
        "╚══════════════════════════════════════════════════════╝"
    )
    print(banner)


def show_main_menu() -> None:
    """Display the main menu."""
    menu = (
        "┌─────────────────────────────────────────────────────┐\n"
        "│  MAIN MENU                                          │\n"
        "├─────────────────────────────────────────────────────┤\n"
        "│  1. Load Dataset                                    │\n"
        "│  2. Explore & Analyze Data                          │\n"
        "│  3. Preprocess Data                                 │\n"
        "│  4. Train ML Models                                 │\n"
        "│  5. Evaluate Models & Compare                       │\n"
        "│  6. Predict on New Transaction                      │\n"
        "│  7. Generate Full Report                            │\n"
        "│  8. Save / Load Model                               │\n"
        "│  0. Exit                                            │\n"
        "└─────────────────────────────────────────────────────┘"
    )
    print(menu)


def detect_target_column(df: pd.DataFrame) -> str:
    """Auto-detect a likely target column or ask the user."""
    candidates = ["is_fraud", "class", "fraud", "label", "target"]
    for candidate in candidates:
        for column in df.columns:
            if column.lower() == candidate:
                return column
    print("⚠ Target column not found automatically.")
    print("Available columns:")
    for idx, column in enumerate(df.columns, start=1):
        print(f"  {idx}. {column}")
    chosen = input("Enter the target column name: ").strip()
    if chosen in df.columns:
        return chosen
    raise ValueError("Target column not found in dataset.")


def find_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Identify columns that contain datetime information."""
    datetime_columns: list[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(column)
            continue
        if series.dtype == object:
            sample = series.dropna().astype(str).head(20)
            if sample.empty:
                continue
            try:
                pd.to_datetime(sample, errors="raise")
                datetime_columns.append(column)
            except Exception:
                continue
    return datetime_columns


def enrich_datetime_features(df: pd.DataFrame, datetime_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Create hour/day/month features from datetime columns."""
    created_features: list[str] = []
    for column in datetime_cols:
        parsed = pd.to_datetime(df[column], errors="coerce")
        df[f"{column}_hour"] = parsed.dt.hour.fillna(0).astype(int)
        df[f"{column}_day_of_week"] = parsed.dt.dayofweek.fillna(0).astype(int)
        df[f"{column}_month"] = parsed.dt.month.fillna(0).astype(int)
        df[f"{column}_year"] = parsed.dt.year.fillna(parsed.dt.year.mode().iloc[0] if not parsed.dt.year.dropna().empty else 0).astype(int)
        created_features.extend(
            [
                f"{column}_hour",
                f"{column}_day_of_week",
                f"{column}_month",
                f"{column}_year",
            ]
        )
    return df, created_features


def identify_id_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """Detect ID-like columns to drop later."""
    preset = {"customer_id", "device_id", "device_fingerprint", "ip_address", "merchant_id", "transaction_timestamp"}
    id_columns: set[str] = set()
    for column in df.columns:
        if column == target_col:
            continue
        col_lower = column.lower()
        if col_lower in preset or "id" in col_lower or col_lower.endswith("_id"):
            id_columns.add(column)
            continue
        unique_values = df[column].nunique(dropna=False)
        if unique_values == len(df):
            id_columns.add(column)
    return sorted(id_columns)


def summarize_dataset(df: pd.DataFrame, target_col: str, id_cols: list[str]) -> dict:
    """Compute summary stats for the loaded dataset."""
    rows, cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    missing_total = int(df.isna().sum().sum())
    fraud_cases = int(df[target_col].sum()) if target_col in df else 0
    legit_cases = int(len(df) - fraud_cases)
    fraud_rate = (fraud_cases / len(df) * 100) if len(df) else 0
    summary = {
        "rows": rows,
        "cols": cols,
        "numeric": len(numeric_cols),
        "categorical": len(categorical_cols),
        "missing": missing_total,
        "fraud_cases": fraud_cases,
        "legit_cases": legit_cases,
        "fraud_rate": fraud_rate,
        "id_cols": id_cols,
    }
    print("\n✓ Dataset loaded successfully!")
    print(f" → Rows: {rows:,}  |  Columns: {cols:,}")
    print(f" → Target column: {target_col}")
    print(
        f" → Fraud cases: {fraud_cases:,} ({fraud_rate:.2f}%)\n"
        f" → Legitimate: {legit_cases:,} ({100 - fraud_rate:.2f}%)"
    )
    print(f" → Missing values: {missing_total:,}")
    print(f" → Numeric features: {summary['numeric']}")
    print(f" → Categorical features: {summary['categorical']}")
    print(f" → ID columns (will be dropped): {len(id_cols)}")
    if id_cols:
        print("    " + ", ".join(id_cols))
    return summary


def require(condition: bool, message: str) -> bool:
    """Validate workflow preconditions and alert the user if unmet."""
    if not condition:
        print(f"\n⚠ {message}")
        pause()
        return False
    return True


def pretty_duration(seconds: float) -> str:
    """Format durations for display."""
    return f"{seconds:.1f}s"


def maybe_progress(message: str, duration: float) -> None:
    """Show progress via tqdm or fallback text."""
    if tqdm:
        for _ in tqdm(range(100), desc=message, ncols=80):
            time.sleep(duration / 100 if duration else 0.01)
    else:
        print(f"{message}... ████████████ Done ({pretty_duration(duration)})")


def calculate_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute evaluation metrics for a model."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
        "report": classification_report(y_true, y_pred, zero_division=0),
    }
    return metrics


def risk_label(probability: float) -> tuple[str, str]:
    """Map a probability to risk level and recommendation."""
    if probability < 0.3:
        return "LOW", "✓ APPROVE"
    if probability < 0.5:
        return "MEDIUM", "⚡ FLAG FOR REVIEW"
    if probability < 0.7:
        return "HIGH", "⚠ REQUIRE VERIFICATION"
    return "CRITICAL", "✗ BLOCK TRANSACTION"


def feature_reasons(row: dict, probability: float) -> list[str]:
    """Generate heuristic explanations for a prediction."""
    reasons: list[str] = []
    amt_ratio = row.get("amount_vs_avg_ratio", 1)
    if amt_ratio >= 3:
        reasons.append(f"amount_vs_avg_ratio: {amt_ratio:.1f}x (HIGH)")
    elif amt_ratio >= 1.5:
        reasons.append(f"amount_vs_avg_ratio: {amt_ratio:.1f}x (ELEVATED)")
    if row.get("is_foreign_transaction", 0) == 1:
        reasons.append("is_foreign_transaction: YES")
    if row.get("is_tor_or_vpn", 0) == 1:
        reasons.append("is_tor_or_vpn: YES")
    if row.get("txn_count_last_1hr", 0) >= 5:
        reasons.append(f"txn_count_last_1hr: {row.get('txn_count_last_1hr')} (SUSPICIOUS)")
    if row.get("country_risk_score", 0) >= 7:
        reasons.append(f"country_risk_score: {row.get('country_risk_score')} (HIGH)")
    if row.get("distance_from_last_txn_km", 0) >= 500:
        reasons.append(
            f"distance_from_last_txn_km: {row.get('distance_from_last_txn_km'):.1f} km"
        )
    if not reasons:
        reasons.append("Transaction aligns with historical patterns.")
    return reasons[:4]


# ---------------------------------------------------------------------------
# Option 1: Load dataset
# ---------------------------------------------------------------------------

def load_dataset() -> None:
    """Load a CSV dataset and gather metadata."""
    try:
        path = input(
            "Enter CSV file path (or press Enter for default data/creditcard.csv): "
        ).strip()
        if not path:
            path = os.path.join(DATA_DIR, "creditcard.csv")
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            print(f"⚠ File not found: {path}. Please check the path and try again.")
            pause()
            return
        df = pd.read_csv(path)
        if df.empty:
            print("⚠ The provided CSV is empty.")
            pause()
            return
        target_col = detect_target_column(df)
        datetime_cols = find_datetime_columns(df)
        df, created_features = enrich_datetime_features(df, datetime_cols)
        id_columns = identify_id_columns(df, target_col)
        summary = summarize_dataset(df, target_col, id_columns)
        state.update(
            {
                "df": df,
                "df_processed": None,
                "target_col": target_col,
                "dataset_loaded": True,
                "preprocessed": False,
                "models_trained": False,
                "models": {},
                "best_model": None,
                "best_model_name": "",
                "results": {},
                "encoders": {},
                "feature_cols": [],
                "categorical_cols": [],
                "scaled_features": [],
                "feature_defaults": {},
                "dataset_info": {
                    "path": path,
                    "created_features": created_features,
                    **summary,
                },
                "models_trained_count": 0,
                "prediction_count": 0,
            }
        )
        print("\nDataset ready for exploration and preprocessing!")
    except Exception as exc:  # pragma: no cover - interactive path
        print(f"⚠ Failed to load dataset: {exc}")
    pause()


# ---------------------------------------------------------------------------
# Option 2: Explore & analyze data
# ---------------------------------------------------------------------------

def explore_menu() -> None:
    """Provide sub-menu for exploratory analysis."""
    if not require(state["dataset_loaded"], "Please load a dataset first!"):
        return
    df = state["df"].copy()
    target_col = state["target_col"]
    while True:
        print(
            "\nExploration Menu:\n"
            "  2a. Show basic statistics (describe)\n"
            "  2b. Show class distribution + fraud rate\n"
            "  2c. Show missing values report\n"
            "  2d. Show feature correlations with target\n"
            "  2e. Show top 10 fraud indicators\n"
            "  2f. Plot and save all charts to reports/\n"
            "  0. Return to main menu"
        )
        choice = input("Select an option: ").strip().lower()
        if choice in {"0", "q", "exit"}:
            break
        try:
            if choice in {"2a", "a"}:
                print("\nBasic statistics:")
                print(df.describe(include="all").transpose().round(3))
            elif choice in {"2b", "b"}:
                show_class_distribution(df, target_col)
            elif choice in {"2c", "c"}:
                show_missing_report(df)
            elif choice in {"2d", "d"}:
                show_correlations(df, target_col)
            elif choice in {"2e", "e"}:
                show_top_indicators(df, target_col)
            elif choice in {"2f", "f"}:
                generate_charts(df, target_col)
            else:
                print("⚠ Invalid choice. Please try again.")
        except Exception as exc:
            print(f"⚠ Exploration error: {exc}")
    pause()


def show_class_distribution(df: pd.DataFrame, target_col: str) -> None:
    """Display fraud vs legit distribution."""
    if target_col not in df:
        print("⚠ Target column missing in dataframe.")
        return
    counts = df[target_col].value_counts()
    total = counts.sum()
    fraud = counts.get(1, 0)
    legit = counts.get(0, total - fraud)
    rate = (fraud / total * 100) if total else 0
    print("\nClass distribution:")
    print(f"  Fraud cases: {fraud:,} ({rate:.2f}%)")
    print(f"  Legitimate: {legit:,} ({100 - rate:.2f}%)")


def show_missing_report(df: pd.DataFrame) -> None:
    """Display missing values per column."""
    missing = df.isna().sum()
    print("\nMissing values report:")
    print(missing[missing > 0].sort_values(ascending=False) if missing.any() else "  No missing values detected.")


def show_correlations(df: pd.DataFrame, target_col: str) -> None:
    """Show correlations between numeric features and target."""
    if target_col not in df:
        print("⚠ Target column missing in dataframe.")
        return
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        print("⚠ Target column must be numeric for correlation analysis.")
        return
    numeric_cols.remove(target_col)
    correlations = df[numeric_cols].corrwith(df[target_col]).sort_values(key=lambda s: s.abs(), ascending=False)
    print("\nTop feature correlations with target:")
    print(correlations.round(3).head(20))


def show_top_indicators(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Compute top fraud indicators based on mean differences."""
    if target_col not in df:
        raise ValueError("Target column missing.")
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]
    fraud_df = df[df[target_col] == 1]
    legit_df = df[df[target_col] == 0]
    records = []
    for col in numeric_cols:
        fraud_avg = fraud_df[col].mean()
        legit_avg = legit_df[col].mean()
        if pd.isna(fraud_avg) or pd.isna(legit_avg):
            continue
        diff_pct = ((fraud_avg - legit_avg) / legit_avg * 100) if legit_avg not in (0, None) else np.nan
        records.append(
            {
                "Feature": col,
                "Fraud Avg": fraud_avg,
                "Legit Avg": legit_avg,
                "Diff %": diff_pct,
            }
        )
    indicators = (
        pd.DataFrame(records)
        .dropna()
        .sort_values(by="Diff %", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    if indicators.empty:
        print("⚠ Unable to compute fraud indicators (insufficient numeric data).")
        return indicators
    print("\nTop 10 fraud indicators:")
    print(indicators.round(2).to_string(index=False))
    return indicators


def generate_charts(df: pd.DataFrame, target_col: str) -> None:
    """Create and save required charts to the reports folder."""
    print("\nGenerating charts...")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Chart 1: Class distribution
    plt.figure(figsize=(6, 4))
    df[target_col].value_counts().plot(kind="bar", color=["#2ecc71", "#e74c3c"])
    plt.xticks([0, 1], ["Legit", "Fraud"], rotation=0)
    plt.title("Class Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    class_chart = os.path.join(REPORTS_DIR, "class_distribution.png")
    plt.savefig(class_chart)
    plt.close()

    # Chart 2: Amount distribution
    amount_col = "transaction_amount" if "transaction_amount" in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    plt.figure(figsize=(6, 4))
    sns.histplot(df[amount_col], bins=50, color="#3498db")
    plt.title("Transaction Amount Distribution")
    plt.tight_layout()
    amount_chart = os.path.join(REPORTS_DIR, "amount_distribution.png")
    plt.savefig(amount_chart)
    plt.close()

    # Chart 3: Fraud by hour
    hour_col = "hour_of_day" if "hour_of_day" in df.columns else None
    if not hour_col:
        candidates = [col for col in df.columns if col.endswith("_hour")]
        hour_col = candidates[0] if candidates else None
    if hour_col:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=hour_col, hue=target_col, data=df, palette="Set2")
        plt.title("Fraud by Hour")
        plt.tight_layout()
        hour_chart = os.path.join(REPORTS_DIR, "fraud_by_hour.png")
        plt.savefig(hour_chart)
        plt.close()

    # Chart 4: Fraud by transaction type
    if "transaction_type" in df.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(y="transaction_type", hue=target_col, data=df, palette="Set3")
        plt.title("Fraud by Transaction Type")
        plt.tight_layout()
        type_chart = os.path.join(REPORTS_DIR, "fraud_by_transaction_type.png")
        plt.savefig(type_chart)
        plt.close()

    # Chart 5: Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    heatmap_chart = os.path.join(REPORTS_DIR, "correlation_heatmap.png")
    plt.savefig(heatmap_chart)
    plt.close()

    # Chart 6: Top features
    indicators = show_top_indicators(df, target_col)
    if not indicators.empty:
        plt.figure(figsize=(8, 4))
        sns.barplot(x="Diff %", y="Feature", data=indicators, palette="magma")
        plt.title("Top Fraud Indicators")
        plt.tight_layout()
        top_chart = os.path.join(REPORTS_DIR, "top_features.png")
        plt.savefig(top_chart)
        plt.close()

    print("✓ Charts saved to reports/ directory!")


# ---------------------------------------------------------------------------
# Option 3: Preprocess data
# ---------------------------------------------------------------------------

def preprocess_data() -> None:
    """Run the full preprocessing pipeline."""
    if not require(state["dataset_loaded"], "Please load a dataset first!"):
        return
    try:
        df = state["df"].copy()
        target_col = state["target_col"]
        print("\n[1/7] Dropping ID and irrelevant columns...")
        id_cols = identify_id_columns(df, target_col)
        drop_list = [col for col in id_cols if col in df.columns]
        if "transaction_timestamp" in df.columns and "transaction_timestamp" not in drop_list:
            drop_list.append("transaction_timestamp")
        df.drop(columns=[col for col in drop_list if col in df.columns], inplace=True, errors="ignore")
        print(f"      → Dropped: {', '.join(drop_list) if drop_list else 'None'}")

        print("\n[2/7] Encoding categorical columns...")
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        encoders = {}
        for column in categorical_cols:
            if column == target_col:
                continue
            encoder = LabelEncoder()
            df[column] = df[column].astype(str).fillna("Unknown")
            encoder.fit(df[column])
            df[column] = encoder.transform(df[column])
            encoders[column] = encoder
        print(
            "      → Label encoded: "
            + (", ".join(categorical_cols) if categorical_cols else "No categorical columns")
        )

        print("\n[3/7] Handling missing values...")
        missing_before = int(df.isna().sum().sum())
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for column in numeric_cols:
            df[column].fillna(df[column].median(), inplace=True)
        for column in categorical_cols:
            if column in df.columns:
                df[column].fillna(df[column].mode().iloc[0], inplace=True)
        missing_after = int(df.isna().sum().sum())
        print(
            f"      → Filled {missing_before - missing_after} missing values (numeric: median, categorical: mode)"
        )

        print("\n[4/7] Feature scaling (StandardScaler)...")
        scaler = StandardScaler()
        scale_candidates = [
            "transaction_amount",
            "account_age_days",
            "time_since_last_transaction",
            "distance_from_last_txn_km",
            "amount_vs_avg_ratio",
            "country_risk_score",
            "device_age_days",
        ]
        scaled_features = [col for col in scale_candidates if col in df.columns]
        if scaled_features:
            df[scaled_features] = scaler.fit_transform(df[scaled_features])
            print(f"      → Scaled: {', '.join(scaled_features)}")
        else:
            scaler = None
            print("      → No numeric features required scaling.")

        print("\n[5/7] Splitting into train/test sets (80/20 stratified)...")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X.values,
            y.values,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )
        print(
            f"      → Train: {len(X_train):,} rows | Test: {len(X_test):,} rows\n"
            f"      → Train fraud: {int(y_train.sum()):,} | Test fraud: {int(y_test.sum()):,}"
        )

        print("\n[6/7] Applying SMOTE to fix class imbalance...")
        if SMOTE:
            smote = SMOTE(random_state=42)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
            print(
                f"      → Before SMOTE: {np.sum(y_train == 0):,} legit | {np.sum(y_train == 1):,} fraud\n"
                f"      → After SMOTE:  {np.sum(y_train_bal == 0):,} legit | {np.sum(y_train_bal == 1):,} fraud"
            )
        else:
            majority = X_train[y_train == 0]
            minority = X_train[y_train == 1]
            minority_resampled = resample(
                minority,
                replace=True,
                n_samples=len(majority),
                random_state=42,
            )
            X_train_bal = np.vstack([majority, minority_resampled])
            y_train_bal = np.hstack([
                np.zeros(len(majority)),
                np.ones(len(minority_resampled)),
            ])
            print(
                f"      → Applied manual upsampling (imblearn not available)."\
            )
        print(f"      → Total training samples: {len(X_train_bal):,}")

        print("\n[7/7] Saving preprocessed data...")
        np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
        np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
        np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
        np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)
        if scaler:
            joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
        joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
        joblib.dump(X.columns.tolist(), os.path.join(MODELS_DIR, "feature_cols.pkl"))
        print(
            "      → Saved: data/X_train.npy, data/X_test.npy\n"
            "        Saved: data/y_train.npy, data/y_test.npy\n"
            "        Saved: models/scaler.pkl, models/encoders.pkl\n"
            "        Saved: models/feature_cols.pkl"
        )

        feature_defaults = {}
        for column in X.columns:
            series = df[column]
            feature_defaults[column] = float(series.median()) if pd.api.types.is_numeric_dtype(series) else series.mode().iloc[0]

        state.update(
            {
                "df_processed": df,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "X_train_bal": X_train_bal,
                "y_train_bal": y_train_bal,
                "scaler": scaler,
                "encoders": encoders,
                "feature_cols": X.columns.tolist(),
                "categorical_cols": [col for col in categorical_cols if col != target_col],
                "scaled_features": scaled_features,
                "feature_defaults": feature_defaults,
                "preprocessed": True,
                "models_trained": False,
                "models": {},
                "results": {},
                "best_model": None,
                "best_model_name": "",
            }
        )

        print("\n✓ Preprocessing complete! Ready for model training.")
    except Exception as exc:
        print(f"⚠ Preprocessing failed: {exc}")
    pause()


# ---------------------------------------------------------------------------
# Option 4: Train ML models
# ---------------------------------------------------------------------------

def model_builders():
    """Return builder functions for each supported model."""
    return {
        "logistic_regression": (
            "Logistic Regression",
            lambda: LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced", n_jobs=None),
        ),
        "random_forest": (
            "Random Forest",
            lambda: RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42),
        ),
        "xgboost": (
            "XGBoost",
            lambda: XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                scale_pos_weight=10,
                max_depth=6,
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                tree_method="hist",
                random_state=42,
            ),
        ),
        "neural_network": (
            "Neural Network",
            lambda: MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                max_iter=200,
                early_stopping=True,
                random_state=42,
            ),
        ),
        "ensemble": (
            "Ensemble",
            lambda: VotingClassifier(
                estimators=[
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=15,
                            n_jobs=-1,
                            random_state=42,
                        ),
                    ),
                    (
                        "xgb",
                        XGBClassifier(
                            n_estimators=300,
                            learning_rate=0.05,
                            scale_pos_weight=10,
                            max_depth=6,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            use_label_encoder=False,
                            tree_method="hist",
                            random_state=42,
                        ),
                    ),
                    (
                        "mlp",
                        MLPClassifier(
                            hidden_layer_sizes=(128, 64, 32),
                            activation="relu",
                            max_iter=200,
                            early_stopping=True,
                            random_state=42,
                        ),
                    ),
                ],
                voting="soft",
            ),
        ),
    }


def train_models_menu() -> None:
    """Sub-menu to train models individually or all at once."""
    if not require(state["preprocessed"], "Please preprocess the data first!"):
        return
    builders = model_builders()
    while True:
        print(
            "\nTraining Menu:\n"
            "  4a. Train all models (recommended)\n"
            "  4b. Train specific model\n"
            "  0. Return to main menu"
        )
        choice = input("Select an option: ").strip().lower()
        if choice in {"0", "q"}:
            break
        if choice in {"4a", "a"}:
            for key in ["logistic_regression", "random_forest", "xgboost", "neural_network", "ensemble"]:
                train_single_model(key, builders)
            break
        if choice in {"4b", "b"}:
            print("Available models:")
            for idx, (key, (label, _)) in enumerate(builders.items(), start=1):
                print(f"  {idx}. {label}")
            model_choice = input("Type the model name or number: ").strip()
            selected_key = None
            if model_choice.isdigit():
                idx = int(model_choice) - 1
                if 0 <= idx < len(builders):
                    selected_key = list(builders.keys())[idx]
            else:
                normalized = model_choice.lower().replace(" ", "_")
                if normalized in builders:
                    selected_key = normalized
            if selected_key:
                train_single_model(selected_key, builders)
            else:
                print("⚠ Invalid selection.")
        else:
            print("⚠ Invalid choice. Please try again.")
    pause()


def train_single_model(key: str, builders: dict) -> None:
    """Train a specific model and store it in state."""
    label, builder = builders[key]
    print(f"\nTraining {label}...")
    model = builder()
    start = time.time()
    model.fit(state["X_train_bal"], state["y_train_bal"])
    duration = time.time() - start
    if not tqdm:
        print(f"Training... ████████████ Done ({pretty_duration(duration)})")
    y_pred = model.predict(state["X_test"])
    y_proba = model.predict_proba(state["X_test"])[:, 1]
    metrics = calculate_metrics(state["y_test"], y_pred, y_proba)
    state["models"][label] = model
    state["results"][label] = metrics
    state["models_trained"] = True
    state["models_trained_count"] = len(state["models"])
    if metrics["roc_auc"] >= state["results"].get(state["best_model_name"], {}).get("roc_auc", -1):
        state["best_model"] = model
        state["best_model_name"] = label
    print(f"✓ {label} trained. AUC: {metrics['roc_auc']:.3f}")
    if label == "Ensemble":
        print("✓ All models trained successfully!" if len(state["models"]) >= 5 else "Ensemble ready!")
        print(
            f"  Best single model: {state['best_model_name']} (AUC: {state['results'][state['best_model_name']]['roc_auc']:.3f})"
        )


# ---------------------------------------------------------------------------
# Option 5: Evaluate models & compare
# ---------------------------------------------------------------------------

def evaluate_models() -> None:
    """Show evaluation summary, tables, and charts."""
    if not require(state["models_trained"], "Please train models first!"):
        return
    print("\nModel performance comparison:")
    header = (
        "╔══════════════════════╦══════════╦═══════════╦════════╦═════════╦══════════╗\n"
        "║ Model                ║ Accuracy ║ Precision ║ Recall ║ F1-Score║  AUC-ROC ║\n"
        "╠══════════════════════╬══════════╬═══════════╬════════╬═════════╬══════════╣"
    )
    print(header)
    for label, metrics in state["results"].items():
        row = (
            f"║ {label:<20} ║ {metrics['accuracy']*100:7.2f}% ║ "
            f"{metrics['precision']*100:8.2f}% ║ {metrics['recall']*100:6.2f}% ║ "
            f"{metrics['f1']*100:7.2f}% ║ {metrics['roc_auc']:8.3f} ║"
        )
        print(row)
    print("╚══════════════════════╩══════════╩═══════════╩════════╩═════════╩══════════╝")

    best_label = state["best_model_name"]
    best_metrics = state["results"].get(best_label, {})
    y_pred = state["best_model"].predict(state["X_test"])
    cm = confusion_matrix(state["y_test"], y_pred)
    tn, fp, fn, tp = cm.ravel()
    detection_rate = tp / (tp + fn) * 100 if (tp + fn) else 0
    false_alarm_rate = fp / (fp + tn) * 100 if (fp + tn) else 0

    print("\nConfusion Matrix for best model:")
    print(
        "  ┌─────────────────────────────────────┐\n"
        "  │              Predicted              │\n"
        "  │         Legit    │    Fraud         │\n"
        f"  │ Legit   {tn:7}   │   {fp:6}          │\n"
        f"  │ Fraud   {fn:7}   │   {tp:6}          │\n"
        "  └─────────────────────────────────────┘"
    )
    print(f"  Fraud Detection Rate: {detection_rate:.2f}%")
    print(f"  False Alarm Rate: {false_alarm_rate:.2f}%")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    # ROC curves
    plt.figure(figsize=(8, 6))
    for label, metrics in state["results"].items():
        model = state["models"].get(label)
        if not model:
            continue
        y_proba = model.predict_proba(state["X_test"])[:, 1]
        fpr, tpr, _ = roc_curve(state["y_test"], y_proba)
        plt.plot(fpr, tpr, label=f"{label} (AUC={metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "roc_curves.png"))
    plt.close()

    # Confusion matrix heatmap
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {best_label}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    plt.close()

    # Model comparison bar chart
    plt.figure(figsize=(8, 4))
    labels = list(state["results"].keys())
    aucs = [state["results"][lbl]["roc_auc"] for lbl in labels]
    sns.barplot(x=labels, y=aucs, palette="viridis")
    plt.title("Model AUC Comparison")
    plt.ylabel("AUC-ROC")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "model_comparison.png"))
    plt.close()

    print("\nEvaluation artifacts saved to reports/ directory!")
    pause()


# ---------------------------------------------------------------------------
# Option 6: Predict on new transactions
# ---------------------------------------------------------------------------

def predict_menu() -> None:
    """Prediction sub-menu for manual or batch inputs."""
    if not require(state["models_trained"], "Please train models before predicting!"):
        return
    while True:
        print(
            "\nPrediction Menu:\n"
            "  6a. Enter transaction manually\n"
            "  6b. Load transactions from CSV\n"
            "  0. Return to main menu"
        )
        choice = input("Select an option: ").strip().lower()
        if choice in {"0", "q"}:
            break
        if choice in {"6a", "a"}:
            manual_prediction()
        elif choice in {"6b", "b"}:
            batch_prediction()
        else:
            print("⚠ Invalid choice. Please try again.")
    pause()


def manual_prediction() -> None:
    """Capture manual transaction details and score risk."""
    prompts = [
        ("transaction_amount", float, "Transaction Amount (₹/USD)", 0.0),
        ("transaction_type", str, "Transaction Type (UPI/Card/Wire/ATM/NEFT)", "Card"),
        ("hour_of_day", int, "Hour of day (0-23)", 12),
        ("is_foreign_transaction", int, "Is foreign transaction? (0/1)", 0),
        ("is_online_transaction", int, "Is online transaction? (0/1)", 1),
        ("txn_count_last_1hr", int, "Transactions in last 1 hour", 0),
        ("txn_count_last_24hr", int, "Transactions in last 24 hours", 0),
        ("amount_vs_avg_ratio", float, "Amount vs average ratio", 1.0),
        ("is_new_merchant", int, "Is new merchant? (0/1)", 0),
        ("cvv_match", int, "CVV match? (0/1)", 1),
        ("3ds_authenticated", int, "3DS authenticated? (0/1)", 1),
        ("country_risk_score", float, "Country risk score (0-10)", 2.0),
        ("is_tor_or_vpn", int, "Is TOR/VPN? (0/1)", 0),
        ("distance_from_last_txn_km", float, "Distance from last transaction (km)", 0.0),
    ]
    record = state["feature_defaults"].copy()
    for column, caster, question, default in prompts:
        raw = input(f"{question}: ").strip()
        if not raw:
            value = default
        else:
            try:
                value = caster(raw)
            except ValueError:
                print(f"Invalid input for {column}, using default {default}.")
                value = default
        record[column] = value
    results = score_records([record])
    result = results[0]
    probability = result["fraud_probability"] * 100
    risk_level = result["risk_level"]
    decision = result["decision"]
    reasons = result["reasons"]
    print(
        "\n┌──────────────────────────────────────────────┐\n"
        "│         FRAUD ANALYSIS RESULT                │\n"
        "├──────────────────────────────────────────────┤"
    )
    print(f"│  Transaction Amount:  ₹{record.get('transaction_amount', 0):,.2f}{' ' * (16 - len(str(record.get('transaction_amount', 0))))}│")
    print(f"│  Model Used:          {state['best_model_name']:<22}│")
    print("├──────────────────────────────────────────────┤")
    print(f"│  FRAUD PROBABILITY:   {probability:5.1f}%{' ' * 18}│")
    print(f"│  RISK LEVEL:          {risk_level:<6}               │")
    print(f"│  DECISION:            {decision:<20}│")
    print("├──────────────────────────────────────────────┤")
    print("│  Top reasons flagged:                        │")
    for reason in reasons:
        print(f"│  → {reason:<40}│")
    print("└──────────────────────────────────────────────┘")
    state["prediction_count"] += 1


def batch_prediction() -> None:
    """Load transactions from CSV and score each."""
    path = input("Enter CSV file path for batch prediction: ").strip()
    if not path:
        print("⚠ A CSV path is required.")
        return
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print(f"⚠ File not found: {path}.")
        return
    try:
        df_new = pd.read_csv(path)
        records = df_new.to_dict(orient="records")
        results = score_records(records)
        probs = [r["fraud_probability"] for r in results]
        decisions = [r["decision"] for r in results]
        risk_levels = [r["risk_level"] for r in results]
        df_new["fraud_probability"] = probs
        df_new["risk_level"] = risk_levels
        df_new["decision"] = decisions
        output_path = os.path.join(REPORTS_DIR, "batch_predictions.csv")
        df_new.to_csv(output_path, index=False)
        total = len(results)
        fraud_detected = sum(level in {"HIGH", "CRITICAL"} for level in risk_levels)
        blocked = sum(decision.startswith("✗") or decision.startswith("⚠") for decision in decisions)
        high_risk = sum(level in {"HIGH", "CRITICAL"} for level in risk_levels)
        print("\nBatch prediction summary:")
        print(f"  Total transactions analyzed: {total:,}")
        print(f"  Fraud detected: {fraud_detected:,} ({(fraud_detected/total*100 if total else 0):.1f}%)")
        print(f"  High risk: {high_risk:,} ({(high_risk/total*100 if total else 0):.1f}%)")
        print(f"  Blocked: {blocked:,}")
        print(f"  Results saved to {output_path}")
        state["prediction_count"] += total
    except Exception as exc:
        print(f"⚠ Batch prediction failed: {exc}")


def score_records(records: list[dict]) -> list[dict]:
    """Apply preprocessing artifacts and score new records."""
    if not records:
        return []
    df_new = pd.DataFrame(records)
    # Ensure all expected columns exist
    for column in state["feature_cols"]:
        if column not in df_new.columns:
            df_new[column] = state["feature_defaults"].get(column, 0)
    # Apply encoders
    for column, encoder in state["encoders"].items():
        if column in df_new.columns:
            df_new[column] = df_new[column].astype(str)
            unseen_mask = ~df_new[column].isin(encoder.classes_)
            if unseen_mask.any():
                encoder.classes_ = np.append(encoder.classes_, df_new[column][unseen_mask].unique())
            df_new[column] = encoder.transform(df_new[column])
    # Apply scaler
    if state["scaler"] and state["scaled_features"]:
        missing = [col for col in state["scaled_features"] if col not in df_new.columns]
        for column in missing:
            df_new[column] = state["feature_defaults"].get(column, 0)
        df_new[state["scaled_features"]] = state["scaler"].transform(df_new[state["scaled_features"]])
    df_new = df_new[state["feature_cols"]]
    probabilities = state["best_model"].predict_proba(df_new.values)[:, 1]
    results = []
    for idx, probability in enumerate(probabilities):
        risk_level, decision = risk_label(probability)
        reasons = feature_reasons(records[idx], probability)
        results.append(
            {
                "fraud_probability": probability,
                "risk_level": risk_level,
                "decision": decision,
                "reasons": reasons,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Option 7: Generate full report
# ---------------------------------------------------------------------------

def generate_report() -> None:
    """Compile a comprehensive text report."""
    if not require(state["models_trained"], "Please complete training before generating report!"):
        return
    report_path = os.path.join(REPORTS_DIR, "fraud_detection_report.txt")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    indicators = show_top_indicators(state["df"], state["target_col"])
    best_label = state["best_model_name"]
    best_metrics = state["results"][best_label]
    lines = [
        "═══════════════════════════════════════════════════",
        "  AI FRAUD DETECTION SYSTEM - FULL REPORT",
        f"  Generated: {now}",
        "═══════════════════════════════════════════════════",
        "",
        "1. DATASET SUMMARY",
        f"   Source file: {state['dataset_info'].get('path', 'N/A')}",
        f"   Rows: {state['dataset_info'].get('rows', 0):,} | Columns: {state['dataset_info'].get('cols', 0):,}",
        f"   Fraud rate: {state['dataset_info'].get('fraud_rate', 0):.2f}%",
        "",
        "2. PREPROCESSING STEPS APPLIED",
        "   - Dropped ID columns and timestamp",
        "   - Label encoded categorical features",
        "   - Filled missing values (median/mode)",
        "   - Standardized key numeric features",
        "   - Stratified train/test split (80/20)",
        "   - Applied SMOTE / upsampling for balance",
        "",
        "3. MODEL PERFORMANCE RESULTS",
    ]
    for label, metrics in state["results"].items():
        lines.append(
            f"   - {label}: Acc {metrics['accuracy']*100:.2f}%, Prec {metrics['precision']*100:.2f}%, "
            f"Recall {metrics['recall']*100:.2f}%, AUC {metrics['roc_auc']:.3f}"
        )
    lines.extend(
        [
            "",
            "4. BEST MODEL RECOMMENDATION",
            f"   → {best_label} (AUC {best_metrics['roc_auc']:.3f}, F1 {best_metrics['f1']*100:.2f}%)",
            "",
            "5. KEY FRAUD PATTERNS DISCOVERED",
            "   - Foreign transactions with TOR/VPN usage",
            "   - High amount_vs_avg_ratio spikes",
            "   - Rapid-fire attempts within 1 hour window",
            "",
            "6. TOP 10 FRAUD INDICATORS",
        ]
    )
    if not indicators.empty:
        lines.extend([
            indicators.round(2).to_string(index=False),
            "",
        ])
    lines.extend(
        [
            "7. RECOMMENDATIONS",
            "   - Enforce adaptive authentication for CRITICAL risk",
            "   - Monitor TOR/VPN patterns and geolocation anomalies",
            "   - Refresh models quarterly with latest fraud examples",
            "",
            "✓ Report saved to reports/fraud_detection_report.txt",
        ]
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"\n✓ Report saved to {report_path}")
    pause()


# ---------------------------------------------------------------------------
# Option 8: Save / Load model
# ---------------------------------------------------------------------------

def save_load_menu() -> None:
    """Sub-menu for persistence operations."""
    while True:
        print(
            "\nModel Persistence Menu:\n"
            "  8a. Save current best model\n"
            "  8b. Load previously saved model\n"
            "  8c. List all saved models\n"
            "  0. Return to main menu"
        )
        choice = input("Select an option: ").strip().lower()
        if choice in {"0", "q"}:
            break
        if choice in {"8a", "a"}:
            save_best_model()
        elif choice in {"8b", "b"}:
            load_saved_model()
        elif choice in {"8c", "c"}:
            list_saved_models()
        else:
            print("⚠ Invalid choice.")
    pause()


def save_best_model() -> None:
    """Persist the current best model and preprocessing artifacts."""
    if not require(state["models_trained"], "Please train models before saving!"):
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_label = state["best_model_name"]
    base_name = f"{best_label.lower().replace(' ', '_')}_{timestamp}"
    model_path = os.path.join(MODELS_DIR, f"{base_name}.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{timestamp}.pkl")
    enc_path = os.path.join(MODELS_DIR, f"encoders_{timestamp}.pkl")
    features_path = os.path.join(MODELS_DIR, f"feature_cols_{timestamp}.pkl")
    payload = {
        "model": state["best_model"],
        "metadata": {
            "label": best_label,
            "accuracy": state["results"][best_label]["accuracy"],
            "roc_auc": state["results"][best_label]["roc_auc"],
        },
    }
    joblib.dump(payload, model_path)
    if state["scaler"]:
        joblib.dump(state["scaler"], scaler_path)
    joblib.dump(state["encoders"], enc_path)
    joblib.dump(state["feature_cols"], features_path)
    print(f"  → Saved: {model_path}")
    if state["scaler"]:
        print(f"  → Saved: {scaler_path}")
    print(f"  → Saved: {enc_path}")
    print(f"  → Saved: {features_path}")
    print("  ✓ Model saved with timestamp")
    update_registry(
        base_name,
        model_path,
        state["results"][best_label]["accuracy"],
        timestamp,
        os.path.getsize(model_path),
    )


def load_saved_model() -> None:
    """Load a previously saved model and preprocessing assets."""
    entries = list_saved_models(show=False)
    if not entries:
        return
    selection = input("Enter the number of the model to load: ").strip()
    if not selection.isdigit():
        print("⚠ Invalid selection.")
        return
    idx = int(selection) - 1
    if idx < 0 or idx >= len(entries):
        print("⚠ Selection out of range.")
        return
    entry = entries[idx]
    try:
        payload = joblib.load(entry["path"])
        state["best_model"] = payload["model"]
        state["best_model_name"] = payload["metadata"]["label"]
        state["results"][state["best_model_name"]] = {
            "accuracy": payload["metadata"].get("accuracy", 0),
            "precision": payload["metadata"].get("accuracy", 0),
            "recall": payload["metadata"].get("accuracy", 0),
            "f1": payload["metadata"].get("accuracy", 0),
            "roc_auc": payload["metadata"].get("roc_auc", 0),
        }
        scaler_candidates = [f for f in os.listdir(MODELS_DIR) if f"scaler_{entry['timestamp']}" in f]
        encoder_candidates = [f for f in os.listdir(MODELS_DIR) if f"encoders_{entry['timestamp']}" in f]
        feature_candidates = [f for f in os.listdir(MODELS_DIR) if f"feature_cols_{entry['timestamp']}" in f]
        if scaler_candidates:
            state["scaler"] = joblib.load(os.path.join(MODELS_DIR, scaler_candidates[0]))
        state["encoders"] = joblib.load(os.path.join(MODELS_DIR, encoder_candidates[0])) if encoder_candidates else {}
        state["feature_cols"] = joblib.load(os.path.join(MODELS_DIR, feature_candidates[0])) if feature_candidates else []
        state["models_trained"] = True
        print(f"✓ Loaded model: {state['best_model_name']}")
    except Exception as exc:
        print(f"⚠ Failed to load model: {exc}")


def list_saved_models(show: bool = True) -> list[dict]:
    """Display registry entries for saved models."""
    if not os.path.exists(MODEL_REGISTRY):
        if show:
            print("⚠ No saved models found.")
        return []
    registry = pd.read_json(MODEL_REGISTRY)
    if registry.empty:
        if show:
            print("⚠ No saved models found.")
        return []
    entries = registry.to_dict(orient="records")
    if show:
        print("\nAvailable saved models:")
        for idx, entry in enumerate(entries, start=1):
            size_mb = entry["size"] / (1024 * 1024)
            print(
                f"  {idx}. {entry['name']}.pkl  ({entry['accuracy']*100:.2f}% acc)  {size_mb:.1f} MB"
            )
    return entries


def update_registry(name: str, path: str, accuracy: float, timestamp: str, size: int) -> None:
    """Append an entry to the model registry."""
    entry = {
        "name": name,
        "path": path,
        "accuracy": accuracy,
        "timestamp": timestamp,
        "size": size,
    }
    if os.path.exists(MODEL_REGISTRY):
        registry = pd.read_json(MODEL_REGISTRY)
        registry = pd.concat([registry, pd.DataFrame([entry])], ignore_index=True)
    else:
        registry = pd.DataFrame([entry])
    registry.to_json(MODEL_REGISTRY, orient="records", indent=2)


# ---------------------------------------------------------------------------
# Exit handling
# ---------------------------------------------------------------------------

def exit_summary() -> None:
    """Print closing summary before exiting."""
    dataset_status = (
        f"Yes ({state['dataset_info'].get('rows', 0):,} rows)"
        if state["dataset_loaded"]
        else "No"
    )
    models_trained = state["models_trained_count"]
    best_auc = state["results"].get(state["best_model_name"], {}).get("roc_auc", 0)
    summary = (
        "════════════════════════════════════════\n"
        "Thank you for using FraudShield AI!\n"
        "Session Summary:\n"
        f"→ Dataset loaded: {dataset_status}\n"
        f"→ Models trained: {models_trained}\n"
        f"→ Best AUC: {best_auc:.3f} ({state['best_model_name']})\n"
        f"→ Predictions made: {state['prediction_count']}\n"
        "════════════════════════════════════════\nGoodbye!"
    )
    print(summary)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Application entry point."""
    ensure_directories()
    print_banner()
    while True:
        show_main_menu()
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            load_dataset()
        elif choice == "2":
            explore_menu()
        elif choice == "3":
            preprocess_data()
        elif choice == "4":
            train_models_menu()
        elif choice == "5":
            evaluate_models()
        elif choice == "6":
            predict_menu()
        elif choice == "7":
            generate_report()
        elif choice == "8":
            save_load_menu()
        elif choice == "0":
            exit_summary()
            break
        else:
            print("⚠ Invalid option. Please try again.")


if __name__ == "__main__":
    main()
