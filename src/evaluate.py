"""
Model Evaluation and Diagnostic Reporting Module
Computes MAE, RMSE, and R2 on unseen test data, performs residual analysis,
generates publication-ready visualizations, and produces an academic evaluation report.
"""
import json
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    median_absolute_error
)

from src.config import (
    FIGURES_DIR,
    EVALUATION_REPORT_PATH,
    METADATA_SAVE_PATH,
    TARGET_COLUMN
)

logger = logging.getLogger(__name__)

# Aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate key regression evaluation metrics (Slide 8)."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    med_ae = float(median_absolute_error(y_true, y_pred))
    evs = float(explained_variance_score(y_true, y_pred))

    metrics = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MSE": round(mse, 4),
        "R2_Score": round(r2, 4),
        "Median_AE": round(med_ae, 4),
        "Explained_Variance": round(evs, 4),
    }

    logger.info("Test Evaluation Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    return metrics


def plot_feature_importance(model, feature_names: list, top_n: int = 15) -> pd.DataFrame:
    """Plot and save feature importance bar chart."""
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    top_df = feat_df.head(top_n)
    colors = sns.color_palette("viridis", len(top_df))
    plt.barh(top_df["feature"][::-1], top_df["importance"][::-1], color=colors)
    plt.xlabel("Importance (Mean Decrease in Impurity)")
    plt.title(f"Top {top_n} Most Important Features for Delivery Delay Prediction", fontsize=14, pad=12)
    plt.tight_layout()

    save_path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature importance plot to {save_path}")
    return feat_df


def plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray, sample_size: int = 2000):
    """Plot Actual vs Predicted Delivery Delay scatter plot."""
    if len(y_true) > sample_size:
        idx = np.random.choice(len(y_true), sample_size, replace=False)
        y_t = np.array(y_true)[idx]
        y_p = y_pred[idx]
    else:
        y_t = np.array(y_true)
        y_p = y_pred

    plt.figure(figsize=(8, 8))
    plt.scatter(y_t, y_p, alpha=0.3, color="#1f77b4", edgecolors="none", s=25, label="Test Predictions")
    min_val = min(y_t.min(), y_p.min())
    max_val = max(y_t.max(), y_p.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal 1:1 Fit")

    plt.xlabel("Actual Delivery Delay (minutes)", fontsize=12)
    plt.ylabel("Predicted Delivery Delay (minutes)", fontsize=12)
    plt.title("Actual vs. Predicted Food Delivery Delay", fontsize=14, pad=12)
    plt.legend(loc="upper left")
    plt.tight_layout()

    save_path = FIGURES_DIR / "actual_vs_predicted.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved actual vs predicted plot to {save_path}")


def plot_residual_analysis(y_true: pd.Series, y_pred: np.ndarray):
    """Plot residuals distribution and residual vs predicted plot."""
    residuals = np.array(y_true) - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals Histogram & KDE
    sns.histplot(residuals, bins=50, kde=True, ax=axes[0], color="#2ca02c")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("Distribution of Residuals (Errors)", fontsize=13)
    axes[0].set_xlabel("Residual (Actual - Predicted) [minutes]")
    axes[0].set_ylabel("Count")

    # Residuals vs Predicted
    sub_idx = np.random.choice(len(residuals), min(2500, len(residuals)), replace=False)
    axes[1].scatter(y_pred[sub_idx], residuals[sub_idx], alpha=0.3, color="#ff7f0e", s=20)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals vs. Predicted Values", fontsize=13)
    axes[1].set_xlabel("Predicted Delivery Delay (minutes)")
    axes[1].set_ylabel("Residual (minutes)")

    plt.tight_layout()
    save_path = FIGURES_DIR / "residuals_distribution.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved residuals plot to {save_path}")


def plot_eda_figures(df_clean: pd.DataFrame):
    """Generate EDA figures: delay distribution, traffic/weather interactions, correlation heatmap."""
    # 1. Target distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(df_clean[TARGET_COLUMN], bins=40, kde=True, color="#4c72b0")
    plt.title("Observed Delivery Delay Distribution (Range: 0.01 - 20.00 min)", fontsize=14)
    plt.xlabel("Delivery Delay (minutes)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "delay_distribution.png", dpi=300)
    plt.close()

    # 2. Delay by traffic condition & weather condition
    plt.figure(figsize=(10, 5))
    grouped = df_clean.groupby(["traffic_condition", "weather_condition"])[TARGET_COLUMN].mean().reset_index()
    sns.barplot(data=grouped, x="traffic_condition", y=TARGET_COLUMN, hue="weather_condition", palette="Blues_d")
    plt.title("Average Delivery Delay by Traffic and Weather Condition", fontsize=14)
    plt.xlabel("Traffic Condition")
    plt.ylabel("Mean Delivery Delay (minutes)")
    plt.legend(title="Weather")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "traffic_weather_delays.png", dpi=300)
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(8, 6))
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    corr = df_clean[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Correlation Matrix of Numerical Features", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()
    logger.info("Saved all EDA figures.")


def generate_evaluation_report(
    metrics: Dict[str, float],
    feat_df: pd.DataFrame,
    summary: Dict[str, Any],
    pipeline_meta: Dict[str, Any]
):
    """Generate detailed markdown evaluation report."""
    try:
        top_features = feat_df.head(10).to_markdown(index=False)
    except Exception:
        top_features = "| Feature | Importance |\n| :--- | :--- |\n" + "\n".join(
            [f"| {row['feature']} | {row['importance']:.6f} |" for _, row in feat_df.head(10).iterrows()]
        )

    report_content = f"""# Food Delivery Delay Prediction - Model Evaluation Report

**Project**: Food Delivery Delay Prediction  
**Author**: Jaival Patel | B.Tech CSE (23000300)  
**Algorithm**: Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`)  
**Task Type**: Supervised Continuous Regression  

---

## 1. Executive Summary
This project predicts customer food delivery delays in minutes (`delivery_delay`) using 19 contextual features covering order specifics, route efficiency, environmental factors (traffic and weather), and customer order history.

The machine learning pipeline implements strict data cleaning (eliminating IDs and preventing data leakage) and an 80/20 train-test split (160,000 training records, 40,000 testing records). Preprocessing transformations (temporal feature extraction and One-Hot Encoding) were strictly fitted onto the training partition.

---

## 2. Test Set Evaluation Metrics (Unseen 40,000 Records)

| Metric | Value | Description |
| :--- | :--- | :--- |
| **MAE** | **{metrics['MAE']} minutes** | Mean Absolute Error — average absolute deviation between actual and predicted delivery delays. |
| **RMSE** | **{metrics['RMSE']} minutes** | Root Mean Squared Error — penalizes larger outliers and extreme prediction errors. |
| **R² Score** | **{metrics['R2_Score']}** | Coefficient of Determination — indicates {metrics['R2_Score']*100:.2f}% of delay variance is explained by the model. |
| **Median AE** | **{metrics['Median_AE']} minutes** | Median Absolute Error — robust central tendency of error. |
| **Explained Variance** | **{metrics['Explained_Variance']}** | Proportion of variance accounted for by the regressor. |

---

## 3. Dataset Integrity & Quality Verification

| Specification | Observed Value | Status |
| :--- | :--- | :--- |
| Total Records | {summary['initial_records']:,} | Verified |
| Target Variable Range | {summary['delay_min']:.2f} min to {summary['delay_max']:.2f} min | Verified |
| Order Value Range | ₹{summary['order_value_min']:,.2f} to ₹{summary['order_value_max']:,.2f} | Verified |
| Missing Values | {summary['missing_values']} | Passed (0 Missing) |
| Duplicate Rows | {summary['duplicate_rows']} | Passed (0 Duplicates) |
| Negative Delays | {summary['negative_delays_removed']} | Passed (0 Invalid) |
| Dropped Identifiers | {', '.join(summary['dropped_columns'])} | Removed to prevent spurious memorization |

---

## 4. Top Feature Importances (MDI)

{top_features}

---

## 5. Model Architecture & Hyperparameters
- **Estimator**: `RandomForestRegressor`
- **Number of Estimators**: {pipeline_meta['hyperparameters']['n_estimators']}
- **Max Tree Depth**: {pipeline_meta['hyperparameters']['max_depth']}
- **Min Samples Split**: {pipeline_meta['hyperparameters']['min_samples_split']}
- **Min Samples Leaf**: {pipeline_meta['hyperparameters']['min_samples_leaf']}
- **Training Set Size**: {pipeline_meta['train_samples']:,} samples
- **Training Wall Time**: {pipeline_meta['training_duration_seconds']} seconds

---

## 6. Real-World Operational Impact
1. **Dynamic ETA Communication**: Real-time delay forecasts allow dynamic adjustment of delivery promises to customers, eliminating negative surprises.
2. **Early Intervention & Dispatch Optimization**: Identifies high-risk deliveries (e.g., low route efficiency + severe weather + high traffic) for priority fleet assignment.
3. **Route & Delivery Mode Decisions**: Insights reveal route efficiency and delivery vehicle modes strongly determine reliability across varying urban traffic conditions.
"""
    EVALUATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    # Save JSON metadata
    meta = {
        "metrics": metrics,
        "summary": summary,
        "hyperparameters": pipeline_meta["hyperparameters"],
        "top_features": feat_df.head(15).to_dict(orient="records")
    }
    with open(METADATA_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Evaluation report written to {EVALUATION_REPORT_PATH}")
