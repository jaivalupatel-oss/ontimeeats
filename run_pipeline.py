"""
Master Execution Pipeline
Orchestrates end-to-end flow: Ingestion -> Cleaning -> Preprocessing -> Model Training -> Evaluation -> Diagnostics.
Author: Jaival Patel (B.Tech CSE | 23000300)
"""
import sys
import time
import logging
import argparse
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_raw_data, validate_and_clean_data
from src.preprocessing import prepare_train_test_data
from src.train import train_model
from src.evaluate import (
    compute_metrics,
    plot_feature_importance,
    plot_actual_vs_predicted,
    plot_residual_analysis,
    plot_eda_figures,
    generate_evaluation_report,
)
from src.predict import DeliveryDelayPredictor
from src.config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PipelineRunner")


def run_full_pipeline():
    """Execute end-to-end DSML pipeline from raw data to diagnostics."""
    logger.info("=" * 70)
    logger.info("STARTING FOOD DELIVERY DELAY PREDICTION DSML PIPELINE")
    logger.info("=" * 70)

    overall_start = time.time()

    # Step 1: Load Raw Data
    raw_df = load_raw_data()

    # Step 2: Validate and Clean Data
    clean_df, data_summary = validate_and_clean_data(raw_df)

    # Step 3: Exploratory Data Analysis Plots
    logger.info("Generating EDA figures...")
    plot_eda_figures(clean_df)

    # Step 4: Preprocessing & 80/20 Train-Test Split
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data(
        clean_df, test_size=0.20, random_state=42
    )

    # Step 5: Model Training (Random Forest Regressor)
    model, pipeline_package = train_model(X_train, y_train, preprocessor)

    # Step 6: Model Evaluation on 40,000 Unseen Test Records
    logger.info("Evaluating model on 20% unseen test set (40,000 records)...")
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    # Step 7: Visual Diagnostics & Residual Analysis
    logger.info("Generating diagnostic evaluation figures...")
    feat_df = plot_feature_importance(model, preprocessor.encoded_feature_names, top_n=15)
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residual_analysis(y_test, y_pred)

    # Step 8: Generate Comprehensive Markdown Evaluation Report
    generate_evaluation_report(metrics, feat_df, data_summary, pipeline_package)

    # Step 9: Validate Slide 9 Example Prediction
    logger.info("Validating Slide 9 example order prediction...")
    predictor = DeliveryDelayPredictor()
    example_order = {
        "food_item": "Pizza",
        "order_time": "24-05-2024",
        "delivery_distance": 10.5,
        "order_value": 680.0,
        "delivery_method": "Bike",
        "traffic_condition": "High",
        "weather_condition": "Rainy",
        "route_taken": "Route_2",
        "age": 28,
        "gender": "Male",
        "location": "Ahmedabad",
        "order_history": 25,
        "customer_rating": 4,
        "preferred_cuisine": "Italian",
        "order_frequency": "Weekly",
        "loyalty_program": "Yes",
        "route_type": "Mixed",
        "route_efficiency": 0.45,
        "traffic_avoidance": "No"
    }
    sample_result = predictor.predict_single(example_order)

    total_time = time.time() - overall_start
    logger.info("=" * 70)
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS")
    logger.info(f"Test MAE: {metrics['MAE']} min | RMSE: {metrics['RMSE']} min | R2 Score: {metrics['R2_Score']}")
    logger.info(f"Slide 9 Example Prediction: {sample_result['predicted_delay_minutes']} min ({sample_result['risk_level']})")
    logger.info("=" * 70)


def run_predict_sample():
    """Run real-time inference on the Slide 9 demonstration order."""
    predictor = DeliveryDelayPredictor()
    order = {
        "food_item": "Pizza",
        "order_time": "24-05-2024",
        "delivery_distance": 10.5,
        "order_value": 680.0,
        "delivery_method": "Bike",
        "traffic_condition": "High",
        "weather_condition": "Rainy",
        "route_taken": "Route_2",
        "age": 28,
        "gender": "Male",
        "location": "Ahmedabad",
        "order_history": 25,
        "customer_rating": 4,
        "preferred_cuisine": "Italian",
        "order_frequency": "Weekly",
        "loyalty_program": "Yes",
        "route_type": "Mixed",
        "route_efficiency": 0.45,
        "traffic_avoidance": "No"
    }
    res = predictor.predict_single(order)
    print("\n" + "=" * 50)
    print("SLIDE 9 EXAMPLE INFERENCE RESULT")
    print("=" * 50)
    print(f"Predicted Delay:     {res['predicted_delay_minutes']} minutes")
    print(f"Risk Tier:           {res['risk_level']}")
    print(f"Recommended Action:  {res['recommended_action']}")
    print("=" * 50 + "\n")


def run_batch_predict(filepath=None):
    """Run batch prediction on sample fleet orders CSV."""
    if filepath is None:
        filepath = DATA_DIR / "sample_orders_batch.csv"
    logger.info(f"Running batch prediction on {filepath}...")
    df = pd.read_csv(filepath)
    predictor = DeliveryDelayPredictor()
    results = predictor.predict_batch(df)

    print("\n" + "=" * 70)
    print("BATCH FLEET DISPATCH INFERENCE RESULTS")
    print("=" * 70)
    cols = ["food_item", "delivery_distance", "delivery_method", "traffic_condition", "predicted_delay_minutes", "risk_level"]
    print(results[cols].to_string(index=False))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Food Delivery Delay Prediction Pipeline")
    parser.add_argument(
        "--mode",
        choices=["all", "predict-sample", "batch-predict"],
        default="all",
        help="Pipeline execution mode (default: all)"
    )
    args = parser.parse_args()

    if args.mode == "all":
        run_full_pipeline()
    elif args.mode == "predict-sample":
        run_predict_sample()
    elif args.mode == "batch-predict":
        run_batch_predict()
