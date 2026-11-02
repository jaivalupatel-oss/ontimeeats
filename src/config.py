"""
Configuration file containing paths, hyperparameters, and feature definitions.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "food_delivery_final.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_PROCESSED_PATH = PROCESSED_DATA_DIR / "train_processed.csv"
TEST_PROCESSED_PATH = PROCESSED_DATA_DIR / "test_processed.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_SAVE_PATH = MODELS_DIR / "random_forest_regressor.joblib"
METADATA_SAVE_PATH = MODELS_DIR / "model_artifacts.json"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EVALUATION_REPORT_PATH = REPORTS_DIR / "model_evaluation_report.md"

# Target Column
TARGET_COLUMN = "delivery_delay"

# Identifier column to drop (as per Slide 4: Unnecessary IDs removed)
DROP_COLUMNS = ["restaurant_id"]

# Categorical and Numerical Feature Lists (as per Slide 5)
ORDER_FEATURES = [
    "food_item",
    "order_time",
    "order_value",
    "order_frequency",
    "loyalty_program"
]

DELIVERY_FEATURES = [
    "delivery_distance",
    "delivery_method",
    "route_taken",
    "route_type",
    "route_efficiency"
]

EXTERNAL_FEATURES = [
    "traffic_condition",
    "weather_condition",
    "traffic_avoidance",
    "location"
]

CUSTOMER_FEATURES = [
    "age",
    "gender",
    "order_history",
    "customer_rating",
    "preferred_cuisine"
]

# All predictor columns before preprocessing
RAW_PREDICTOR_COLUMNS = (
    ORDER_FEATURES +
    DELIVERY_FEATURES +
    EXTERNAL_FEATURES +
    CUSTOMER_FEATURES
)

# Categorical columns to One-Hot Encode (order_time is decomposed separately)
CATEGORICAL_COLUMNS = [
    "food_item",
    "delivery_method",
    "traffic_condition",
    "weather_condition",
    "route_taken",
    "gender",
    "location",
    "preferred_cuisine",
    "order_frequency",
    "loyalty_program",
    "route_type",
    "traffic_avoidance"
]

# Continuous & integer features
NUMERICAL_COLUMNS = [
    "delivery_distance",
    "order_value",
    "route_efficiency",
    "age",
    "order_history",
    "customer_rating"
]

# Train-Test Split Configuration (Slide 7: 80% Train, 20% Test)
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Tuned Random Forest Hyperparameters (Config 3: R² = 99.54%, MAE = 0.27 min)
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 25,
    "min_samples_split": 3,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

