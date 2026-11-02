"""
Model Training Module
Trains a Random Forest Regressor on the 80% training data (160,000 records)
and persists the trained model pipeline.
"""
import time
import logging
from typing import Dict, Any, Tuple
import joblib
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

from src.config import MODEL_SAVE_PATH, MODELS_DIR, RF_PARAMS
from src.preprocessing import DeliveryDelayPreprocessor

logger = logging.getLogger(__name__)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: DeliveryDelayPreprocessor,
    params: Dict[str, Any] = None
) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    """
    Train Random Forest Regressor on training set.
    """
    if params is None:
        params = RF_PARAMS

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing Random Forest Regressor with hyperparameters:")
    for k, v in params.items():
        logger.info(f"  {k}: {v}")

    rf = RandomForestRegressor(**params)

    start_time = time.time()
    logger.info(f"Training on {len(X_train):,} records with {X_train.shape[1]} features...")
    rf.fit(X_train, y_train)
    training_duration = time.time() - start_time
    logger.info(f"Training completed in {training_duration:.2f} seconds.")

    pipeline_package = {
        "model": rf,
        "preprocessor": preprocessor,
        "feature_names": preprocessor.encoded_feature_names,
        "hyperparameters": params,
        "training_duration_seconds": round(training_duration, 2),
        "train_samples": len(X_train),
    }

    joblib.dump(pipeline_package, MODEL_SAVE_PATH)
    logger.info(f"Pipeline successfully saved to {MODEL_SAVE_PATH}")

    return rf, pipeline_package

