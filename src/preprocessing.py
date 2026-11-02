"""
Feature Engineering and Preprocessing Pipeline
Decomposes order dates, one-hot encodes categorical variables strictly fitted on training data,
and prepares train/test matrices to prevent data leakage.
"""
import logging
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib

from src.config import (
    TARGET_COLUMN,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
    PROCESSED_DATA_DIR,
    TRAIN_PROCESSED_PATH,
    TEST_PROCESSED_PATH,
)

logger = logging.getLogger(__name__)


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from 'order_time' (format: DD-MM-YYYY)."""
    df = df.copy()
    if "order_time" in df.columns:
        date_series = pd.to_datetime(df["order_time"], format="%d-%m-%Y")
        df["order_month"] = date_series.dt.month
        df["order_day"] = date_series.dt.day
        df["order_dayofweek"] = date_series.dt.dayofweek
        df["is_weekend"] = date_series.dt.dayofweek.isin([5, 6]).astype(int)
        df = df.drop(columns=["order_time"])
    return df


class DeliveryDelayPreprocessor:
    """
    Production preprocessor that fits one-hot encoding exclusively on training data
    to strictly prevent data leakage (Slide 7 requirement).
    """

    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.categorical_cols = CATEGORICAL_COLUMNS
        self.numerical_cols = NUMERICAL_COLUMNS + ["order_month", "order_day", "order_dayofweek", "is_weekend"]
        self.encoded_feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        """Fit preprocessor on training feature matrix."""
        X_feat = extract_date_features(X)
        self.encoder.fit(X_feat[self.categorical_cols])
        encoded_cats = self.encoder.get_feature_names_out(self.categorical_cols).tolist()
        self.encoded_feature_names = self.numerical_cols + encoded_cats
        self.is_fitted = True
        logger.info(f"Preprocessor fitted. Total features after encoding: {len(self.encoded_feature_names)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input features using the fitted encoder."""
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet!")

        X_feat = extract_date_features(X)
        num_part = X_feat[self.numerical_cols].reset_index(drop=True)
        cat_array = self.encoder.transform(X_feat[self.categorical_cols])
        encoded_cats = self.encoder.get_feature_names_out(self.categorical_cols).tolist()
        cat_part = pd.DataFrame(cat_array, columns=encoded_cats)

        transformed_df = pd.concat([num_part, cat_part], axis=1)
        return transformed_df

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform."""
        return self.fit(X).transform(X)


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save_processed: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, DeliveryDelayPreprocessor]:
    """
    Split data into 80% train and 20% test, fit preprocessor on train,
    and return processed X_train, X_test, y_train, y_test, and preprocessor.
    """
    logger.info(f"Splitting dataset: {1 - test_size:.0%} train, {test_size:.0%} test...")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Training rows: {len(X_train_raw):,}, Testing rows: {len(X_test_raw):,}")

    preprocessor = DeliveryDelayPreprocessor()
    preprocessor.fit(X_train_raw)

    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    if save_processed:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Save a sample or header of processed data to avoid disk bloat, or save full
        train_out = pd.concat([X_train.head(5000), y_train.head(5000).reset_index(drop=True)], axis=1)
        test_out = pd.concat([X_test.head(1000), y_test.head(1000).reset_index(drop=True)], axis=1)
        train_out.to_csv(TRAIN_PROCESSED_PATH, index=False)
        test_out.to_csv(TEST_PROCESSED_PATH, index=False)
        logger.info(f"Saved processed sample files to {PROCESSED_DATA_DIR}")

    return X_train, X_test, y_train, y_test, preprocessor

