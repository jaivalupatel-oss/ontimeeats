"""
Unit tests for feature engineering, datetime extraction, and One-Hot encoding.
"""
import unittest
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import extract_date_features, DeliveryDelayPreprocessor, prepare_train_test_data
from src.config import TARGET_COLUMN


class TestPreprocessing(unittest.TestCase):
    """Test suite for preprocessing and feature engineering."""

    def setUp(self):
        self.sample_df = pd.DataFrame({
            "food_item": ["Pizza", "Burger", "Biryani"],
            "order_time": ["24-05-2024", "15-08-2024", "01-01-2024"],
            "order_value": [500.0, 750.0, 300.0],
            "order_frequency": ["Weekly", "Monthly", "Daily"],
            "loyalty_program": ["Yes", "No", "Yes"],
            "delivery_distance": [5.2, 8.4, 2.1],
            "delivery_method": ["Bike", "Car", "Bike"],
            "route_taken": ["Route_1", "Route_2", "Route_1"],
            "route_type": ["Urban", "Mixed", "Urban"],
            "route_efficiency": [0.75, 0.45, 0.90],
            "traffic_condition": ["Low", "High", "Low"],
            "weather_condition": ["Clear", "Rainy", "Sunny"],
            "traffic_avoidance": ["Yes", "No", "Yes"],
            "location": ["Downtown", "Suburbs", "Downtown"],
            "age": [28, 35, 22],
            "gender": ["Male", "Female", "Male"],
            "order_history": [15, 42, 5],
            "customer_rating": [5, 4, 5],
            "preferred_cuisine": ["Italian", "American", "Indian"],
            "delivery_delay": [2.5, 14.2, 1.8]
        })

    def test_extract_date_features(self):
        """Verify decomposition of DD-MM-YYYY into day, month, dayofweek, and weekend."""
        df_feat = extract_date_features(self.sample_df)
        self.assertNotIn("order_time", df_feat.columns)
        self.assertIn("order_month", df_feat.columns)
        self.assertIn("order_day", df_feat.columns)
        self.assertIn("order_dayofweek", df_feat.columns)
        self.assertIn("is_weekend", df_feat.columns)

        # 24-05-2024 was a Friday (dayofweek 4, is_weekend 0)
        self.assertEqual(df_feat.loc[0, "order_month"], 5)
        self.assertEqual(df_feat.loc[0, "order_day"], 24)
        self.assertEqual(df_feat.loc[0, "order_dayofweek"], 4)
        self.assertEqual(df_feat.loc[0, "is_weekend"], 0)

    def test_preprocessor_fit_transform(self):
        """Verify preprocessor transforms categorical features and retains numerical columns."""
        preprocessor = DeliveryDelayPreprocessor()
        X = self.sample_df.drop(columns=[TARGET_COLUMN])
        X_trans = preprocessor.fit_transform(X)

        self.assertTrue(preprocessor.is_fitted)
        self.assertEqual(len(X_trans), len(self.sample_df))
        self.assertGreater(len(preprocessor.encoded_feature_names), len(X.columns))

    def test_unseen_categories_graceful_handling(self):
        """Verify preprocessor does not fail on unseen category values in transform."""
        preprocessor = DeliveryDelayPreprocessor()
        X_train = self.sample_df.drop(columns=[TARGET_COLUMN])
        preprocessor.fit(X_train)

        # Create input with unknown categories
        unseen_df = X_train.copy()
        unseen_df.loc[0, "food_item"] = "Exotic_Delicacy_Unknown"
        unseen_df.loc[0, "weather_condition"] = "Hailstorm_Rare"

        # Transform should succeed without error (OneHotEncoder handle_unknown='ignore')
        X_unseen_trans = preprocessor.transform(unseen_df)
        self.assertEqual(len(X_unseen_trans), len(unseen_df))
        self.assertFalse(X_unseen_trans.isnull().any().any())

    def test_prepare_train_test_data(self):
        """Verify train-test split shapes and anti-leakage encapsulation."""
        # Create a 20-row sample to allow 80/20 splitting
        df_large = pd.concat([self.sample_df] * 10, ignore_index=True)
        X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data(
            df_large, test_size=0.20, random_state=42, save_processed=False
        )

        self.assertEqual(len(X_train), 24)
        self.assertEqual(len(X_test), 6)
        self.assertEqual(len(y_train), 24)
        self.assertEqual(len(y_test), 6)
        self.assertEqual(X_train.shape[1], X_test.shape[1])


if __name__ == "__main__":
    unittest.main()
