"""
Unit tests for data loading, data validation, and data cleaning routines.
"""
import unittest
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import validate_and_clean_data
from src.config import TARGET_COLUMN, DROP_COLUMNS


class TestDataLoader(unittest.TestCase):
    """Test suite for data loading and cleaning functions."""

    def setUp(self):
        """Create a mock raw dataset containing edge cases."""
        self.sample_data = pd.DataFrame({
            "restaurant_id": ["REST_101", "REST_102", "REST_103", "REST_104", "REST_105"],
            "food_item": ["Pizza", "Burger", "Biryani", "Salad", "Pasta"],
            "order_time": ["24-05-2024", "25-05-2024", "26-05-2024", "27-05-2024", "28-05-2024"],
            "order_value": [500.0, 750.0, 300.0, 1200.0, 450.0],
            "order_frequency": ["Weekly", "Monthly", "Daily", "Weekly", "Rarely"],
            "loyalty_program": ["Yes", "No", "Yes", "Yes", "No"],
            "delivery_distance": [5.2, 8.4, 2.1, 12.0, 6.5],
            "delivery_method": ["Bike", "Car", "Bike", "Car", "Walk"],
            "route_taken": ["Route_1", "Route_2", "Route_1", "Route_3", "Route_2"],
            "route_type": ["Urban", "Mixed", "Urban", "Highway", "Urban"],
            "route_efficiency": [0.75, 0.45, 0.90, 0.35, 0.60],
            "traffic_condition": ["Low", "High", "Low", "High", "Medium"],
            "weather_condition": ["Clear", "Rainy", "Sunny", "Stormy", "Clear"],
            "traffic_avoidance": ["Yes", "No", "Yes", "No", "Yes"],
            "location": ["Downtown", "Suburbs", "Downtown", "West End", "North Suburbs"],
            "age": [28, 35, 22, 45, 31],
            "gender": ["Male", "Female", "Male", "Female", "Male"],
            "order_history": [15, 42, 5, 80, 20],
            "customer_rating": [5, 4, 5, 3, 4],
            "preferred_cuisine": ["Italian", "American", "Indian", "Healthy", "Italian"],
            "delivery_delay": [2.5, 14.2, -1.0, 18.5, 6.0]  # Record 3 has invalid negative delay
        })

    def test_negative_delays_filtered(self):
        """Verify negative delivery delays are filtered out as per Slide 4."""
        clean_df, summary = validate_and_clean_data(self.sample_data)
        self.assertEqual(summary["negative_delays_removed"], 1)
        self.assertEqual(len(clean_df), 4)
        self.assertTrue((clean_df[TARGET_COLUMN] >= 0).all())

    def test_unnecessary_ids_dropped(self):
        """Verify identifier columns (restaurant_id) are dropped to prevent leakage."""
        clean_df, summary = validate_and_clean_data(self.sample_data)
        for drop_col in DROP_COLUMNS:
            self.assertNotIn(drop_col, clean_df.columns)
            self.assertIn(drop_col, summary["dropped_columns"])

    def test_summary_metrics(self):
        """Verify data cleaning summary accurately reports metrics."""
        clean_df, summary = validate_and_clean_data(self.sample_data)
        self.assertEqual(summary["initial_records"], 5)
        self.assertEqual(summary["final_records"], 4)
        self.assertEqual(summary["missing_values"], 0)
        self.assertEqual(summary["duplicate_rows"], 0)
        self.assertAlmostEqual(summary["delay_min"], 2.5)
        self.assertAlmostEqual(summary["delay_max"], 18.5)


if __name__ == "__main__":
    unittest.main()
