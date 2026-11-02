"""
Unit tests for real-time model inference, risk classification, and batch prediction.
"""
import unittest
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import DeliveryDelayPredictor


class TestPredict(unittest.TestCase):
    """Test suite for inference pipeline and operational risk logic."""

    @classmethod
    def setUpClass(cls):
        """Initialize the predictor once for the test suite."""
        cls.predictor = DeliveryDelayPredictor()

    def test_risk_classification_thresholds(self):
        """Verify risk tiers: Low (<5 min), Moderate (5-12 min), High (>12 min)."""
        low = self.predictor.classify_risk(3.5)
        self.assertEqual(low["risk_level"], "Low Delay Risk")
        self.assertEqual(low["risk_color"], "green")

        moderate = self.predictor.classify_risk(8.0)
        self.assertEqual(moderate["risk_level"], "Moderate Delay Risk")
        self.assertEqual(moderate["risk_color"], "orange")

        high = self.predictor.classify_risk(14.5)
        self.assertEqual(high["risk_level"], "High Delay Risk")
        self.assertEqual(high["risk_color"], "red")

    def test_slide_9_example_prediction(self):
        """Verify inference execution on the Slide 9 reference order."""
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

        res = self.predictor.predict_single(example_order)
        self.assertIn("predicted_delay_minutes", res)
        self.assertIn("risk_level", res)
        self.assertIn("recommended_action", res)

        delay = res["predicted_delay_minutes"]
        self.assertGreaterEqual(delay, 0.01)
        self.assertLessEqual(delay, 20.00)
        self.assertIsInstance(delay, float)

    def test_batch_prediction(self):
        """Verify batch inference across multiple orders."""
        batch_df = pd.DataFrame([
            {
                "food_item": "Burger",
                "order_time": "25-05-2024",
                "delivery_distance": 3.0,
                "order_value": 350.0,
                "delivery_method": "Bike",
                "traffic_condition": "Low",
                "weather_condition": "Clear",
                "route_taken": "Route_1",
                "age": 25,
                "gender": "Female",
                "location": "Downtown",
                "order_history": 10,
                "customer_rating": 5,
                "preferred_cuisine": "American",
                "order_frequency": "Daily",
                "loyalty_program": "Yes",
                "route_type": "Urban",
                "route_efficiency": 0.85,
                "traffic_avoidance": "Yes"
            },
            {
                "food_item": "Biryani",
                "order_time": "26-05-2024",
                "delivery_distance": 14.0,
                "order_value": 1100.0,
                "delivery_method": "Car",
                "traffic_condition": "High",
                "weather_condition": "Stormy",
                "route_taken": "Route_4",
                "age": 42,
                "gender": "Male",
                "location": "Suburbs",
                "order_history": 60,
                "customer_rating": 3,
                "preferred_cuisine": "Indian",
                "order_frequency": "Monthly",
                "loyalty_program": "No",
                "route_type": "Mixed",
                "route_efficiency": 0.25,
                "traffic_avoidance": "No"
            }
        ])

        batch_result = self.predictor.predict_batch(batch_df)
        self.assertIn("predicted_delay_minutes", batch_result.columns)
        self.assertIn("risk_level", batch_result.columns)
        self.assertEqual(len(batch_result), 2)
        self.assertTrue(((batch_result["predicted_delay_minutes"] >= 0.01) & 
                         (batch_result["predicted_delay_minutes"] <= 20.00)).all())
        valid_tiers = {"Low Delay Risk", "Moderate Delay Risk", "High Delay Risk"}
        for tier in batch_result["risk_level"]:
            self.assertIn(tier, valid_tiers)


if __name__ == "__main__":
    unittest.main()
