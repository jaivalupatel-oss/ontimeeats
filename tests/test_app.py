"""
Integration tests for the Flask Web Application and REST API endpoints.
"""
import unittest
import json
import io
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app import app


class TestWebApp(unittest.TestCase):
    """Test suite for Flask dashboard and API."""

    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_home_page_renders(self):
        """Verify the dashboard home page loads successfully (HTTP 200)."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Food Delivery Delay Prediction", response.data)
        self.assertIn(b"Jaival Patel", response.data)

    def test_predict_endpoint_valid_payload(self):
        """Verify single order prediction API returns correct schema and values."""
        payload = {
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

        response = self.client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        res_data = json.loads(response.data)
        self.assertEqual(res_data["status"], "success")
        self.assertIn("data", res_data)
        self.assertIn("predicted_delay_minutes", res_data["data"])
        self.assertIn("risk_level", res_data["data"])
        self.assertIn("recommended_action", res_data["data"])

    def test_predict_endpoint_invalid_payload(self):
        """Verify API handles invalid or malformed payload gracefully."""
        response = self.client.post(
            "/predict",
            data="invalid-payload-not-json",
            content_type="text/plain"
        )
        self.assertEqual(response.status_code, 400)
        res_data = json.loads(response.data)
        self.assertEqual(res_data["status"], "error")

    def test_sample_batch_data_endpoint(self):
        """Verify sample batch data endpoint returns list of orders."""
        response = self.client.get("/sample_batch_data")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertGreater(len(data["orders"]), 0)

    def test_predict_batch_endpoint_json(self):
        """Verify batch prediction endpoint with JSON array of orders."""
        payload = [
            {
                "food_item": "Pizza",
                "order_time": "24-05-2024",
                "delivery_distance": 5.0,
                "order_value": 500.0,
                "delivery_method": "Bike",
                "traffic_condition": "Low",
                "weather_condition": "Clear",
                "route_taken": "Route_1",
                "age": 30,
                "gender": "Male",
                "location": "Downtown",
                "order_history": 20,
                "customer_rating": 5,
                "preferred_cuisine": "Italian",
                "order_frequency": "Weekly",
                "loyalty_program": "Yes",
                "route_type": "Urban",
                "route_efficiency": 0.8,
                "traffic_avoidance": "Yes"
            }
        ]
        response = self.client.post(
            "/predict_batch",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("summary", data)
        self.assertIn("records", data)
        self.assertEqual(data["summary"]["total_orders"], 1)

    def test_simulate_routes_endpoint(self):
        """Verify route simulation endpoint returns ranked alternatives."""
        base_order = {
            "food_item": "Pizza",
            "order_time": "24-05-2024",
            "delivery_distance": 4.5,
            "order_value": 600.0,
            "traffic_condition": "Medium",
            "weather_condition": "Rainy",
            "age": 28,
            "gender": "Male",
            "location": "Downtown",
            "order_history": 15,
            "customer_rating": 4,
            "preferred_cuisine": "Italian",
            "order_frequency": "Weekly",
            "loyalty_program": "Yes",
            "route_type": "Urban",
            "traffic_avoidance": "No"
        }
        response = self.client.post(
            "/simulate_routes",
            data=json.dumps(base_order),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("optimal_choice", data)
        self.assertIn("scenarios", data)
        self.assertGreater(len(data["scenarios"]), 0)


if __name__ == "__main__":
    unittest.main()

