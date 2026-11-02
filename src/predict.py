"""
Inference and Operational Prediction Module
Provides real-time delay inference, risk classification, and decision-support guidance.
"""
import sys
import logging
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import joblib

from src.config import MODEL_SAVE_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DeliveryDelayPredictor:
    """
    Inference engine that wraps the trained preprocessor and Random Forest model.
    """

    def __init__(self, model_path=MODEL_SAVE_PATH):
        self.model_path = model_path
        self._load_pipeline()

    def _load_pipeline(self):
        logger.info(f"Loading model pipeline from {self.model_path}...")
        package = joblib.load(self.model_path)
        self.model = package["model"]
        self.preprocessor = package["preprocessor"]
        self.feature_names = package["feature_names"]
        logger.info("Pipeline loaded successfully.")

    @staticmethod
    def classify_risk(predicted_delay: float) -> Dict[str, str]:
        """Classify operational delay risk and assign actionable intervention."""
        if predicted_delay < 5.0:
            return {
                "risk_level": "Low Delay Risk",
                "risk_color": "green",
                "action": "Standard dispatch. Order progressing on scheduled ETA."
            }
        elif predicted_delay <= 12.0:
            return {
                "risk_level": "Moderate Delay Risk",
                "risk_color": "orange",
                "action": "Proactive notification. Send updated dynamic ETA to customer."
            }
        else:
            return {
                "risk_level": "High Delay Risk",
                "risk_color": "red",
                "action": "Urgent intervention! Assign priority rider, notify restaurant for expedited dispatch."
            }

    def predict_single(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict delivery delay in minutes for a single incoming order.
        """
        # Ensure DataFrame shape
        df_input = pd.DataFrame([order_data])

        # Apply preprocessing
        X_encoded = self.preprocessor.transform(df_input)

        # Ensure exact column alignment
        X_aligned = X_encoded.reindex(columns=self.feature_names, fill_value=0)

        # Predict
        predicted_delay = float(self.model.predict(X_aligned)[0])
        predicted_delay = round(max(0.01, min(predicted_delay, 20.0)), 2)

        risk_info = self.classify_risk(predicted_delay)

        return {
            "predicted_delay_minutes": predicted_delay,
            "risk_level": risk_info["risk_level"],
            "risk_color": risk_info["risk_color"],
            "recommended_action": risk_info["action"],
            "order_summary": order_data
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict delays for a batch of orders.
        """
        X_encoded = self.preprocessor.transform(df)
        X_aligned = X_encoded.reindex(columns=self.feature_names, fill_value=0)
        preds = self.model.predict(X_aligned)
        preds = np.clip(preds, 0.01, 20.00)

        result_df = df.copy()
        result_df["predicted_delay_minutes"] = np.round(preds, 2)
        result_df["risk_level"] = [self.classify_risk(p)["risk_level"] for p in preds]
        return result_df


if __name__ == "__main__":
    # Example order from Slide 9
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

    predictor = DeliveryDelayPredictor()
    result = predictor.predict_single(example_order)

    print("\n" + "=" * 50)
    print("SLIDE 9 EXAMPLE ORDER PREDICTION")
    print("=" * 50)
    print(f"Distance: {example_order['delivery_distance']} km")
    print(f"Traffic: {example_order['traffic_condition']} | Weather: {example_order['weather_condition']}")
    print(f"Delivery Method: {example_order['delivery_method']} | Route Efficiency: {example_order['route_efficiency']}")
    print(f"Order Value: ₹{example_order['order_value']}")
    print("-" * 50)
    print(f"PREDICTED DELAY: {result['predicted_delay_minutes']} minutes")
    print(f"Risk Assessment: {result['risk_level']}")
    print(f"Recommended Action: {result['recommended_action']}")
    print("=" * 50 + "\n")

