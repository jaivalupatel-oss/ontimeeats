"""
Flask Web Application for Food Delivery Delay Prediction
Serves interactive prediction dashboard, batch fleet dispatch inference,
what-if route simulator, and REST API endpoints.
Author: Jaival Patel (B.Tech CSE | 23000300)
"""
import io
import sys
import os
import json
from pathlib import Path
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory, Response

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import DeliveryDelayPredictor
from src.config import FIGURES_DIR, METADATA_SAVE_PATH, DATA_DIR

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "app" / "templates"),
    static_folder=str(PROJECT_ROOT / "app" / "static")
)

# Global predictor instance
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = DeliveryDelayPredictor()
    return predictor


@app.route("/")
def home():
    """Render main interactive dashboard."""
    metadata = {}
    if METADATA_SAVE_PATH.exists():
        try:
            with open(METADATA_SAVE_PATH, "r") as f:
                metadata = json.load(f)
        except Exception:
            pass
    return render_template("index.html", metadata=metadata)


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for single order delay prediction."""
    try:
        data = request.get_json(force=True)
        pred_engine = get_predictor()
        result = pred_engine.predict_single(data)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    API endpoint for batch prediction.
    Accepts CSV file upload or JSON payload of orders.
    Returns predicted delays, risk levels, and dispatch summary statistics.
    """
    try:
        pred_engine = get_predictor()

        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"status": "error", "message": "No file selected"}), 400
            df = pd.read_csv(file)
        elif request.is_json:
            json_data = request.get_json(force=True)
            if isinstance(json_data, dict) and "orders" in json_data:
                df = pd.DataFrame(json_data["orders"])
            elif isinstance(json_data, list):
                df = pd.DataFrame(json_data)
            else:
                return jsonify({"status": "error", "message": "Invalid JSON format for batch"}), 400
        else:
            return jsonify({"status": "error", "message": "Expected CSV file or JSON array"}), 400

        # Run batch prediction
        results_df = pred_engine.predict_batch(df)

        # Calculate fleet statistics
        total_orders = len(results_df)
        avg_delay = round(float(results_df["predicted_delay_minutes"].mean()), 2)
        min_delay = round(float(results_df["predicted_delay_minutes"].min()), 2)
        max_delay = round(float(results_df["predicted_delay_minutes"].max()), 2)

        risk_counts = results_df["risk_level"].value_counts().to_dict()
        low_count = int(risk_counts.get("Low Delay Risk", 0))
        mod_count = int(risk_counts.get("Moderate Delay Risk", 0))
        high_count = int(risk_counts.get("High Delay Risk", 0))

        # Add risk colors and actions to each record
        records = results_df.to_dict(orient="records")
        for rec in records:
            info = pred_engine.classify_risk(rec["predicted_delay_minutes"])
            rec["risk_color"] = info["risk_color"]
            rec["recommended_action"] = info["action"]

        summary = {
            "total_orders": total_orders,
            "average_delay_minutes": avg_delay,
            "min_delay_minutes": min_delay,
            "max_delay_minutes": max_delay,
            "low_risk_count": low_count,
            "moderate_risk_count": mod_count,
            "high_risk_count": high_count,
            "high_risk_percentage": round((high_count / total_orders) * 100, 1) if total_orders else 0
        }

        return jsonify({
            "status": "success",
            "summary": summary,
            "records": records
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/simulate_routes", methods=["POST"])
def simulate_routes():
    """
    Simulates delay outcomes across different delivery methods and routes
    for a given order context to support dispatch optimization.
    """
    try:
        base_order = request.get_json(force=True)
        pred_engine = get_predictor()

        methods = ["Bike", "Car", "Walk"]
        routes = [
            ("Route_1", "Express Arterial"),
            ("Route_2", "City Center"),
            ("Route_3", "Ring Road"),
            ("Route_4", "Residential Corridors"),
            ("Route_5", "Industrial Bypass")
        ]

        scenarios = []
        for method in methods:
            # Skip Walk if distance > 4 km as it's operationally unfeasible
            if method == "Walk" and float(base_order.get("delivery_distance", 5)) > 4.0:
                continue

            for route_code, route_name in routes:
                order_variant = dict(base_order)
                order_variant["delivery_method"] = method
                order_variant["route_taken"] = route_code

                # Align corridor type with method
                if method == "Bike":
                    order_variant["route_type"] = "Bike-friendly"
                    eff = 0.65
                elif method == "Car":
                    order_variant["route_type"] = "Car-only"
                    eff = 0.50 if "Route_3" in route_code or "Route_1" in route_code else 0.40
                else:
                    order_variant["route_type"] = "Mixed"
                    eff = 0.75
                order_variant["route_efficiency"] = eff

                res = pred_engine.predict_single(order_variant)
                scenarios.append({
                    "delivery_method": method,
                    "route_code": route_code,
                    "route_name": route_name,
                    "predicted_delay": res["predicted_delay_minutes"],
                    "risk_level": res["risk_level"],
                    "risk_color": res["risk_color"]
                })

        scenarios.sort(key=lambda s: s["predicted_delay"])

        return jsonify({
            "status": "success",
            "optimal_choice": scenarios[0] if scenarios else None,
            "scenarios": scenarios
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/sample_batch_data", methods=["GET"])
def sample_batch_data():
    """Return pre-packaged sample fleet batch orders."""
    sample_path = DATA_DIR / "sample_orders_batch.csv"
    if not sample_path.exists():
        return jsonify({"status": "error", "message": "Sample batch file not found"}), 404
    df = pd.read_csv(sample_path)
    return jsonify({"status": "success", "orders": df.to_dict(orient="records")})


@app.route("/static/reports/figures/<path:filename>")
def serve_figure(filename):
    """Serve dynamically generated diagnostic charts."""
    return send_from_directory(FIGURES_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Food Delivery Delay Prediction Web App on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
