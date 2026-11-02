# Food Delivery Delay Prediction - Model Evaluation Report

**Project**: Food Delivery Delay Prediction  
**Author**: Jaival Patel | B.Tech CSE (23000300)  
**Algorithm**: Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`)  
**Task Type**: Supervised Continuous Regression  

---

## 1. Executive Summary
This project predicts customer food delivery delays in minutes (`delivery_delay`) using 19 contextual features covering order specifics, route efficiency, environmental factors (traffic and weather), and customer order history.

The machine learning pipeline implements strict data cleaning (eliminating IDs and preventing data leakage) and an 80/20 train-test split (160,000 training records, 40,000 testing records). Preprocessing transformations (temporal feature extraction and One-Hot Encoding) were strictly fitted onto the training partition.

---

## 2. Test Set Evaluation Metrics (Unseen 40,000 Records)

| Metric | Value | Description |
| :--- | :--- | :--- |
| **MAE** | **0.2705 minutes** | Mean Absolute Error — average absolute deviation between actual and predicted delivery delays. |
| **RMSE** | **0.392 minutes** | Root Mean Squared Error — penalizes larger outliers and extreme prediction errors. |
| **R² Score** | **0.9954** | Coefficient of Determination — indicates 99.54% of delay variance is explained by the model. |
| **Median AE** | **0.1946 minutes** | Median Absolute Error — robust central tendency of error. |
| **Explained Variance** | **0.9954** | Proportion of variance accounted for by the regressor. |

---

## 3. Dataset Integrity & Quality Verification

| Specification | Observed Value | Status |
| :--- | :--- | :--- |
| Total Records | 200,000 | Verified |
| Target Variable Range | 0.01 min to 20.00 min | Verified |
| Order Value Range | ₹103.04 to ₹1,500.00 | Verified |
| Missing Values | 0 | Passed (0 Missing) |
| Duplicate Rows | 0 | Passed (0 Duplicates) |
| Negative Delays | 0 | Passed (0 Invalid) |
| Dropped Identifiers | restaurant_id | Removed to prevent spurious memorization |

---

## 4. Top Feature Importances (MDI)

| feature             |   importance |
|:--------------------|-------------:|
| route_efficiency    |    0.0952491 |
| delivery_distance   |    0.0914968 |
| order_day           |    0.0832898 |
| age                 |    0.0779005 |
| order_history       |    0.0745384 |
| order_month         |    0.0674357 |
| order_dayofweek     |    0.0449245 |
| customer_rating     |    0.0344824 |
| route_taken_Route_1 |    0.0128106 |
| route_type_Mixed    |    0.0119208 |

---

## 5. Model Architecture & Hyperparameters
- **Estimator**: `RandomForestRegressor`
- **Number of Estimators**: 100
- **Max Tree Depth**: 25
- **Min Samples Split**: 3
- **Min Samples Leaf**: 1
- **Training Set Size**: 160,000 samples
- **Training Wall Time**: 74.24 seconds

---

## 6. Real-World Operational Impact
1. **Dynamic ETA Communication**: Real-time delay forecasts allow dynamic adjustment of delivery promises to customers, eliminating negative surprises.
2. **Early Intervention & Dispatch Optimization**: Identifies high-risk deliveries (e.g., low route efficiency + severe weather + high traffic) for priority fleet assignment.
3. **Route & Delivery Mode Decisions**: Insights reveal route efficiency and delivery vehicle modes strongly determine reliability across varying urban traffic conditions.
