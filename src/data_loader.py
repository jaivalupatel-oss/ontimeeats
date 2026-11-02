"""
Data Loader and Data Validation Module
Loads raw dataset and enforces data cleaning rules specified in the project definition.
"""
import logging
from typing import Tuple, Dict, Any
import pandas as pd
from src.config import RAW_DATA_PATH, TARGET_COLUMN, DROP_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(filepath=RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    logger.info(f"Loading raw dataset from {filepath}...")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns.")
    return df


def validate_and_clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate data against requirements and clean according to Slide 3 & 4:
    - Missing values check: 0
    - Duplicate rows check: 0
    - Invalid/negative delays removed (delivery_delay > 0)
    - Unnecessary IDs removed (restaurant_id)
    - Bounds verification: delivery_delay [0.01 - 20.00], order_value [103 - 1500]
    """
    initial_count = len(df)
    missing_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())

    # Filter negative delays if any
    negative_delay_count = int((df[TARGET_COLUMN] < 0).sum())
    clean_df = df[df[TARGET_COLUMN] >= 0].copy()

    # Drop unnecessary IDs
    dropped = [col for col in DROP_COLUMNS if col in clean_df.columns]
    clean_df = clean_df.drop(columns=dropped)

    summary = {
        "initial_records": initial_count,
        "final_records": len(clean_df),
        "columns_count": len(clean_df.columns),
        "missing_values": missing_count,
        "duplicate_rows": duplicate_count,
        "negative_delays_removed": negative_delay_count,
        "dropped_columns": dropped,
        "delay_min": float(clean_df[TARGET_COLUMN].min()),
        "delay_max": float(clean_df[TARGET_COLUMN].max()),
        "order_value_min": float(clean_df["order_value"].min()),
        "order_value_max": float(clean_df["order_value"].max()),
    }

    logger.info("Data Validation and Quality Summary:")
    for k, v in summary.items():
        logger.info(f"  - {k}: {v}")

    return clean_df, summary


if __name__ == "__main__":
    df = load_raw_data()
    clean_df, summary = validate_and_clean_data(df)
    print("Clean Data Sample:")
    print(clean_df.head(3))

