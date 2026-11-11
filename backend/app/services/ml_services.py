import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = BASE_DIR / "data"


FORECAST_MODEL_PATH = (
    DATA_DIR / "random_forest_sales_model.pkl"
)

ANOMALY_MODEL_PATH = (
    DATA_DIR / "isolation_forest_model.pkl"
)

KMEANS_MODEL_PATH = (
    DATA_DIR / "customer_kmeans_model.pkl"
)

SCALER_PATH = (
    DATA_DIR / "customer_rfm_scaler.pkl"
)

FORECAST_DATA_PATH = (
    DATA_DIR / "future_revenue_forecast.csv"
)

ANOMALY_DATA_PATH = (
    DATA_DIR / "sales_anomalies.csv"
)

CUSTOMER_DATA_PATH = (
    DATA_DIR / "customer_segments.csv"
)


def get_forecast():
    forecast_df = pd.read_csv(FORECAST_DATA_PATH)

    forecast_df["Date"] = pd.to_datetime(
        forecast_df["Date"]
    ).dt.strftime("%Y-%m-%d")

    return forecast_df.to_dict(
        orient="records"
    )


def get_anomalies():
    anomaly_df = pd.read_csv(ANOMALY_DATA_PATH)

    anomaly_df["Date"] = pd.to_datetime(
        anomaly_df["Date"]
    ).dt.strftime("%Y-%m-%d")

    anomalies = anomaly_df[
        anomaly_df["AnomalyLabel"] == "Anomaly"
    ].copy()

    anomalies = anomalies.sort_values(
        "Revenue",
        ascending=False
    )

    return anomalies.to_dict(
        orient="records"
    )


def get_customer_segments():
    customer_df = pd.read_csv(
        CUSTOMER_DATA_PATH
    )

    segment_summary = (
        customer_df.groupby("Segment")
        .agg(
            customers=("Customer ID", "count"),
            avg_recency=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
            avg_monetary=("Monetary", "mean")
        )
        .round(2)
        .reset_index()
    )

    return segment_summary.to_dict(
        orient="records"
    )


def get_ml_model_status():
    return {
        "forecast_model": FORECAST_MODEL_PATH.exists(),
        "anomaly_model": ANOMALY_MODEL_PATH.exists(),
        "customer_kmeans_model": KMEANS_MODEL_PATH.exists(),
        "customer_scaler": SCALER_PATH.exists()
    }