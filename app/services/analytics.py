import pandas as pd
from pathlib import Path


DATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "cleaned_transactions.csv"
)


def load_data():
    df = pd.read_csv(DATA_PATH)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    return df


def get_kpis():
    df = load_data()

    total_revenue = df["Sales"].sum()
    total_units = df["Quantity"].sum()
    total_orders = df["Invoice"].nunique()
    total_customers = df["Customer ID"].nunique()

    average_order_value = (
        df.groupby("Invoice")["Sales"]
        .sum()
        .mean()
    )

    return {
        "total_revenue": round(float(total_revenue), 2),
        "total_units": int(total_units),
        "total_orders": int(total_orders),
        "total_customers": int(total_customers),
        "average_order_value": round(float(average_order_value), 2)
    }


def get_monthly_revenue():
    df = load_data()

    monthly_revenue = (
        df.set_index("InvoiceDate")
        .resample("ME")["Sales"]
        .sum()
        .reset_index()
    )

    monthly_revenue["InvoiceDate"] = (
        monthly_revenue["InvoiceDate"]
        .dt.strftime("%Y-%m")
    )

    return monthly_revenue.rename(
        columns={
            "InvoiceDate": "month",
            "Sales": "revenue"
        }
    ).to_dict(orient="records")


def get_top_products(limit=10):
    df = load_data()

    top_products = (
        df.groupby("Description")
        .agg(
            revenue=("Sales", "sum"),
            units_sold=("Quantity", "sum")
        )
        .sort_values("revenue", ascending=False)
        .head(limit)
        .reset_index()
    )

    top_products["revenue"] = (
        top_products["revenue"].round(2)
    )

    return top_products.to_dict(orient="records")