from pathlib import Path
from typing import Any

import pandas as pd

from app.services.workspace_manager import (
    load_workspace_tables,
)


SUPPORTED_AGGREGATIONS = {
    "sum",
    "mean",
    "count",
    "nunique",
    "min",
    "max",
}


SUPPORTED_CHART_TYPES = {
    "line",
    "bar",
    "pie",
    "area",
}


# ============================================================
# BASIC AGGREGATION
# ============================================================

def apply_aggregation(
    series: pd.Series,
    aggregation: str,
) -> float | int:
    """
    Apply a supported aggregation to a pandas Series.

    These calculations are deterministic and do not
    involve the LLM.
    """

    aggregation = str(
        aggregation
    ).lower()

    if aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation: {aggregation}"
        )

    if aggregation == "sum":
        result = series.sum()

    elif aggregation == "mean":
        result = series.mean()

    elif aggregation == "count":
        result = series.count()

    elif aggregation == "nunique":
        result = series.nunique(
            dropna=True
        )

    elif aggregation == "min":
        result = series.min()

    elif aggregation == "max":
        result = series.max()

    else:
        raise ValueError(
            f"Unsupported aggregation: {aggregation}"
        )

    if pd.isna(result):
        return 0

    if isinstance(result, float):
        return round(
            float(result),
            2,
        )

    if isinstance(result, int):
        return int(result)

    try:
        return float(result)

    except (TypeError, ValueError):
        return result


# ============================================================
# JSON SAFETY
# ============================================================

def make_json_safe(
    value: Any,
) -> Any:
    """
    Convert pandas/numpy values into values that
    FastAPI can safely serialize as JSON.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(
        value,
        (
            pd.Timestamp,
            pd.DatetimeIndex,
        ),
    ):
        return str(value)

    if hasattr(
        value,
        "item",
    ):
        try:
            return value.item()

        except (
            ValueError,
            TypeError,
        ):
            pass

    return value


# ============================================================
# TABLE / COLUMN HELPERS
# ============================================================

def get_table(
    tables: dict[str, pd.DataFrame],
    table_name: str,
) -> pd.DataFrame:
    """
    Return a table from the workspace.
    """

    if table_name not in tables:
        raise ValueError(
            f"Table '{table_name}' was not found "
            "in the workspace."
        )

    dataframe = tables[table_name]

    # Store table name for better error messages.
    dataframe.attrs["table_name"] = table_name

    return dataframe


def get_column(
    dataframe: pd.DataFrame,
    column_name: str,
) -> pd.Series:
    """
    Return a column from a dataframe.
    """

    if column_name not in dataframe.columns:
        raise ValueError(
            f"Column '{column_name}' was not found "
            f"in table "
            f"'{dataframe.attrs.get('table_name', 'unknown')}'."
        )

    return dataframe[column_name]


# ============================================================
# SEMANTIC COLUMN HELPERS
# ============================================================

def normalize_column_name(
    column_name: Any,
) -> str:
    return (
        str(column_name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def find_order_column(
    dataframe: pd.DataFrame,
    preferred_column: str | None = None,
) -> str | None:
    """
    Find an order/invoice identifier.

    Prefer the column explicitly supplied by the
    dashboard generator.
    """

    if preferred_column:
        if preferred_column in dataframe.columns:
            return preferred_column

    preferred_names = [
        "Invoice",
        "Order ID",
        "OrderID",
        "Order",
        "Transaction ID",
        "TransactionID",
    ]

    # Exact / normalized matching.
    for preferred in preferred_names:

        normalized_preferred = normalize_column_name(
            preferred
        )

        for column in dataframe.columns:

            if (
                normalize_column_name(column)
                == normalized_preferred
            ):
                return str(column)

    # Substring matching.
    keywords = [
        "invoice",
        "orderid",
        "transactionid",
    ]

    for column in dataframe.columns:

        normalized = normalize_column_name(
            column
        )

        if any(
            keyword in normalized
            for keyword in keywords
        ):
            return str(column)

    return None


def find_sales_column(
    dataframe: pd.DataFrame,
    preferred_column: str | None = None,
) -> str | None:
    """
    Find the most likely sales/revenue column.

    Prefer the column explicitly supplied by
    the dashboard generator.
    """

    if preferred_column:

        if preferred_column in dataframe.columns:

            if pd.api.types.is_numeric_dtype(
                dataframe[preferred_column]
            ):
                return preferred_column

    preferred_names = [
        "Sales",
        "Revenue",
        "Amount",
        "Total Sales",
        "Total Revenue",
        "Value",
    ]

    for preferred in preferred_names:

        normalized_preferred = normalize_column_name(
            preferred
        )

        for column in dataframe.columns:

            if (
                normalize_column_name(column)
                == normalized_preferred
            ):
                if pd.api.types.is_numeric_dtype(
                    dataframe[column]
                ):
                    return str(column)

    keywords = [
        "sales",
        "revenue",
        "amount",
        "total",
        "value",
        "income",
    ]

    for column in dataframe.columns:

        normalized = normalize_column_name(
            column
        )

        if any(
            keyword in normalized
            for keyword in keywords
        ):

            if pd.api.types.is_numeric_dtype(
                dataframe[column]
            ):
                return str(column)

    return None


# ============================================================
# SEMANTIC KPI CALCULATIONS
# ============================================================

def calculate_average_order_value(
    dataframe: pd.DataFrame,
    sales_column: str,
    order_column: str,
) -> float:
    """
    Calculate Average Order Value correctly.

    AOV is NOT:

        mean(Sales)

    Instead:

        1. Group transaction rows by order.
        2. Sum sales for each order.
        3. Take the mean of those order totals.
    """

    if sales_column not in dataframe.columns:
        raise ValueError(
            f"Sales column '{sales_column}' "
            "does not exist."
        )

    if order_column not in dataframe.columns:
        raise ValueError(
            f"Order column '{order_column}' "
            "does not exist."
        )

    working = dataframe[
        [
            order_column,
            sales_column,
        ]
    ].copy()

    # Orders without an identifier cannot be used
    # reliably for order-level AOV.
    working = working.dropna(
        subset=[
            order_column,
        ]
    )

    if working.empty:
        return 0.0

    # Ensure sales values are numeric.
    working[sales_column] = pd.to_numeric(
        working[sales_column],
        errors="coerce",
    )

    working = working.dropna(
        subset=[
            sales_column,
        ]
    )

    if working.empty:
        return 0.0

    order_totals = (
        working
        .groupby(
            order_column,
            dropna=True,
        )[sales_column]
        .sum()
    )

    if order_totals.empty:
        return 0.0

    aov = order_totals.mean()

    if pd.isna(aov):
        return 0.0

    return round(
        float(aov),
        2,
    )


def execute_kpi(
    tables: dict[str, pd.DataFrame],
    kpi: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute one validated KPI definition.

    Standard KPIs use the requested aggregation.

    Semantic KPIs, such as Average Order Value,
    use specialized deterministic calculations.
    """

    table_name = kpi.get(
        "table"
    )

    column_name = kpi.get(
        "column"
    )

    aggregation = kpi.get(
        "aggregation",
        "sum",
    )

    name = kpi.get(
        "name",
        "KPI",
    )

    semantic_metric = kpi.get(
        "semantic_metric"
    )

    if not table_name:
        raise ValueError(
            f"KPI '{name}' does not specify a table."
        )

    if not column_name:
        raise ValueError(
            f"KPI '{name}' does not specify a column."
        )

    dataframe = get_table(
        tables,
        table_name,
    )

    # --------------------------------------------------------
    # Average Order Value
    # --------------------------------------------------------

    if semantic_metric == "average_order_value":

        order_column = kpi.get(
            "order_column"
        )

        if not order_column:
            order_column = find_order_column(
                dataframe
            )

        if not order_column:
            raise ValueError(
                f"KPI '{name}' requires an "
                "order/invoice identifier."
            )

        sales_column = find_sales_column(
            dataframe,
            preferred_column=column_name,
        )

        if not sales_column:
            raise ValueError(
                f"KPI '{name}' requires a "
                "sales/revenue column."
            )

        value = calculate_average_order_value(
            dataframe=dataframe,
            sales_column=sales_column,
            order_column=order_column,
        )

        return {
            "name": name,
            "table": table_name,
            "column": sales_column,
            "aggregation": "mean",
            "semantic_metric": "average_order_value",
            "order_column": order_column,
            "value": value,
        }

    # --------------------------------------------------------
    # Standard KPI
    # --------------------------------------------------------

    series = get_column(
        dataframe,
        column_name,
    )

    value = apply_aggregation(
        series,
        aggregation,
    )

    return {
        "name": name,
        "table": table_name,
        "column": column_name,
        "aggregation": aggregation,
        "value": value,
    }


# ============================================================
# DATETIME
# ============================================================

def prepare_datetime_column(
    series: pd.Series,
) -> pd.Series:
    """
    Convert a column to datetime for time-series
    dashboard calculations.
    """

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return series

    converted = pd.to_datetime(
        series,
        format="mixed",
        errors="coerce",
    )

    valid_ratio = (
        converted.notna().mean()
        if len(series) > 0
        else 0
    )

    if valid_ratio < 0.90:
        raise ValueError(
            "Column could not be reliably converted "
            "to datetime."
        )

    return converted


# ============================================================
# LINE / AREA CHART
# ============================================================

def execute_line_or_area_chart(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    aggregation: str,
    chart_type: str,
    limit: int | None = None,
) -> dict[str, Any]:
    """
    Execute a line or area chart.

    Datetime X columns are aggregated daily.
    Other X columns are grouped categorically.
    """

    if x_column not in dataframe.columns:
        raise ValueError(
            f"X column '{x_column}' does not exist."
        )

    if y_column not in dataframe.columns:
        raise ValueError(
            f"Y column '{y_column}' does not exist."
        )

    working = dataframe[
        [
            x_column,
            y_column,
        ]
    ].copy()

    # Try datetime conversion.
    converted_x = pd.to_datetime(
        working[x_column],
        format="mixed",
        errors="coerce",
    )

    conversion_ratio = (
        converted_x.notna().mean()
        if len(working) > 0
        else 0
    )

    if conversion_ratio >= 0.90:

        working[x_column] = converted_x

        working = working.dropna(
            subset=[
                x_column,
                y_column,
            ]
        )

        grouped = (
            working
            .set_index(x_column)
            .resample("D")[y_column]
            .agg(aggregation)
            .reset_index()
        )

        grouped = grouped.dropna(
            subset=[
                y_column,
            ]
        )

        grouped[x_column] = (
            grouped[x_column]
            .dt.strftime(
                "%Y-%m-%d"
            )
        )

    else:

        grouped = (
            working
            .groupby(
                x_column,
                dropna=False,
            )[y_column]
            .agg(aggregation)
            .reset_index()
        )

        grouped = grouped.sort_values(
            y_column,
            ascending=False,
        )

    if limit is not None:

        limit = max(
            1,
            min(
                int(limit),
                100,
            ),
        )

        grouped = grouped.head(
            limit
        )

    data = []

    for _, row in grouped.iterrows():

        data.append(
            {
                "x": make_json_safe(
                    row[x_column]
                ),
                "y": make_json_safe(
                    row[y_column]
                ),
            }
        )

    return {
        "type": chart_type,
        "x": x_column,
        "y": y_column,
        "aggregation": aggregation,
        "data": data,
    }


# ============================================================
# BAR CHART
# ============================================================

def execute_bar_chart(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str | None,
    aggregation: str,
    limit: int | None = None,
) -> dict[str, Any]:
    """
    Execute a categorical bar chart.
    """

    if x_column not in dataframe.columns:
        raise ValueError(
            f"X column '{x_column}' does not exist."
        )

    if y_column is not None:

        if y_column not in dataframe.columns:
            raise ValueError(
                f"Y column '{y_column}' does not exist."
            )

        grouped = (
            dataframe
            .groupby(
                x_column,
                dropna=False,
            )[y_column]
            .agg(aggregation)
            .reset_index()
        )

        grouped = grouped.sort_values(
            y_column,
            ascending=False,
        )

    else:

        grouped = (
            dataframe[
                x_column
            ]
            .value_counts(
                dropna=False
            )
            .reset_index()
        )

        grouped.columns = [
            x_column,
            "value",
        ]

        y_column = "value"

    if limit is not None:

        limit = max(
            1,
            min(
                int(limit),
                100,
            ),
        )

        grouped = grouped.head(
            limit
        )

    data = []

    for _, row in grouped.iterrows():

        data.append(
            {
                "x": make_json_safe(
                    row[x_column]
                ),
                "y": make_json_safe(
                    row[y_column]
                ),
            }
        )

    return {
        "type": "bar",
        "x": x_column,
        "y": y_column,
        "aggregation": aggregation,
        "data": data,
    }


# ============================================================
# PIE CHART
# ============================================================

def execute_pie_chart(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str | None,
    aggregation: str,
    limit: int | None = None,
) -> dict[str, Any]:
    """
    Execute a pie chart.

    A pie chart represents category shares.
    """

    if x_column not in dataframe.columns:
        raise ValueError(
            f"X column '{x_column}' does not exist."
        )

    if y_column is not None:

        if y_column not in dataframe.columns:
            raise ValueError(
                f"Y column '{y_column}' does not exist."
            )

        grouped = (
            dataframe
            .groupby(
                x_column,
                dropna=False,
            )[y_column]
            .agg(aggregation)
            .reset_index()
        )

        value_column = y_column

    else:

        grouped = (
            dataframe[
                x_column
            ]
            .value_counts(
                dropna=False
            )
            .reset_index()
        )

        grouped.columns = [
            x_column,
            "value",
        ]

        value_column = "value"

    grouped = grouped.sort_values(
        value_column,
        ascending=False,
    )

    if limit is not None:

        limit = max(
            1,
            min(
                int(limit),
                20,
            ),
        )

        grouped = grouped.head(
            limit
        )

    data = []

    for _, row in grouped.iterrows():

        data.append(
            {
                "name": make_json_safe(
                    row[x_column]
                ),
                "value": make_json_safe(
                    row[value_column]
                ),
            }
        )

    return {
        "type": "pie",
        "x": x_column,
        "y": y_column,
        "aggregation": aggregation,
        "data": data,
    }


# ============================================================
# CHART EXECUTION
# ============================================================

def execute_chart(
    tables: dict[str, pd.DataFrame],
    chart: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute one validated chart.
    """

    chart_type = str(
        chart.get(
            "type",
            "bar",
        )
    ).lower()

    table_name = chart.get(
        "table"
    )

    x_column = chart.get(
        "x"
    )

    y_column = chart.get(
        "y"
    )

    aggregation = str(
        chart.get(
            "aggregation",
            "sum",
        )
    ).lower()

    limit = chart.get(
        "limit"
    )

    if chart_type not in SUPPORTED_CHART_TYPES:
        raise ValueError(
            f"Unsupported chart type: "
            f"{chart_type}"
        )

    if not table_name:
        raise ValueError(
            "Chart does not specify a table."
        )

    if not x_column:
        raise ValueError(
            "Chart does not specify an X column."
        )

    dataframe = get_table(
        tables,
        table_name,
    )

    if chart_type in {
        "line",
        "area",
    }:

        if not y_column:
            raise ValueError(
                f"{chart_type} chart requires "
                "a Y column."
            )

        return execute_line_or_area_chart(
            dataframe=dataframe,
            x_column=x_column,
            y_column=y_column,
            aggregation=aggregation,
            chart_type=chart_type,
            limit=limit,
        )

    if chart_type == "bar":

        return execute_bar_chart(
            dataframe=dataframe,
            x_column=x_column,
            y_column=y_column,
            aggregation=aggregation,
            limit=limit,
        )

    if chart_type == "pie":

        return execute_pie_chart(
            dataframe=dataframe,
            x_column=x_column,
            y_column=y_column,
            aggregation=aggregation,
            limit=limit,
        )

    raise ValueError(
        f"Unsupported chart type: {chart_type}"
    )


# ============================================================
# DASHBOARD EXECUTION
# ============================================================

def execute_dashboard(
    tables: dict[str, pd.DataFrame],
    dashboard_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute a validated dashboard plan against
    the actual workspace data.

    All numerical calculations are deterministic.
    """

    if not isinstance(
        tables,
        dict,
    ):
        raise TypeError(
            "tables must be a dictionary."
        )

    if not isinstance(
        dashboard_plan,
        dict,
    ):
        raise TypeError(
            "dashboard_plan must be a dictionary."
        )

    # --------------------------------------------------
    # KPIs
    # --------------------------------------------------

    kpis = []

    kpi_errors = []

    for kpi in dashboard_plan.get(
        "kpis",
        [],
    ):

        try:

            result = execute_kpi(
                tables=tables,
                kpi=kpi,
            )

            kpis.append(
                result
            )

        except Exception as exc:

            kpi_errors.append(
                {
                    "kpi": kpi,
                    "error": str(exc),
                }
            )

    # --------------------------------------------------
    # Charts
    # --------------------------------------------------

    charts = []

    chart_errors = []

    for chart in dashboard_plan.get(
        "charts",
        [],
    ):

        try:

            result = execute_chart(
                tables=tables,
                chart=chart,
            )

            result["title"] = chart.get(
                "title",
                "Chart",
            )

            charts.append(
                result
            )

        except Exception as exc:

            chart_errors.append(
                {
                    "chart": chart,
                    "error": str(exc),
                }
            )

    # --------------------------------------------------
    # Filters
    # --------------------------------------------------

    filters = []

    for filter_item in dashboard_plan.get(
        "filters",
        [],
    ):

        if not isinstance(
            filter_item,
            dict,
        ):
            continue

        table_name = filter_item.get(
            "table"
        )

        column_name = filter_item.get(
            "column"
        )

        if (
            table_name not in tables
            or not column_name
            or column_name
            not in tables[
                table_name
            ].columns
        ):
            continue

        filters.append(
            {
                "table": table_name,
                "column": column_name,
            }
        )

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    return {
        "title": dashboard_plan.get(
            "title",
            "Data Analysis Dashboard",
        ),
        "description": dashboard_plan.get(
            "description",
            "",
        ),
        "kpis": kpis,
        "charts": charts,
        "filters": filters,
        "execution": {
            "kpi_count": len(kpis),
            "chart_count": len(charts),
            "filter_count": len(filters),
            "kpi_errors": kpi_errors,
            "chart_errors": chart_errors,
        },
    }


# ============================================================
# WORKSPACE DASHBOARD EXECUTION
# ============================================================

def execute_workspace_dashboard(
    workspace_id: str,
    dashboard_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Load all tables from a persistent workspace
    and execute a validated dashboard plan.
    """

    tables = load_workspace_tables(
        workspace_id
    )

    if not tables:
        raise ValueError(
            "The workspace contains no tables."
        )

    return execute_dashboard(
        tables=tables,
        dashboard_plan=dashboard_plan,
    )