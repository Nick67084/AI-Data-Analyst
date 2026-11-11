from __future__ import annotations

import json
from typing import Any

import pandas as pd
import requests

from app.services.workspace_manager import load_workspace_tables


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

MAX_KPIS = 6
MAX_CHARTS = 6
MAX_FILTERS = 6


# ============================================================
# BASIC HELPERS
# ============================================================

def _normalize_aggregation(value: Any) -> str:
    """Normalize common aggregation names."""
    if value is None:
        return "sum"

    value = str(value).strip().lower()

    aliases = {
        "average": "mean",
        "avg": "mean",
        "distinct_count": "nunique",
        "unique_count": "nunique",
        "count_distinct": "nunique",
    }

    return aliases.get(value, value)


def _safe_string(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def _normalize_column_name(value: Any) -> str:
    return (
        _safe_string(value)
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def _find_column(
    dataframe: pd.DataFrame,
    requested_column: Any,
) -> str | None:
    """
    Find a real DataFrame column using:
    1. exact match
    2. case-insensitive match
    3. normalized match
    """

    if requested_column is None:
        return None

    requested = _safe_string(requested_column)

    # Exact match
    if requested in dataframe.columns:
        return requested

    # Case-insensitive match
    requested_lower = requested.lower()

    for column in dataframe.columns:
        if str(column).lower() == requested_lower:
            return str(column)

    # Normalized match
    requested_normalized = _normalize_column_name(requested)

    for column in dataframe.columns:
        if _normalize_column_name(column) == requested_normalized:
            return str(column)

    return None


def _find_table(
    tables: dict[str, pd.DataFrame],
    requested_table: Any,
) -> tuple[str | None, pd.DataFrame | None]:
    """
    Find a real table using exact, case-insensitive,
    and normalized matching.
    """

    if requested_table is None:
        return None, None

    requested = _safe_string(requested_table)

    # Exact
    if requested in tables:
        return requested, tables[requested]

    # Case-insensitive
    requested_lower = requested.lower()

    for table_name, dataframe in tables.items():
        if table_name.lower() == requested_lower:
            return table_name, dataframe

    # Normalized
    requested_normalized = _normalize_column_name(requested)

    for table_name, dataframe in tables.items():
        if _normalize_column_name(table_name) == requested_normalized:
            return table_name, dataframe

    return None, None


# ============================================================
# WORKSPACE RESOLUTION
# ============================================================

def _get_workspace_id(
    ai_summary: dict[str, Any],
) -> str | None:
    """
    Get the workspace ID from the confirmed workspace summary structure.

    Expected:
        ai_summary["workspace"] = {
            "filename": "api_test_workspace",
            "file_type": "workspace",
            ...
        }
    """

    workspace = ai_summary.get("workspace")

    if isinstance(workspace, dict):
        filename = workspace.get("filename")

        if isinstance(filename, str) and filename.strip():
            return filename.strip()

    # Backward-compatible fallback.
    filename = ai_summary.get("filename")

    if isinstance(filename, str) and filename.strip():
        return filename.strip()

    workspace_id = ai_summary.get("workspace_id")

    if isinstance(workspace_id, str) and workspace_id.strip():
        return workspace_id.strip()

    return None


def _load_actual_workspace_tables(
    ai_summary: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """
    Load the actual DataFrames belonging to the workspace.

    Validation must happen against real DataFrames,
    not only LLM metadata.
    """

    workspace_id = _get_workspace_id(ai_summary)

    if not workspace_id:
        return {}

    try:
        tables = load_workspace_tables(workspace_id)

        if isinstance(tables, dict):
            return tables

    except Exception:
        return {}

    return {}


# ============================================================
# COLUMN DETECTION
# ============================================================

def _find_identifier_column(
    dataframe: pd.DataFrame,
    preferred_keywords: list[str],
) -> str | None:
    """
    Find a likely identifier column.

    Used for:
    - Invoice
    - Customer ID
    - Order ID
    """

    columns = list(dataframe.columns)

    # First look for explicit keyword matches.
    for keyword in preferred_keywords:
        normalized_keyword = _normalize_column_name(keyword)

        for column in columns:
            normalized_column = _normalize_column_name(column)

            if normalized_column == normalized_keyword:
                return str(column)

    # Then substring matching.
    for keyword in preferred_keywords:
        normalized_keyword = _normalize_column_name(keyword)

        for column in columns:
            normalized_column = _normalize_column_name(column)

            if normalized_keyword in normalized_column:
                return str(column)

    return None


def _find_sales_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """
    Find the most likely revenue/sales column.
    """

    preferred = [
        "Sales",
        "Revenue",
        "Amount",
        "Total Sales",
        "Total Revenue",
        "Value",
    ]

    for name in preferred:
        column = _find_column(dataframe, name)

        if column is not None:
            if pd.api.types.is_numeric_dtype(dataframe[column]):
                return column

    # Fallback to numeric columns containing useful keywords.
    keywords = [
        "sales",
        "revenue",
        "amount",
        "total",
        "value",
        "income",
    ]

    for column in dataframe.columns:
        normalized = _normalize_column_name(column)

        if any(keyword in normalized for keyword in keywords):
            if pd.api.types.is_numeric_dtype(dataframe[column]):
                return str(column)

    return None


def _find_order_column(
    dataframe: pd.DataFrame,
) -> str | None:
    return _find_identifier_column(
        dataframe,
        [
            "Invoice",
            "Order ID",
            "OrderID",
            "Order",
            "Transaction ID",
            "TransactionID",
        ],
    )


def _find_customer_column(
    dataframe: pd.DataFrame,
) -> str | None:
    return _find_identifier_column(
        dataframe,
        [
            "Customer ID",
            "CustomerID",
            "Customer",
            "User ID",
            "UserID",
        ],
    )


# ============================================================
# SEMANTIC METRICS
# ============================================================

def _is_average_order_value(
    name: Any,
) -> bool:
    """
    Detect whether a KPI is asking for Average Order Value.
    """

    normalized = _normalize_column_name(name)

    patterns = [
        "averageordervalue",
        "avgordervalue",
        "aov",
        "meanordervalue",
    ]

    return normalized in patterns


def _contains_any_keyword(
    value: Any,
    keywords: list[str],
) -> bool:
    normalized = _normalize_column_name(value)

    return any(
        keyword in normalized
        for keyword in keywords
    )


def _get_metric_family(
    name: Any,
) -> str | None:
    """
    Determine the semantic family requested by a KPI/chart title.

    Returns:
        sales
        profit
        cost
        quantity
        customer
        order
        price
        None
    """

    normalized = _normalize_column_name(name)

    if any(
        keyword in normalized
        for keyword in [
            "revenue",
            "sales",
            "turnover",
            "income",
        ]
    ):
        return "sales"

    if "profit" in normalized:
        return "profit"

    if any(
        keyword in normalized
        for keyword in [
            "cost",
            "expense",
        ]
    ):
        return "cost"

    if any(
        keyword in normalized
        for keyword in [
            "quantity",
            "units",
            "unitssold",
        ]
    ):
        return "quantity"

    if any(
        keyword in normalized
        for keyword in [
            "customer",
            "customers",
        ]
    ):
        return "customer"

    if any(
        keyword in normalized
        for keyword in [
            "order",
            "orders",
            "invoice",
            "invoices",
        ]
    ):
        return "order"

    if "price" in normalized:
        return "price"

    return None


def _get_column_metric_family(
    column_name: Any,
) -> str | None:
    """
    Determine the semantic family of a real data column.
    """

    normalized = _normalize_column_name(column_name)

    if any(
        keyword in normalized
        for keyword in [
            "sales",
            "revenue",
            "turnover",
            "income",
            "amount",
        ]
    ):
        return "sales"

    if "profit" in normalized:
        return "profit"

    if any(
        keyword in normalized
        for keyword in [
            "cost",
            "expense",
        ]
    ):
        return "cost"

    if any(
        keyword in normalized
        for keyword in [
            "quantity",
            "units",
        ]
    ):
        return "quantity"

    if any(
        keyword in normalized
        for keyword in [
            "customerid",
            "userid",
        ]
    ):
        return "customer"

    if any(
        keyword in normalized
        for keyword in [
            "invoice",
            "orderid",
            "transactionid",
        ]
    ):
        return "order"

    if "price" in normalized:
        return "price"

    return None


def _is_semantically_valid_metric(
    metric_name: Any,
    column_name: Any,
) -> tuple[bool, str | None]:
    """
    Check whether a KPI/chart metric name is compatible
    with the selected data column.

    This is intentionally conservative.

    Example:
        Total Revenue + Price
        -> rejected

        Total Revenue + Sales
        -> accepted
    """

    metric_family = _get_metric_family(metric_name)
    column_family = _get_column_metric_family(column_name)

    # If we cannot determine the meaning, allow the
    # structural validator to handle it.
    if metric_family is None:
        return True, None

    # Customer/order metrics are normally handled by
    # their identifier-specific logic.
    if metric_family in {
        "customer",
        "order",
    }:
        return True, None

    if column_family is None:
        return True, None

    if metric_family == column_family:
        return True, None

    # Price is intentionally NOT treated as revenue/sales.
    #
    # This prevents:
    # Total Revenue -> Price
    #
    # even though Price is numeric.

    return False, (
        f"Metric '{_safe_string(metric_name)}' is semantically "
        f"incompatible with column '{_safe_string(column_name)}'. "
        f"The requested metric represents '{metric_family}', "
        f"but the selected column represents '{column_family}'."
    )


# ============================================================
# LLM CATALOG
# ============================================================

def _build_table_catalog(
    tables: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """
    Build compact metadata for the LLM.

    The LLM gets metadata rather than the full dataset.
    """

    catalog = []

    for table_name, dataframe in tables.items():
        columns = []

        for column in dataframe.columns:
            series = dataframe[column]

            if pd.api.types.is_datetime64_any_dtype(series):
                detected_type = "datetime"

            elif pd.api.types.is_numeric_dtype(series):
                detected_type = "numeric"

            elif pd.api.types.is_bool_dtype(series):
                detected_type = "boolean"

            else:
                detected_type = "categorical/text"

            columns.append(
                {
                    "name": str(column),
                    "dtype": str(series.dtype),
                    "detected_type": detected_type,
                    "null_percentage": round(
                        float(series.isna().mean() * 100),
                        2,
                    ),
                    "unique_count": int(
                        series.nunique(dropna=True)
                    ),
                }
            )

        catalog.append(
            {
                "table_name": table_name,
                "rows": int(len(dataframe)),
                "columns": int(len(dataframe.columns)),
                "column_details": columns,
            }
        )

    return catalog


# ============================================================
# LLM PROMPT
# ============================================================

def _build_prompt(
    catalog: list[dict[str, Any]],
    llm_context: str,
    user_request: str = "",
) -> str:

    request_text = (
        user_request.strip()
        if user_request
        else "Create a useful general business analytics dashboard."
    )

    return f"""
You are an AI Data Analyst.

Your job is to design a dashboard configuration from the provided
workspace metadata.

IMPORTANT:

- You are NOT given the full dataset.
- Do NOT invent table names.
- Do NOT invent column names.
- Use ONLY tables and columns appearing in the metadata.
- The backend will independently validate your proposal.
- Do not calculate numeric KPI values yourself.
- Return only JSON.

USER REQUEST:

{request_text}

WORKSPACE METADATA:

{json.dumps(catalog, indent=2, default=str)}

ADDITIONAL ANALYTICAL CONTEXT:

{llm_context}

DASHBOARD RULES:

1. KPIs

Use useful business KPIs such as:

- total sales/revenue
- total units/quantity
- total orders
- total customers
- average order value

For distinct orders/customers:

- aggregation should normally be "nunique".

IMPORTANT AVERAGE ORDER VALUE RULE:

Average Order Value is NOT the average of the Price column.

Correct concept:

1. Group transaction rows by order/invoice.
2. Sum sales/revenue for each order.
3. Calculate the mean of those order totals.

Therefore, when requesting Average Order Value:

- choose the actual sales/revenue column as "column"
- use aggregation "mean"
- the backend may attach the semantic metric
  "average_order_value"
- the backend will calculate the metric correctly.

IMPORTANT SEMANTIC MEASURE RULE:

Choose the actual business measure represented by the KPI.

Examples:

- Total Sales -> Sales/Revenue/Amount
- Total Revenue -> Revenue/Sales/Amount
- Total Profit -> Profit
- Total Cost -> Cost/Expense
- Total Quantity -> Quantity/Units

Do NOT use a Price column as Total Revenue or Total Sales
unless the column itself actually represents total sales/revenue.

2. CHARTS

Supported chart types:

- line
- bar
- area
- pie

Each chart should contain:

- type
- title
- table
- x
- y
- aggregation

Use:

- datetime + numeric measure for trends
- categorical + numeric measure for category comparisons
- product + numeric measure for product analysis

The y column must represent the metric described by the chart title.

Examples:

- Monthly Revenue -> Sales/Revenue column
- Sales by Country -> Sales/Revenue column
- Profit by Category -> Profit column
- Quantity by Product -> Quantity/Units column

3. FILTERS

Use real categorical or datetime columns.

4. DO NOT:

- invent columns
- invent tables
- generate Python code
- generate React code
- generate SQL
- return markdown

Return exactly this JSON structure:

{{
  "title": "Dashboard title",
  "description": "Short dashboard description",
  "kpis": [
    {{
      "name": "Total Sales",
      "table": "actual_table_name",
      "column": "actual_column_name",
      "aggregation": "sum"
    }}
  ],
  "charts": [
    {{
      "type": "line",
      "title": "Monthly Sales",
      "table": "actual_table_name",
      "x": "actual_date_column",
      "y": "actual_sales_column",
      "aggregation": "sum"
    }}
  ],
  "filters": [
    {{
      "column": "actual_column_name",
      "table": "actual_table_name"
    }}
  ]
}}
"""


# ============================================================
# OLLAMA
# ============================================================

def _call_ollama(
    prompt: str,
) -> dict[str, Any]:

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    result = response.json()

    raw_response = result.get("response", "")

    if not raw_response:
        raise ValueError(
            "Ollama returned an empty response."
        )

    if isinstance(raw_response, dict):
        return raw_response

    return json.loads(raw_response)


# ============================================================
# KPI VALIDATION
# ============================================================

def _validate_kpi(
    kpi: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any] | None, str | None]:

    name = _safe_string(
        kpi.get("name")
    )

    table_requested = kpi.get("table")
    column_requested = kpi.get("column")

    table_name, dataframe = _find_table(
        tables,
        table_requested,
    )

    if dataframe is None:
        return None, (
            f"KPI '{name}' references unknown table "
            f"'{table_requested}'."
        )

    column_name = _find_column(
        dataframe,
        column_requested,
    )

    if column_name is None:
        return None, (
            f"KPI '{name}' references unknown column "
            f"'{column_requested}'."
        )

    aggregation = _normalize_aggregation(
        kpi.get("aggregation")
    )

    allowed_aggregations = {
        "sum",
        "mean",
        "count",
        "nunique",
        "min",
        "max",
    }

    if aggregation not in allowed_aggregations:
        return None, (
            f"KPI '{name}' uses unsupported aggregation "
            f"'{aggregation}'."
        )

    # --------------------------------------------------------
    # Semantic validation
    # --------------------------------------------------------

    validated = {
        "name": name,
        "table": table_name,
        "column": column_name,
        "aggregation": aggregation,
    }

    # --------------------------------------------------------
    # Average Order Value
    # --------------------------------------------------------

    if _is_average_order_value(name):

        sales_column = _find_sales_column(
            dataframe
        )

        order_column = _find_order_column(
            dataframe
        )

        if sales_column is None:
            return None, (
                f"KPI '{name}' requires a sales/revenue "
                "column, but none was found."
            )

        if order_column is None:
            return None, (
                f"KPI '{name}' requires an order/invoice "
                "identifier, but none was found."
            )

        validated["column"] = sales_column
        validated["aggregation"] = "mean"
        validated["semantic_metric"] = (
            "average_order_value"
        )
        validated["order_column"] = order_column

        return validated, None

    # --------------------------------------------------------
    # Total orders
    # --------------------------------------------------------

    normalized_name = _normalize_column_name(
        name
    )

    if (
        "totalorders" in normalized_name
        or normalized_name == "orders"
    ):

        order_column = _find_order_column(
            dataframe
        )

        if order_column is None:
            return None, (
                f"KPI '{name}' requires an order/invoice "
                "identifier, but none was found."
            )

        validated["column"] = order_column
        validated["aggregation"] = "nunique"

        return validated, None

    # --------------------------------------------------------
    # Total customers
    # --------------------------------------------------------

    if (
        "totalcustomers" in normalized_name
        or normalized_name == "customers"
    ):

        customer_column = _find_customer_column(
            dataframe
        )

        if customer_column is None:
            return None, (
                f"KPI '{name}' requires a customer identifier, "
                "but none was found."
            )

        validated["column"] = customer_column
        validated["aggregation"] = "nunique"

        return validated, None

    # --------------------------------------------------------
    # General semantic validation
    # --------------------------------------------------------

    semantic_valid, semantic_error = (
        _is_semantically_valid_metric(
            metric_name=name,
            column_name=column_name,
        )
    )

    if not semantic_valid:
        return None, semantic_error

    return validated, None


# ============================================================
# CHART VALIDATION
# ============================================================

def _validate_chart(
    chart: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any] | None, str | None]:

    chart_type = _safe_string(
        chart.get("type")
    ).lower()

    if chart_type not in {
        "line",
        "bar",
        "area",
        "pie",
    }:
        return None, (
            f"Unsupported chart type '{chart_type}'."
        )

    title = _safe_string(
        chart.get("title")
    )

    table_requested = chart.get("table")

    table_name, dataframe = _find_table(
        tables,
        table_requested,
    )

    if dataframe is None:
        return None, (
            f"Chart '{title}' references unknown table "
            f"'{table_requested}'."
        )

    x_requested = chart.get("x")
    y_requested = chart.get("y")

    x_column = _find_column(
        dataframe,
        x_requested,
    )

    if x_column is None:
        return None, (
            f"Chart '{title}' references unknown x column "
            f"'{x_requested}'."
        )

    y_column = _find_column(
        dataframe,
        y_requested,
    )

    if y_column is None:
        return None, (
            f"Chart '{title}' references unknown y column "
            f"'{y_requested}'."
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe[y_column]
    ):
        return None, (
            f"Chart '{title}' uses non-numeric y column "
            f"'{y_column}'."
        )

    aggregation = _normalize_aggregation(
        chart.get("aggregation")
    )

    if aggregation not in {
        "sum",
        "mean",
        "count",
        "nunique",
        "min",
        "max",
    }:
        return None, (
            f"Chart '{title}' uses unsupported aggregation "
            f"'{aggregation}'."
        )

    # --------------------------------------------------------
    # Semantic validation for chart metric
    # --------------------------------------------------------

    semantic_valid, semantic_error = (
        _is_semantically_valid_metric(
            metric_name=title,
            column_name=y_column,
        )
    )

    if not semantic_valid:
        return None, semantic_error

    validated = {
        "type": chart_type,
        "title": title,
        "table": table_name,
        "x": x_column,
        "y": y_column,
        "aggregation": aggregation,
    }

    # Optional chart limit.
    if chart.get("limit") is not None:

        try:
            limit = int(
                chart["limit"]
            )

            if limit > 0:
                validated["limit"] = min(
                    limit,
                    50,
                )

        except (TypeError, ValueError):
            pass

    return validated, None


# ============================================================
# FILTER VALIDATION
# ============================================================

def _validate_filter(
    filter_item: Any,
    tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any] | None, str | None]:

    if isinstance(filter_item, str):

        requested_column = filter_item

        # Search all tables for a matching column.
        for table_name, dataframe in tables.items():

            column_name = _find_column(
                dataframe,
                requested_column,
            )

            if column_name is not None:
                return {
                    "table": table_name,
                    "column": column_name,
                }, None

        return None, (
            f"Filter references unknown column "
            f"'{requested_column}'."
        )

    if not isinstance(filter_item, dict):
        return None, (
            "Filter must be a string or object."
        )

    table_requested = filter_item.get("table")
    column_requested = filter_item.get("column")

    table_name, dataframe = _find_table(
        tables,
        table_requested,
    )

    if dataframe is None:
        return None, (
            f"Filter references unknown table "
            f"'{table_requested}'."
        )

    column_name = _find_column(
        dataframe,
        column_requested,
    )

    if column_name is None:
        return None, (
            f"Filter references unknown column "
            f"'{column_requested}'."
        )

    return {
        "table": table_name,
        "column": column_name,
    }, None


# ============================================================
# DASHBOARD VALIDATION
# ============================================================

def _validate_dashboard_plan(
    plan: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:

    raw_kpis = plan.get(
        "kpis",
        []
    )

    raw_charts = plan.get(
        "charts",
        []
    )

    raw_filters = plan.get(
        "filters",
        []
    )

    if not isinstance(raw_kpis, list):
        raw_kpis = []

    if not isinstance(raw_charts, list):
        raw_charts = []

    if not isinstance(raw_filters, list):
        raw_filters = []

    accepted_kpis = []
    rejected_kpis = []

    accepted_charts = []
    rejected_charts = []

    accepted_filters = []
    rejected_filters = []

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    for kpi in raw_kpis[:MAX_KPIS]:

        if not isinstance(kpi, dict):

            rejected_kpis.append(
                {
                    "item": kpi,
                    "reason": "KPI must be an object.",
                }
            )

            continue

        validated, error = _validate_kpi(
            kpi,
            tables,
        )

        if error:

            rejected_kpis.append(
                {
                    "item": kpi,
                    "reason": error,
                }
            )

        else:

            accepted_kpis.append(
                validated
            )

    # --------------------------------------------------------
    # Ensure Average Order Value exists when possible.
    # --------------------------------------------------------

    has_aov = any(
        item.get("semantic_metric")
        == "average_order_value"
        for item in accepted_kpis
    )

    if not has_aov:

        for table_name, dataframe in tables.items():

            sales_column = _find_sales_column(
                dataframe
            )

            order_column = _find_order_column(
                dataframe
            )

            if (
                sales_column is not None
                and order_column is not None
            ):

                accepted_kpis.append(
                    {
                        "name": "Average Order Value",
                        "table": table_name,
                        "column": sales_column,
                        "aggregation": "mean",
                        "semantic_metric": (
                            "average_order_value"
                        ),
                        "order_column": order_column,
                    }
                )

                break

    accepted_kpis = accepted_kpis[:MAX_KPIS]

    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------

    for chart in raw_charts[:MAX_CHARTS]:

        if not isinstance(chart, dict):

            rejected_charts.append(
                {
                    "item": chart,
                    "reason": "Chart must be an object.",
                }
            )

            continue

        validated, error = _validate_chart(
            chart,
            tables,
        )

        if error:

            rejected_charts.append(
                {
                    "item": chart,
                    "reason": error,
                }
            )

        else:

            accepted_charts.append(
                validated
            )

    accepted_charts = accepted_charts[:MAX_CHARTS]

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    for filter_item in raw_filters[:MAX_FILTERS]:

        validated, error = _validate_filter(
            filter_item,
            tables,
        )

        if error:

            rejected_filters.append(
                {
                    "item": filter_item,
                    "reason": error,
                }
            )

        else:

            accepted_filters.append(
                validated
            )

    accepted_filters = accepted_filters[:MAX_FILTERS]

    # --------------------------------------------------------
    # Final dashboard
    # --------------------------------------------------------

    validated_plan = {
        "title": _safe_string(
            plan.get("title")
        ) or "AI Generated Dashboard",

        "description": _safe_string(
            plan.get("description")
        ),

        "kpis": accepted_kpis,

        "charts": accepted_charts,

        "filters": accepted_filters,
    }

    validation = {
        "total_kpis": len(raw_kpis),
        "accepted_kpis": len(accepted_kpis),
        "rejected_kpis": len(rejected_kpis),

        "total_charts": len(raw_charts),
        "accepted_charts": len(accepted_charts),
        "rejected_charts": len(rejected_charts),

        "total_filters": len(raw_filters),
        "accepted_filters": len(accepted_filters),
        "rejected_filters": len(rejected_filters),
    }

    return {
        "plan": validated_plan,
        "validation": validation,
        "rejected_kpis": rejected_kpis,
        "rejected_charts": rejected_charts,
        "rejected_filters": rejected_filters,
    }


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def generate_dashboard_plan(
    ai_summary: dict[str, Any],
    llm_context: str,
    user_request: str = "",
) -> dict[str, Any]:
    """
    Generate and validate a dashboard plan.

    Architecture:

        AI summary
             ↓
        workspace filename
             ↓
        real workspace DataFrames
             ↓
        LLM proposes dashboard
             ↓
        deterministic semantic validation
             ↓
        validated dashboard JSON
    """

    if not isinstance(ai_summary, dict):
        raise TypeError(
            "ai_summary must be a dictionary."
        )

    # --------------------------------------------------------
    # 1. Load REAL workspace tables.
    # --------------------------------------------------------

    tables = _load_actual_workspace_tables(
        ai_summary
    )

    if not tables:
        raise ValueError(
            "Could not load the actual workspace tables. "
            "Make sure ai_summary['workspace']['filename'] "
            "contains a valid workspace ID."
        )

    # --------------------------------------------------------
    # 2. Build metadata for the LLM.
    # --------------------------------------------------------

    catalog = _build_table_catalog(
        tables
    )

    # --------------------------------------------------------
    # 3. Build prompt.
    # --------------------------------------------------------

    prompt = _build_prompt(
        catalog=catalog,
        llm_context=llm_context,
        user_request=user_request,
    )

    # --------------------------------------------------------
    # 4. Ask Ollama to propose dashboard.
    # --------------------------------------------------------

    try:

        raw_plan = _call_ollama(
            prompt
        )

    except Exception as exc:

        raise RuntimeError(
            f"Dashboard generation failed while calling Ollama: {exc}"
        ) from exc

    if not isinstance(raw_plan, dict):
        raise ValueError(
            "Ollama did not return a valid dashboard object."
        )

    # --------------------------------------------------------
    # 5. Deterministically validate against real data.
    # --------------------------------------------------------

    validated = _validate_dashboard_plan(
        plan=raw_plan,
        tables=tables,
    )

    # --------------------------------------------------------
    # 6. Return stable public structure.
    # --------------------------------------------------------

    return {
        "dashboard": validated["plan"],
        "validation": validated["validation"],
        "rejected_kpis": validated["rejected_kpis"],
        "rejected_charts": validated["rejected_charts"],
        "rejected_filters": validated["rejected_filters"],
    }