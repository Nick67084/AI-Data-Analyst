from typing import Any

import pandas as pd


# ==================================================
# COLUMN KEYWORD GROUPS
# ==================================================

IDENTIFIER_KEYWORDS = {
    "id",
    "code",
    "key",
    "number",
    "no",
    "num",
    "invoice",
    "stockcode",
    "customerid",
    "productid",
    "orderid",
    "userid",
}


DATE_KEYWORDS = {
    "date",
    "time",
    "datetime",
    "timestamp",
    "year",
    "month",
    "day",
}


MEASURE_KEYWORDS = {
    "sales",
    "revenue",
    "amount",
    "price",
    "profit",
    "cost",
    "quantity",
    "units",
    "value",
    "income",
    "expense",
}


CATEGORY_KEYWORDS = {
    "country",
    "city",
    "state",
    "region",
    "category",
    "type",
    "status",
    "segment",
    "gender",
    "department",
}


TEXT_KEYWORDS = {
    "description",
    "name",
    "comment",
    "note",
    "address",
}


# ==================================================
# COLUMN NAME NORMALIZATION
# ==================================================

def normalize_column_name(
    column_name: str,
) -> str:
    """
    Normalize a column name so that
    keyword detection is easier.
    """

    return (
        str(column_name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


# ==================================================
# KEYWORD MATCHING
# ==================================================

def contains_keyword(
    column_name: str,
    keywords: set[str],
) -> bool:
    """
    Check whether a column name contains
    one of the supplied keywords.
    """

    normalized = normalize_column_name(
        column_name
    )

    return any(
        keyword in normalized
        for keyword in keywords
    )


# ==================================================
# COLUMN ROLE DETECTION
# ==================================================

def detect_column_role(
    series: pd.Series,
) -> str:
    """
    Detect the analytical role of a column.

    Possible roles:

    - identifier
    - datetime
    - measure
    - categorical
    - text
    - boolean
    """

    column_name = str(
        series.name
    )

    # ----------------------------------------------
    # Datetime
    # ----------------------------------------------

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return "datetime"

    if contains_keyword(
        column_name,
        DATE_KEYWORDS,
    ):
        return "datetime"

    # ----------------------------------------------
    # Boolean
    # ----------------------------------------------

    if pd.api.types.is_bool_dtype(
        series
    ):
        return "boolean"

    # ----------------------------------------------
    # Identifier
    # ----------------------------------------------

    if contains_keyword(
        column_name,
        IDENTIFIER_KEYWORDS,
    ):
        return "identifier"

    # ----------------------------------------------
    # Numeric measure
    # ----------------------------------------------

    if pd.api.types.is_numeric_dtype(
        series
    ):
        return "measure"

    # ----------------------------------------------
    # Text
    # ----------------------------------------------

    if contains_keyword(
        column_name,
        TEXT_KEYWORDS,
    ):
        return "text"

    # ----------------------------------------------
    # Category
    # ----------------------------------------------

    if contains_keyword(
        column_name,
        CATEGORY_KEYWORDS,
    ):
        return "categorical"

    # ----------------------------------------------
    # Default
    # ----------------------------------------------

    return "categorical"


# ==================================================
# SINGLE COLUMN ANALYSIS
# ==================================================

def analyze_column(
    series: pd.Series,
) -> dict[str, Any]:
    """
    Analyze one column and return its
    analytical metadata.
    """

    role = detect_column_role(
        series
    )

    return {
        "name": str(
            series.name
        ),

        "pandas_dtype": str(
            series.dtype
        ),

        "analytical_role": role,

        "row_count": int(
            len(series)
        ),

        "null_count": int(
            series.isna().sum()
        ),

        "null_percentage": round(
            float(
                series.isna().mean()
                * 100
            ),
            2,
        ),

        "unique_count": int(
            series.nunique(
                dropna=True
            )
        ),
    }


# ==================================================
# TABLE COLUMN ANALYSIS
# ==================================================

def analyze_table_columns(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Analyze every column in one table.
    """

    results = []

    for column in dataframe.columns:

        result = analyze_column(
            dataframe[column]
        )

        results.append(
            result
        )

    return results


# ==================================================
# DATASET COLUMN ANALYSIS
# ==================================================

def analyze_dataset_columns(
    tables: dict[str, pd.DataFrame],
) -> dict[str, list[dict[str, Any]]]:
    """
    Analyze columns across multiple tables.
    """

    results = {}

    for table_name, dataframe in (
        tables.items()
    ):

        results[table_name] = (
            analyze_table_columns(
                dataframe
            )
        )

    return results