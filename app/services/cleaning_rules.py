from typing import Any

import pandas as pd


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


def normalize_column_name(
    column_name: str,
) -> str:
    """
    Normalize a column name so it can be
    compared with keyword rules.
    """

    return (
        str(column_name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def contains_keyword(
    column_name: str,
    keywords: set[str],
) -> bool:
    """
    Check whether a normalized column name
    contains one of the supplied keywords.
    """

    normalized = normalize_column_name(
        column_name
    )

    return any(
        keyword in normalized
        for keyword in keywords
    )


def determine_column_type(
    series: pd.Series,
) -> str:
    """
    Determine the analytical type of a column.

    The result is intentionally simple because
    these rules are used for cleaning decisions.
    """

    column_name = str(series.name)

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return "datetime"

    if contains_keyword(
        column_name,
        DATE_KEYWORDS,
    ):
        return "datetime"

    if contains_keyword(
        column_name,
        IDENTIFIER_KEYWORDS,
    ):
        return "identifier"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "measure"

    if contains_keyword(
        column_name,
        TEXT_KEYWORDS,
    ):
        return "text"

    if contains_keyword(
        column_name,
        CATEGORY_KEYWORDS,
    ):
        return "categorical"

    return "categorical"


def get_missing_value_rule(
    series: pd.Series,
) -> dict[str, Any]:
    """
    Determine what should happen when a column
    contains missing values.

    This function recommends a policy.
    It does not modify the dataframe.
    """

    column_name = str(series.name)

    missing_count = int(
        series.isna().sum()
    )

    missing_percentage = (
        float(
            series.isna().mean() * 100
        )
        if len(series) > 0
        else 0.0
    )

    column_type = determine_column_type(
        series
    )

    # ---------------------------------------------------------
    # No missing values
    # ---------------------------------------------------------

    if missing_count == 0:
        return {
            "column": column_name,
            "column_type": column_type,
            "missing_count": 0,
            "missing_percentage": 0.0,
            "recommended_action": "none",
            "reason": (
                "The column does not contain "
                "missing values."
            ),
        }

    # ---------------------------------------------------------
    # Identifier
    # ---------------------------------------------------------

    if column_type == "identifier":
        return {
            "column": column_name,
            "column_type": column_type,
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_percentage,
                2,
            ),
            "recommended_action": "preserve_missing",
            "reason": (
                "Missing identifiers should not "
                "be invented or automatically "
                "imputed."
            ),
        }

    # ---------------------------------------------------------
    # Datetime
    # ---------------------------------------------------------

    if column_type == "datetime":
        return {
            "column": column_name,
            "column_type": column_type,
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_percentage,
                2,
            ),
            "recommended_action": "preserve_missing",
            "reason": (
                "Missing dates should not be "
                "blindly replaced because an "
                "artificial date can distort "
                "time-based analysis."
            ),
        }

    # ---------------------------------------------------------
    # Measure
    # ---------------------------------------------------------

    if column_type == "measure":

        if missing_percentage <= 5:
            action = "consider_median_imputation"
            reason = (
                "A small percentage of missing "
                "numeric values may be suitable "
                "for median-based imputation, "
                "depending on the analytical task."
            )

        elif missing_percentage <= 30:
            action = "review_before_imputation"
            reason = (
                "The amount of missing numeric "
                "data is significant. The system "
                "should review the analytical "
                "context before imputing values."
            )

        else:
            action = "preserve_missing_and_warn"
            reason = (
                "A large percentage of numeric "
                "values is missing. Automatic "
                "imputation may distort analysis."
            )

        return {
            "column": column_name,
            "column_type": column_type,
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_percentage,
                2,
            ),
            "recommended_action": action,
            "reason": reason,
        }

    # ---------------------------------------------------------
    # Categorical
    # ---------------------------------------------------------

    if column_type == "categorical":

        if missing_percentage <= 20:
            action = "consider_unknown_category"
            reason = (
                "Missing categorical values may "
                "be represented as 'Unknown' when "
                "appropriate for the analysis."
            )

        else:
            action = "preserve_missing_and_warn"
            reason = (
                "A large percentage of categorical "
                "values is missing. Automatically "
                "creating a category may hide a "
                "data-quality problem."
            )

        return {
            "column": column_name,
            "column_type": column_type,
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_percentage,
                2,
            ),
            "recommended_action": action,
            "reason": reason,
        }

    # ---------------------------------------------------------
    # Text
    # ---------------------------------------------------------

    if column_type == "text":
        return {
            "column": column_name,
            "column_type": column_type,
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_percentage,
                2,
            ),
            "recommended_action": "preserve_missing",
            "reason": (
                "Missing free-text values should "
                "normally remain missing rather "
                "than being automatically invented."
            ),
        }

    # ---------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------

    return {
        "column": column_name,
        "column_type": column_type,
        "missing_count": missing_count,
        "missing_percentage": round(
            missing_percentage,
            2,
        ),
        "recommended_action": "preserve_missing_and_warn",
        "reason": (
            "The system could not determine a "
            "safe automatic imputation strategy."
        ),
    }


def generate_cleaning_rules(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Generate cleaning recommendations for
    every column in a dataframe.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "generate_cleaning_rules() requires "
            "a pandas DataFrame."
        )

    rules = []

    for column in dataframe.columns:
        rule = get_missing_value_rule(
            dataframe[column]
        )
        rules.append(rule)

    return rules


def get_duplicate_rule(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Generate a recommendation for exact
    duplicate rows.
    """

    duplicate_count = int(
        dataframe.duplicated().sum()
    )

    if duplicate_count == 0:
        return {
            "issue": "duplicate_rows",
            "count": 0,
            "recommended_action": "none",
            "reason": (
                "No exact duplicate rows "
                "were detected."
            ),
        }

    percentage = (
        duplicate_count
        / len(dataframe)
        * 100
        if len(dataframe) > 0
        else 0
    )

    return {
        "issue": "duplicate_rows",
        "count": duplicate_count,
        "percentage": round(
            percentage,
            2,
        ),
        "recommended_action": (
            "review_before_removal"
        ),
        "reason": (
            "Exact duplicate rows were detected. "
            "They should be reviewed before "
            "automatic removal because duplicate "
            "transaction records can sometimes "
            "represent legitimate events."
        ),
    }


def generate_cleaning_plan(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Generate a complete conservative cleaning plan.

    The function only analyzes the data and creates
    recommendations. It does not modify the dataframe.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "generate_cleaning_plan() requires "
            "a pandas DataFrame."
        )

    column_rules = (
        generate_cleaning_rules(
            dataframe
        )
    )

    duplicate_rule = get_duplicate_rule(
        dataframe
    )

    return {
        "rows": int(len(dataframe)),
        "columns": int(
            len(dataframe.columns)
        ),
        "column_rules": column_rules,
        "duplicate_rule": duplicate_rule,
    }