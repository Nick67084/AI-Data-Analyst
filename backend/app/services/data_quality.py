from typing import Any

import pandas as pd


def check_missing_values(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Check missing values in every column.
    """

    issues = []

    for column in dataframe.columns:
        null_count = int(
            dataframe[column].isna().sum()
        )

        null_percentage = round(
            float(
                dataframe[column].isna().mean() * 100
            ),
            2,
        )

        if null_count > 0:
            issues.append(
                {
                    "column": str(column),
                    "issue_type": "missing_values",
                    "count": null_count,
                    "percentage": null_percentage,
                    "severity": (
                        "critical"
                        if null_percentage >= 50
                        else "warning"
                    ),
                    "message": (
                        f"{null_percentage}% of values "
                        f"are missing."
                    ),
                }
            )

    return issues


def check_duplicates(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Check duplicate rows.
    """

    duplicate_count = int(
        dataframe.duplicated().sum()
    )

    duplicate_percentage = round(
        float(
            duplicate_count / len(dataframe) * 100
        ),
        2,
    ) if len(dataframe) > 0 else 0.0

    if duplicate_count == 0:
        severity = "ok"
    elif duplicate_percentage >= 10:
        severity = "critical"
    else:
        severity = "warning"

    return {
        "duplicate_rows": duplicate_count,
        "duplicate_percentage": duplicate_percentage,
        "severity": severity,
    }


def check_empty_columns(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Find columns where every value is missing.
    """

    issues = []

    for column in dataframe.columns:
        if dataframe[column].isna().all():
            issues.append(
                {
                    "column": str(column),
                    "issue_type": "empty_column",
                    "severity": "critical",
                    "message": (
                        "The column contains no usable values."
                    ),
                }
            )

    return issues


def check_constant_columns(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Find columns containing only one unique value.
    """

    issues = []

    for column in dataframe.columns:

        unique_count = dataframe[column].nunique(
            dropna=True
        )

        if unique_count <= 1:
            issues.append(
                {
                    "column": str(column),
                    "issue_type": "constant_column",
                    "severity": "warning",
                    "message": (
                        "The column contains only "
                        "one unique value."
                    ),
                }
            )

    return issues


def check_data_types(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Detect potentially problematic data types.

    The checker avoids treating identifier columns
    as numeric measures.
    """

    issues = []

    date_keywords = {
        "date",
        "time",
        "datetime",
        "timestamp",
        "year",
        "month",
        "day",
    }

    identifier_keywords = {
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

    for column in dataframe.columns:

        series = dataframe[column]

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        non_null = series.dropna()

        if len(non_null) == 0:
            continue

        column_name = str(column).strip().lower()

        # --------------------------------------------------
        # Identifier columns
        # --------------------------------------------------

        looks_like_identifier = any(
            keyword in column_name.replace(
                "_",
                "",
            ).replace(
                " ",
                "",
            )
            for keyword in identifier_keywords
        )

        if looks_like_identifier:
            continue

        # --------------------------------------------------
        # Date/time columns
        # --------------------------------------------------

        looks_like_date = any(
            keyword in column_name
            for keyword in date_keywords
        )

        if looks_like_date:

            datetime_conversion = pd.to_datetime(
                non_null,
                errors="coerce",
                format="mixed",
            )

            datetime_ratio = float(
                datetime_conversion.notna().mean()
            )

            if datetime_ratio >= 0.95:
                issues.append(
                    {
                        "column": str(column),
                        "issue_type": (
                            "datetime_stored_as_text"
                        ),
                        "severity": "warning",
                        "message": (
                            "Most values appear to be "
                            "dates/times but are stored as text."
                        ),
                    }
                )

                continue

        # --------------------------------------------------
        # Numeric-looking text
        # --------------------------------------------------

        numeric_conversion = pd.to_numeric(
            non_null,
            errors="coerce",
        )

        numeric_ratio = float(
            numeric_conversion.notna().mean()
        )

        if numeric_ratio >= 0.95:
            issues.append(
                {
                    "column": str(column),
                    "issue_type": (
                        "numeric_stored_as_text"
                    ),
                    "severity": "warning",
                    "message": (
                        "Most values appear to be "
                        "numeric but are stored as text."
                    ),
                }
            )

    return issues

def check_numeric_values(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Detect invalid infinite numeric values.

    Missing values are not treated as infinite values.
    """

    issues = []

    numeric_columns = dataframe.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        series = dataframe[column]

        infinite_mask = series.isin(
            [float("inf"), float("-inf")]
        )

        infinite_count = int(
            infinite_mask.sum()
        )

        if infinite_count > 0:
            issues.append(
                {
                    "column": str(column),
                    "issue_type": "infinite_values",
                    "count": infinite_count,
                    "severity": "warning",
                    "message": (
                        "The column contains "
                        "infinite numeric values."
                    ),
                }
            )

    return issues

def calculate_quality_status(
    issues: list[dict[str, Any]],
    duplicate_info: dict[str, Any],
) -> str:
    """
    Determine overall dataset quality status.
    """

    if any(
        issue.get("severity") == "critical"
        for issue in issues
    ):
        return "critical"

    if duplicate_info["severity"] == "critical":
        return "critical"

    if issues:
        return "warning"

    if duplicate_info["severity"] == "warning":
        return "warning"

    return "ready"


def profile_data_quality(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Generate a complete data quality report.
    """

    missing_issues = check_missing_values(
        dataframe
    )

    duplicate_info = check_duplicates(
        dataframe
    )

    empty_column_issues = check_empty_columns(
        dataframe
    )

    constant_column_issues = check_constant_columns(
        dataframe
    )

    data_type_issues = check_data_types(
        dataframe
    )

    numeric_issues = check_numeric_values(
        dataframe
    )

    all_issues = (
        missing_issues
        + empty_column_issues
        + constant_column_issues
        + data_type_issues
        + numeric_issues
    )

    quality_status = calculate_quality_status(
        all_issues,
        duplicate_info,
    )

    return {
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "duplicate_rows": duplicate_info[
            "duplicate_rows"
        ],
        "duplicate_percentage": duplicate_info[
            "duplicate_percentage"
        ],
        "status": quality_status,
        "issues": all_issues,
    }