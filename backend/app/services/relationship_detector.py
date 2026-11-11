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


def normalize_column_name(column_name: str) -> str:
    return (
        str(column_name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def is_date_column(column_name: str) -> bool:
    normalized = normalize_column_name(column_name)

    return any(
        keyword in normalized
        for keyword in DATE_KEYWORDS
    )


def is_identifier_column(column_name: str) -> bool:
    if is_date_column(column_name):
        return False

    normalized = normalize_column_name(column_name)

    return any(
        keyword in normalized
        for keyword in IDENTIFIER_KEYWORDS
    )


def get_identifier_columns(
    dataframe: pd.DataFrame,
) -> list[str]:

    identifier_columns = []

    for column in dataframe.columns:

        if is_identifier_column(column):
            identifier_columns.append(str(column))

    return identifier_columns


def normalize_identifier_values(
    series: pd.Series,
) -> pd.Series:

    normalized = series.copy()

    normalized = normalized.where(
        normalized.notna(),
        pd.NA,
    )

    normalized = normalized.astype("string")

    normalized = (
        normalized
        .str.strip()
        .str.lower()
    )

    return normalized


def calculate_value_overlap(
    left_series: pd.Series,
    right_series: pd.Series,
) -> dict[str, Any]:

    left_values = (
        normalize_identifier_values(left_series)
        .dropna()
    )

    right_values = (
        normalize_identifier_values(right_series)
        .dropna()
    )

    left_unique = set(left_values.unique())
    right_unique = set(right_values.unique())

    if not left_unique or not right_unique:

        return {
            "left_unique_values": len(left_unique),
            "right_unique_values": len(right_unique),
            "matching_values": 0,
            "left_match_percentage": 0.0,
            "right_match_percentage": 0.0,
            "overlap_percentage": 0.0,
        }

    matching_values = (
        left_unique & right_unique
    )

    left_match_percentage = (
        len(matching_values)
        / len(left_unique)
        * 100
    )

    right_match_percentage = (
        len(matching_values)
        / len(right_unique)
        * 100
    )

    smaller_unique_count = min(
        len(left_unique),
        len(right_unique),
    )

    overlap_percentage = (
        len(matching_values)
        / smaller_unique_count
        * 100
    )

    return {
        "left_unique_values": len(left_unique),
        "right_unique_values": len(right_unique),
        "matching_values": len(matching_values),
        "left_match_percentage": round(
            left_match_percentage,
            2,
        ),
        "right_match_percentage": round(
            right_match_percentage,
            2,
        ),
        "overlap_percentage": round(
            overlap_percentage,
            2,
        ),
    }


def determine_relationship_type(
    overlap_info: dict[str, Any],
) -> str:

    left_unique = overlap_info[
        "left_unique_values"
    ]

    right_unique = overlap_info[
        "right_unique_values"
    ]

    matching_values = overlap_info[
        "matching_values"
    ]

    left_match_percentage = overlap_info[
        "left_match_percentage"
    ]

    right_match_percentage = overlap_info[
        "right_match_percentage"
    ]

    if matching_values == 0:
        return "no_relationship"

    # Both sides contain exactly the same
    # unique values.
    if (
        left_unique == right_unique
        and left_unique == matching_values
    ):
        return "one_to_one"

    # Left side is fully covered by the right side.
    #
    # Example:
    # orders.Customer ID
    #        ↓
    # customers.Customer ID
    #
    # Many orders can refer to one customer.
    if (
        left_match_percentage == 100.0
        and right_match_percentage < 100.0
    ):
        return "many_to_one"

    # Right side is fully covered by the left side.
    #
    # Example:
    # customers.Customer ID
    #        ↓
    # orders.Customer ID
    #
    # One customer can appear in many orders.
    if (
        right_match_percentage == 100.0
        and left_match_percentage < 100.0
    ):
        return "one_to_many"

    return "possible_many_to_many"


def determine_confidence(
    overlap_info: dict[str, Any],
    relationship_type: str,
) -> str:

    overlap = overlap_info[
        "overlap_percentage"
    ]

    left_match = overlap_info[
        "left_match_percentage"
    ]

    right_match = overlap_info[
        "right_match_percentage"
    ]

    if relationship_type == "no_relationship":
        return "none"

    if (
        overlap >= 95
        and max(
            left_match,
            right_match,
        ) >= 95
    ):
        return "high"

    if (
        overlap >= 70
        and max(
            left_match,
            right_match,
        ) >= 70
    ):
        return "medium"

    return "low"


def compare_columns(
    left_table_name: str,
    left_column: str,
    left_series: pd.Series,
    right_table_name: str,
    right_column: str,
    right_series: pd.Series,
) -> dict[str, Any]:

    overlap_info = calculate_value_overlap(
        left_series,
        right_series,
    )

    relationship_type = (
        determine_relationship_type(
            overlap_info
        )
    )

    confidence = determine_confidence(
        overlap_info,
        relationship_type,
    )

    return {
        "left_table": left_table_name,
        "left_column": left_column,
        "right_table": right_table_name,
        "right_column": right_column,
        "relationship_type": relationship_type,
        "confidence": confidence,
        "left_unique_values": overlap_info[
            "left_unique_values"
        ],
        "right_unique_values": overlap_info[
            "right_unique_values"
        ],
        "matching_values": overlap_info[
            "matching_values"
        ],
        "left_match_percentage": overlap_info[
            "left_match_percentage"
        ],
        "right_match_percentage": overlap_info[
            "right_match_percentage"
        ],
        "overlap_percentage": overlap_info[
            "overlap_percentage"
        ],
    }


def detect_relationships(
    tables: dict[str, pd.DataFrame],
    minimum_overlap: float = 30.0,
) -> list[dict[str, Any]]:

    if not isinstance(tables, dict):
        raise TypeError(
            "tables must be a dictionary "
            "of table names and DataFrames."
        )

    table_names = list(tables.keys())

    relationships = []

    for left_index in range(
        len(table_names)
    ):

        left_table_name = table_names[
            left_index
        ]

        left_table = tables[
            left_table_name
        ]

        left_identifier_columns = (
            get_identifier_columns(
                left_table
            )
        )

        for right_index in range(
            left_index + 1,
            len(table_names),
        ):

            right_table_name = table_names[
                right_index
            ]

            right_table = tables[
                right_table_name
            ]

            right_identifier_columns = (
                get_identifier_columns(
                    right_table
                )
            )

            for left_column in (
                left_identifier_columns
            ):

                for right_column in (
                    right_identifier_columns
                ):

                    comparison = compare_columns(
                        left_table_name,
                        left_column,
                        left_table[left_column],
                        right_table_name,
                        right_column,
                        right_table[right_column],
                    )

                    if (
                        comparison[
                            "overlap_percentage"
                        ]
                        >= minimum_overlap
                    ):

                        relationships.append(
                            comparison
                        )

    return relationships


def detect_relationships_for_single_table(
    dataframe: pd.DataFrame,
    table_name: str = "dataset",
) -> list[dict[str, Any]]:

    identifier_columns = (
        get_identifier_columns(
            dataframe
        )
    )

    results = []

    for column in identifier_columns:

        series = dataframe[column]

        non_null = series.dropna()

        unique_count = int(
            non_null.nunique()
        )

        row_count = int(
            len(non_null)
        )

        uniqueness_ratio = (
            unique_count
            / row_count
            * 100
            if row_count > 0
            else 0.0
        )

        results.append(
            {
                "table": table_name,
                "column": column,
                "non_null_values": row_count,
                "unique_values": unique_count,
                "uniqueness_percentage": round(
                    uniqueness_ratio,
                    2,
                ),
            }
        )

    return results