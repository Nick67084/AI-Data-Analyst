from pathlib import Path
from typing import Any

import pandas as pd

from app.services.cleaning_rules import (
    generate_cleaning_plan,
)


BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


DATE_KEYWORDS = {
    "date",
    "time",
    "datetime",
    "timestamp",
}


def is_date_like_column(
    column_name: str,
) -> bool:
    """
    Determine whether a column name strongly
    suggests date/time information.
    """

    normalized = (
        str(column_name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    return any(
        keyword in normalized
        for keyword in DATE_KEYWORDS
    )


def convert_date_columns(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
]:
    """
    Convert confident date/time columns
    from text to datetime.
    """

    df = dataframe.copy()

    actions = []

    for column in df.columns:

        if not is_date_like_column(column):
            continue

        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(
            series
        ):
            continue

        non_null = series.dropna()

        if len(non_null) == 0:
            continue

        converted = pd.to_datetime(
            non_null,
            errors="coerce",
            format="mixed",
        )

        conversion_ratio = float(
            converted.notna().mean()
        )

        if conversion_ratio >= 0.95:

            original_null_count = int(
                series.isna().sum()
            )

            converted_full = pd.to_datetime(
                series,
                errors="coerce",
                format="mixed",
            )

            new_null_count = int(
                converted_full.isna().sum()
            )

            df[column] = converted_full

            actions.append(
                {
                    "action": "convert_datetime",
                    "column": str(column),
                    "conversion_ratio": round(
                        conversion_ratio * 100,
                        2,
                    ),
                    "new_null_values": max(
                        0,
                        new_null_count
                        - original_null_count,
                    ),
                    "message": (
                        f"Converted '{column}' "
                        "to datetime."
                    ),
                }
            )

    return df, actions


def replace_infinite_values(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
]:
    """
    Replace positive and negative infinity
    with missing values.
    """

    df = dataframe.copy()

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    total_replaced = 0

    for column in numeric_columns:

        infinite_mask = df[column].isin(
            [
                float("inf"),
                float("-inf"),
            ]
        )

        count = int(
            infinite_mask.sum()
        )

        if count > 0:

            df.loc[
                infinite_mask,
                column,
            ] = pd.NA

            total_replaced += count

    actions = []

    if total_replaced > 0:

        actions.append(
            {
                "action": "replace_infinite_values",
                "count": total_replaced,
                "message": (
                    f"Replaced {total_replaced} "
                    "infinite numeric values with "
                    "missing values."
                ),
            }
        )

    return df, actions


def remove_empty_columns(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
]:
    """
    Remove columns where every value
    is missing.
    """

    df = dataframe.copy()

    empty_columns = [
        str(column)
        for column in df.columns
        if df[column].isna().all()
    ]

    if not empty_columns:
        return df, []

    df = df.drop(
        columns=empty_columns
    )

    return df, [
        {
            "action": "remove_empty_columns",
            "columns": empty_columns,
            "count": len(empty_columns),
            "message": (
                "Removed columns containing "
                "only missing values."
            ),
        }
    ]


def inspect_duplicates(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Inspect exact duplicate rows without
    deleting them.
    """

    duplicate_count = int(
        dataframe.duplicated().sum()
    )

    duplicate_percentage = 0.0

    if len(dataframe) > 0:

        duplicate_percentage = round(
            (
                duplicate_count
                / len(dataframe)
                * 100
            ),
            2,
        )

    return {
        "duplicate_rows": duplicate_count,
        "duplicate_percentage": (
            duplicate_percentage
        ),
    }


def remove_exact_duplicates(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
]:
    """
    Remove exact duplicate rows.
    """

    df = dataframe.copy()

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count == 0:
        return df, []

    df = (
        df.drop_duplicates()
        .reset_index(drop=True)
    )

    return df, [
        {
            "action": "remove_exact_duplicates",
            "count": duplicate_count,
            "message": (
                f"Removed {duplicate_count} "
                "exact duplicate rows."
            ),
        }
    ]


def apply_safe_missing_value_rules(
    dataframe: pd.DataFrame,
    cleaning_plan: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
]:
    """
    Apply only missing-value actions that are
    considered safe without requiring additional
    analytical context.

    Important:
    Customer IDs and dates are NOT invented.

    Numeric and categorical imputation is left
    for a later task-specific cleaning stage.
    """

    df = dataframe.copy()

    actions = []

    for rule in cleaning_plan[
        "column_rules"
    ]:

        column = rule["column"]

        if column not in df.columns:
            continue

        recommended_action = rule[
            "recommended_action"
        ]

        # -----------------------------------------------------
        # Identifier
        # -----------------------------------------------------

        if recommended_action == (
            "preserve_missing"
        ):

            missing_count = int(
                df[column].isna().sum()
            )

            if missing_count > 0:

                actions.append(
                    {
                        "action": (
                            "preserve_missing"
                        ),
                        "column": column,
                        "count": missing_count,
                        "message": (
                            f"Preserved {missing_count} "
                            f"missing values in "
                            f"'{column}'."
                        ),
                    }
                )

        # -----------------------------------------------------
        # Large missing numeric/categorical data
        # -----------------------------------------------------

        elif recommended_action == (
            "preserve_missing_and_warn"
        ):

            missing_count = int(
                df[column].isna().sum()
            )

            if missing_count > 0:

                actions.append(
                    {
                        "action": (
                            "preserve_missing_and_warn"
                        ),
                        "column": column,
                        "count": missing_count,
                        "message": (
                            f"Preserved {missing_count} "
                            f"missing values in "
                            f"'{column}' because "
                            "automatic imputation may "
                            "distort the analysis."
                        ),
                    }
                )

    return df, actions


def clean_dataset(
    dataframe: pd.DataFrame,
    remove_duplicates: bool = False,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Perform conservative automatic cleaning.

    The cleaning process first creates a cleaning
    plan and then applies only safe actions.

    Parameters
    ----------
    dataframe:
        Input dataframe.

    remove_duplicates:
        Whether exact duplicate rows should
        be removed.

        Default is False.

    Returns
    -------
    cleaned_dataframe:
        Processed dataframe.

    cleaning_report:
        Detailed cleaning report.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "clean_dataset() requires "
            "a pandas DataFrame."
        )

    df = dataframe.copy()

    original_rows = len(df)
    original_columns = len(
        df.columns
    )

    actions: list[
        dict[str, Any]
    ] = []

    # ---------------------------------------------------------
    # 1. Generate cleaning plan
    # ---------------------------------------------------------

    cleaning_plan = (
        generate_cleaning_plan(df)
    )

    # ---------------------------------------------------------
    # 2. Convert date columns
    # ---------------------------------------------------------

    df, date_actions = (
        convert_date_columns(df)
    )

    actions.extend(
        date_actions
    )

    # ---------------------------------------------------------
    # 3. Replace infinite values
    # ---------------------------------------------------------

    df, infinite_actions = (
        replace_infinite_values(df)
    )

    actions.extend(
        infinite_actions
    )

    # ---------------------------------------------------------
    # 4. Remove completely empty columns
    # ---------------------------------------------------------

    df, empty_column_actions = (
        remove_empty_columns(df)
    )

    actions.extend(
        empty_column_actions
    )

    # ---------------------------------------------------------
    # 5. Apply safe missing-value rules
    # ---------------------------------------------------------

    df, missing_actions = (
        apply_safe_missing_value_rules(
            df,
            cleaning_plan,
        )
    )

    actions.extend(
        missing_actions
    )

    # ---------------------------------------------------------
    # 6. Inspect duplicates
    # ---------------------------------------------------------

    duplicate_info = (
        inspect_duplicates(df)
    )

    # ---------------------------------------------------------
    # 7. Optionally remove duplicates
    # ---------------------------------------------------------

    if remove_duplicates:

        df, duplicate_actions = (
            remove_exact_duplicates(df)
        )

        actions.extend(
            duplicate_actions
        )

    # ---------------------------------------------------------
    # 8. Final report
    # ---------------------------------------------------------

    final_rows = len(df)
    final_columns = len(
        df.columns
    )

    cleaning_report = {
        "original_rows": int(
            original_rows
        ),
        "original_columns": int(
            original_columns
        ),
        "final_rows": int(
            final_rows
        ),
        "final_columns": int(
            final_columns
        ),
        "rows_removed": int(
            original_rows
            - final_rows
        ),
        "columns_removed": int(
            original_columns
            - final_columns
        ),
        "duplicate_rows_detected": int(
            duplicate_info[
                "duplicate_rows"
            ]
        ),
        "duplicate_percentage": float(
            duplicate_info[
                "duplicate_percentage"
            ]
        ),
        "duplicates_removed": bool(
            remove_duplicates
        ),
        "cleaning_plan": cleaning_plan,
        "actions": actions,
    }

    return (
        df,
        cleaning_report,
    )


def save_processed_dataset(
    dataframe: pd.DataFrame,
    original_filename: str,
) -> Path:
    """
    Save the processed dataset separately
    from the original uploaded file.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_path = Path(
        original_filename
    )

    output_filename = (
        f"{original_path.stem}"
        f"_processed"
        f"{original_path.suffix}"
    )

    output_path = (
        PROCESSED_DATA_DIR
        / output_filename
    )

    if (
        original_path.suffix.lower()
        == ".csv"
    ):

        dataframe.to_csv(
            output_path,
            index=False,
        )

    elif (
        original_path.suffix.lower()
        in {".xlsx", ".xls"}
    ):

        dataframe.to_excel(
            output_path,
            index=False,
        )

    else:

        raise ValueError(
            "Unsupported output file type."
        )

    return output_path