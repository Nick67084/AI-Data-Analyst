from pathlib import Path
from typing import Any

import pandas as pd

from app.services.column_detector import analyze_table_columns


BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
WORKSPACE_DIR = DATA_DIR / "workspace"

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
}


def get_file_extension(file_path: str | Path) -> str:
    """
    Return the lowercase file extension.
    """
    return Path(file_path).suffix.lower()


def is_supported_file(file_path: str | Path) -> bool:
    """
    Check whether the uploaded file type is supported.
    """
    return get_file_extension(file_path) in SUPPORTED_EXTENSIONS


def save_uploaded_file(
    file_content: bytes,
    original_filename: str,
) -> Path:
    """
    Save uploaded file bytes to the dataset workspace.

    The original uploaded file is preserved.
    """

    if not original_filename:
        raise ValueError("Uploaded file must have a filename.")

    if not is_supported_file(original_filename):
        raise ValueError(
            f"Unsupported file type: {get_file_extension(original_filename)}"
        )

    WORKSPACE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_filename = Path(original_filename).name
    destination = WORKSPACE_DIR / safe_filename

    destination.write_bytes(file_content)

    return destination


def load_csv(file_path: Path) -> dict[str, pd.DataFrame]:
    """
    Load a CSV file as a single table.
    """

    dataframe = pd.read_csv(
        file_path,
        low_memory=False,
    )

    table_name = file_path.stem

    return {
        table_name: dataframe,
    }


def load_excel(file_path: Path) -> dict[str, pd.DataFrame]:
    """
    Load every worksheet in an Excel workbook.

    Each worksheet becomes a separate table.
    """

    tables: dict[str, pd.DataFrame] = {}

    excel_file = pd.ExcelFile(
        file_path,
        engine="openpyxl",
    )

    for sheet_name in excel_file.sheet_names:

        print(
            f"Reading Excel sheet: {sheet_name}"
        )

        dataframe = pd.read_excel(
            excel_file,
            sheet_name=sheet_name,
        )

        tables[sheet_name] = dataframe

    return tables


def load_dataset(file_path: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load a supported dataset.

    CSV:
        one table

    Excel:
        one table per worksheet
    """

    file_path = Path(file_path)

    extension = get_file_extension(file_path)

    if extension == ".csv":
        return load_csv(file_path)

    if extension in {".xlsx", ".xls"}:
        return load_excel(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


def get_column_type(series: pd.Series) -> str:
    """
    Detect a simple data type for API responses.
    """

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    return "categorical"


def profile_table(
    table_name: str,
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Create a profile for one table.
    """

    print(
        f"Profiling table: {table_name}"
    )

    column_analysis = analyze_table_columns(
        dataframe
    )

    column_details = []

    for column_info in column_analysis:

        column_name = column_info["name"]

        series = dataframe[column_name]

        column_details.append(
            {
                "name": column_name,
                "dtype": str(series.dtype),
                "detected_type": get_column_type(series),
                "analytical_role": column_info[
                    "analytical_role"
                ],
                "null_count": int(
                    series.isna().sum()
                ),
                "null_percentage": round(
                    float(
                        series.isna().mean() * 100
                    ),
                    2,
                ),
                "unique_count": int(
                    series.nunique(
                        dropna=True
                    )
                ),
            }
        )

    return {
        "table_name": table_name,
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),
        "column_details": column_details,
    }


def profile_dataset(
    file_path: str | Path,
) -> dict[str, Any]:
    """
    Load and profile an uploaded dataset.
    """

    file_path = Path(file_path)

    tables = load_dataset(file_path)

    extension = get_file_extension(file_path)

    if extension == ".csv":
        file_type = "csv"
    else:
        file_type = "excel"

    table_profiles = []

    for table_name, dataframe in tables.items():

        table_profile = profile_table(
            table_name,
            dataframe,
        )

        table_profiles.append(
            table_profile
        )

    return {
        "filename": file_path.name,
        "file_type": file_type,
        "tables": table_profiles,
    }