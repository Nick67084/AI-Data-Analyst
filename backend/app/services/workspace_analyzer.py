from pathlib import Path
from typing import Any

import pandas as pd

from app.services.column_detector import (
    analyze_dataset_columns,
)
from app.services.data_cleaning import (
    clean_dataset,
)
from app.services.data_quality import (
    profile_data_quality,
)
from app.services.dataset_service import (
    load_dataset,
    profile_table,
)
from app.services.relationship_detector import (
    detect_relationships,
    detect_relationships_for_single_table,
)
from app.services.workspace_manager import (
    list_workspace_files,
    load_workspace_tables,
)
from app.services.workspace_summary import (
    build_llm_context,
    build_workspace_summary,
)


def analyze_workspace_tables(
    tables: dict[str, pd.DataFrame],
    filename: str = "workspace",
    file_type: str = "workspace",
    files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Analyze multiple tables as one dataset workspace.

    Tables can come from:

    - CSV files
    - Excel worksheets
    - Multiple uploaded files

    The function performs:

    1. Table profiling
    2. Column detection
    3. Data quality analysis
    4. Cleaning analysis
    5. Identifier analysis
    6. Relationship detection
    7. AI-ready workspace summarization
    """

    if not isinstance(tables, dict):
        raise TypeError(
            "tables must be a dictionary "
            "of table names and pandas DataFrames."
        )

    if not tables:
        raise ValueError(
            "The workspace does not contain any tables."
        )

    for table_name, dataframe in tables.items():
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Table '{table_name}' must be "
                "a pandas DataFrame."
            )

    # --------------------------------------------------
    # 1. TABLE PROFILING
    # --------------------------------------------------

    table_profiles = []

    for table_name, dataframe in tables.items():
        table_profiles.append(
            profile_table(
                table_name,
                dataframe,
            )
        )

    # --------------------------------------------------
    # 2. COLUMN ANALYSIS
    # --------------------------------------------------

    column_analysis = analyze_dataset_columns(
        tables
    )

    # --------------------------------------------------
    # 3. DATA QUALITY
    # --------------------------------------------------

    quality_reports = {}

    for table_name, dataframe in tables.items():
        quality_reports[table_name] = (
            profile_data_quality(
                dataframe
            )
        )

    # --------------------------------------------------
    # 4. CLEANING ANALYSIS
    # --------------------------------------------------

    cleaning_reports = {}

    for table_name, dataframe in tables.items():

        cleaned_dataframe, cleaning_report = (
            clean_dataset(
                dataframe=dataframe,
                remove_duplicates=False,
            )
        )

        cleaning_reports[table_name] = {
            "cleaning_report": cleaning_report,
            "original_rows": int(
                len(dataframe)
            ),
            "original_columns": int(
                len(dataframe.columns)
            ),
            "cleaned_rows": int(
                len(cleaned_dataframe)
            ),
            "cleaned_columns": int(
                len(cleaned_dataframe.columns)
            ),
        }

    # --------------------------------------------------
    # 5. IDENTIFIER ANALYSIS
    # --------------------------------------------------

    identifier_analysis = {}

    for table_name, dataframe in tables.items():
        identifier_analysis[table_name] = (
            detect_relationships_for_single_table(
                dataframe,
                table_name,
            )
        )

    # --------------------------------------------------
    # 6. RELATIONSHIP DETECTION
    # --------------------------------------------------

    relationships = detect_relationships(
        tables,
        minimum_overlap=30.0,
    )

    # --------------------------------------------------
    # 7. BASIC WORKSPACE INFORMATION
    # --------------------------------------------------

    total_rows = sum(
        len(dataframe)
        for dataframe in tables.values()
    )

    total_columns = sum(
        len(dataframe.columns)
        for dataframe in tables.values()
    )

    # --------------------------------------------------
    # 8. COMPLETE ANALYSIS RESULT
    # --------------------------------------------------

    analysis_result = {
        "filename": filename,
        "file_type": file_type,
        "files": files or [],
        "table_count": len(tables),
        "total_rows": int(total_rows),
        "total_columns": int(total_columns),
        "tables": table_profiles,
        "column_analysis": column_analysis,
        "data_quality": quality_reports,
        "cleaning": cleaning_reports,
        "identifier_analysis": identifier_analysis,
        "relationships": relationships,
    }

    # --------------------------------------------------
    # 9. BUILD COMPACT AI SUMMARY
    # --------------------------------------------------

    ai_summary = build_workspace_summary(
        analysis_result
    )

    # --------------------------------------------------
    # 10. BUILD LLM CONTEXT
    # --------------------------------------------------

    llm_context = build_llm_context(
        ai_summary
    )

    # --------------------------------------------------
    # 11. ADD AI INFORMATION TO RESULT
    # --------------------------------------------------

    analysis_result["ai_summary"] = (
        ai_summary
    )

    analysis_result["llm_context"] = (
        llm_context
    )

    return analysis_result


def analyze_workspace_file(
    file_path: str | Path,
) -> dict[str, Any]:
    """
    Analyze one uploaded CSV or Excel file.

    CSV:
        One table is created.

    Excel:
        Each worksheet becomes a table.

    The tables are passed to the common
    multi-table workspace analyzer.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: "
            f"{file_path}"
        )

    tables = load_dataset(
        file_path
    )

    if not tables:
        raise ValueError(
            "No tables were found in the uploaded dataset."
        )

    return analyze_workspace_tables(
        tables=tables,
        filename=file_path.name,
        file_type=file_path.suffix.lower(),
        files=[
            {
                "filename": file_path.name,
                "extension": file_path.suffix.lower(),
                "size_bytes": file_path.stat().st_size,
            }
        ],
    )


def analyze_persistent_workspace(
    workspace_id: str,
) -> dict[str, Any]:
    """
    Analyze every supported dataset stored
    inside a persistent workspace.

    Example:

        workspace_001/
            sales.csv
            customers.csv
            products.xlsx

    All tables from all files are loaded
    and analyzed together.
    """

    tables = load_workspace_tables(
        workspace_id
    )

    if not tables:
        raise ValueError(
            "The workspace does not contain "
            "any analyzable tables."
        )

    files = list_workspace_files(
        workspace_id
    )

    return analyze_workspace_tables(
        tables=tables,
        filename=workspace_id,
        file_type="workspace",
        files=files,
    )