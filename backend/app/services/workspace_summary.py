from typing import Any

import pandas as pd


def build_workspace_summary(
    workspace_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert a large workspace analysis result into
    a compact AI-ready summary.

    The summary contains metadata and high-level
    analytical information instead of raw rows.
    """

    if not isinstance(workspace_analysis, dict):
        raise TypeError(
            "workspace_analysis must be a dictionary."
        )

    tables = workspace_analysis.get(
        "tables",
        [],
    )

    column_analysis = workspace_analysis.get(
        "column_analysis",
        {},
    )

    data_quality = workspace_analysis.get(
        "data_quality",
        {},
    )

    relationships = workspace_analysis.get(
        "relationships",
        [],
    )

    files = workspace_analysis.get(
        "files",
        [],
    )

    measures = []
    dimensions = []
    identifiers = []
    dates = []

    # --------------------------------------------------
    # COLUMN ROLE SUMMARY
    # --------------------------------------------------

    for table_name, columns in column_analysis.items():

        for column in columns:

            column_name = column.get(
                "name"
            )

            role = column.get(
                "analytical_role"
            )

            if role == "measure":
                measures.append(
                    {
                        "table": table_name,
                        "column": column_name,
                    }
                )

            elif role == "categorical":
                dimensions.append(
                    {
                        "table": table_name,
                        "column": column_name,
                    }
                )

            elif role == "identifier":
                identifiers.append(
                    {
                        "table": table_name,
                        "column": column_name,
                    }
                )

            elif role == "datetime":
                dates.append(
                    {
                        "table": table_name,
                        "column": column_name,
                    }
                )

    # --------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------

    def remove_duplicates(
        items: list[dict[str, str]],
    ) -> list[dict[str, str]]:

        seen = set()
        result = []

        for item in items:

            key = (
                item.get("table"),
                item.get("column"),
            )

            if key not in seen:
                seen.add(key)
                result.append(item)

        return result

    measures = remove_duplicates(
        measures
    )

    dimensions = remove_duplicates(
        dimensions
    )

    identifiers = remove_duplicates(
        identifiers
    )

    dates = remove_duplicates(
        dates
    )

    # --------------------------------------------------
    # DATA QUALITY SUMMARY
    # --------------------------------------------------

    quality_summary = []

    for table_name, report in data_quality.items():

        quality_summary.append(
            {
                "table": table_name,
                "status": report.get(
                    "status"
                ),
                "rows": report.get(
                    "rows"
                ),
                "columns": report.get(
                    "columns"
                ),
                "duplicate_rows": report.get(
                    "duplicate_rows"
                ),
                "duplicate_percentage": report.get(
                    "duplicate_percentage"
                ),
                "issues": report.get(
                    "issues",
                    [],
                ),
            }
        )

    # --------------------------------------------------
    # RELATIONSHIP SUMMARY
    # --------------------------------------------------

    relationship_summary = []

    for relationship in relationships:

        relationship_summary.append(
            {
                "left_table": relationship.get(
                    "left_table"
                ),
                "left_column": relationship.get(
                    "left_column"
                ),
                "right_table": relationship.get(
                    "right_table"
                ),
                "right_column": relationship.get(
                    "right_column"
                ),
                "relationship_type": relationship.get(
                    "relationship_type"
                ),
                "confidence": relationship.get(
                    "confidence"
                ),
                "overlap_percentage": relationship.get(
                    "overlap_percentage"
                ),
            }
        )

    # --------------------------------------------------
    # POSSIBLE ANALYTICAL TASKS
    # --------------------------------------------------

    possible_tasks = []

    if measures:
        possible_tasks.append(
            "sales analysis"
        )

    if dimensions:
        possible_tasks.append(
            "categorical analysis"
        )

    if identifiers:
        possible_tasks.append(
            "customer or transaction analysis"
        )

    if dates:
        possible_tasks.append(
            "time-series analysis"
        )

    if relationships:
        possible_tasks.append(
            "multi-table analysis"
        )

    # Customer-related analysis
    identifier_names = {
        item["column"].lower()
        for item in identifiers
    }

    if any(
        "customer" in name
        for name in identifier_names
    ):
        possible_tasks.append(
            "customer analysis"
        )

    # Product-related analysis
    identifier_and_dimension_names = {
        item["column"].lower()
        for item in (
            identifiers
            + dimensions
        )
    }

    if any(
        "product" in name
        or "stock" in name
        for name in identifier_and_dimension_names
    ):
        possible_tasks.append(
            "product analysis"
        )

    possible_tasks = list(
        dict.fromkeys(
            possible_tasks
        )
    )

    # --------------------------------------------------
    # TABLE SUMMARY
    # --------------------------------------------------

    table_summary = []

    for table in tables:

        table_summary.append(
            {
                "table_name": table.get(
                    "table_name"
                ),
                "rows": table.get(
                    "rows"
                ),
                "columns": table.get(
                    "columns"
                ),
                "duplicate_rows": table.get(
                    "duplicate_rows"
                ),
            }
        )

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    return {
        "workspace": {
            "filename": workspace_analysis.get(
                "filename"
            ),
            "file_type": workspace_analysis.get(
                "file_type"
            ),
            "table_count": workspace_analysis.get(
                "table_count",
                0,
            ),
            "total_rows": workspace_analysis.get(
                "total_rows",
                0,
            ),
            "total_columns": workspace_analysis.get(
                "total_columns",
                0,
            ),
        },
        "files": files,
        "tables": table_summary,
        "measures": measures,
        "dimensions": dimensions,
        "identifiers": identifiers,
        "dates": dates,
        "data_quality": quality_summary,
        "relationships": relationship_summary,
        "possible_analytical_tasks": possible_tasks,
    }


def build_llm_context(
    workspace_summary: dict[str, Any],
) -> str:
    """
    Convert the structured workspace summary
    into compact text suitable for an LLM prompt.
    """

    if not isinstance(
        workspace_summary,
        dict,
    ):
        raise TypeError(
            "workspace_summary must be a dictionary."
        )

    workspace = workspace_summary.get(
        "workspace",
        {}
    )

    lines = []

    lines.append(
        "WORKSPACE SUMMARY"
    )
    lines.append("")

    # --------------------------------------------------
    # WORKSPACE
    # --------------------------------------------------

    lines.append(
        "Workspace:"
    )

    lines.append(
        f"- Type: {workspace.get('file_type')}"
    )

    lines.append(
        f"- Tables: {workspace.get('table_count', 0)}"
    )

    lines.append(
        f"- Total rows: {workspace.get('total_rows', 0)}"
    )

    lines.append(
        f"- Total columns: {workspace.get('total_columns', 0)}"
    )

    lines.append("")

    # --------------------------------------------------
    # FILES
    # --------------------------------------------------

    files = workspace_summary.get(
        "files",
        []
    )

    if files:

        lines.append(
            "Files:"
        )

        for file_info in files:

            if isinstance(
                file_info,
                dict,
            ):
                lines.append(
                    f"- {file_info.get('filename')}"
                )

            else:
                lines.append(
                    f"- {file_info}"
                )

        lines.append("")

    # --------------------------------------------------
    # TABLES
    # --------------------------------------------------

    lines.append(
        "Tables:"
    )

    for table in workspace_summary.get(
        "tables",
        [],
    ):

        lines.append(
            f"- {table.get('table_name')}: "
            f"{table.get('rows')} rows, "
            f"{table.get('columns')} columns"
        )

    lines.append("")

    # --------------------------------------------------
    # MEASURES
    # --------------------------------------------------

    measures = workspace_summary.get(
        "measures",
        []
    )

    lines.append(
        "Measures:"
    )

    if measures:

        for item in measures:

            lines.append(
                f"- {item.get('table')}.{item.get('column')}"
            )

    else:
        lines.append(
            "- None detected"
        )

    lines.append("")

    # --------------------------------------------------
    # DIMENSIONS
    # --------------------------------------------------

    dimensions = workspace_summary.get(
        "dimensions",
        []
    )

    lines.append(
        "Dimensions:"
    )

    if dimensions:

        for item in dimensions:

            lines.append(
                f"- {item.get('table')}.{item.get('column')}"
            )

    else:
        lines.append(
            "- None detected"
        )

    lines.append("")

    # --------------------------------------------------
    # IDENTIFIERS
    # --------------------------------------------------

    identifiers = workspace_summary.get(
        "identifiers",
        []
    )

    lines.append(
        "Identifiers:"
    )

    if identifiers:

        for item in identifiers:

            lines.append(
                f"- {item.get('table')}.{item.get('column')}"
            )

    else:
        lines.append(
            "- None detected"
        )

    lines.append("")

    # --------------------------------------------------
    # DATES
    # --------------------------------------------------

    dates = workspace_summary.get(
        "dates",
        []
    )

    lines.append(
        "Date fields:"
    )

    if dates:

        for item in dates:

            lines.append(
                f"- {item.get('table')}.{item.get('column')}"
            )

    else:
        lines.append(
            "- None detected"
        )

    lines.append("")

    # --------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------

    lines.append(
        "Data quality:"
    )

    for quality in workspace_summary.get(
        "data_quality",
        [],
    ):

        lines.append(
            f"- {quality.get('table')}: "
            f"{quality.get('status')}, "
            f"{quality.get('duplicate_rows')} duplicate rows"
        )

        issues = quality.get(
            "issues",
            []
        )

        for issue in issues:

            if isinstance(
                issue,
                dict,
            ):

                lines.append(
                    f"  - {issue.get('column')}: "
                    f"{issue.get('issue_type')} "
                    f"({issue.get('severity')})"
                )

    lines.append("")

    # --------------------------------------------------
    # RELATIONSHIPS
    # --------------------------------------------------

    lines.append(
        "Potential relationships:"
    )

    relationships = workspace_summary.get(
        "relationships",
        []
    )

    if relationships:

        for relationship in relationships:

            lines.append(
                f"- "
                f"{relationship.get('left_table')}."
                f"{relationship.get('left_column')} "
                f"-> "
                f"{relationship.get('right_table')}."
                f"{relationship.get('right_column')} "
                f"({relationship.get('relationship_type')}, "
                f"{relationship.get('confidence')} confidence)"
            )

    else:
        lines.append(
            "- None detected"
        )

    lines.append("")

    # --------------------------------------------------
    # POSSIBLE TASKS
    # --------------------------------------------------

    lines.append(
        "Possible analytical tasks:"
    )

    for task in workspace_summary.get(
        "possible_analytical_tasks",
        [],
    ):

        lines.append(
            f"- {task}"
        )

    return "\n".join(lines)