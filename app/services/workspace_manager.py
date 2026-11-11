from pathlib import Path
from typing import Any

from app.services.dataset_service import (
    is_supported_file,
    load_dataset,
    save_uploaded_file,
)


BASE_DIR = Path(__file__).resolve().parents[3]

WORKSPACE_ROOT = (
    BASE_DIR
    / "data"
    / "workspaces"
)


def create_workspace(workspace_id: str) -> Path:
    """
    Create a directory for one analysis workspace.

    Example:
        data/workspaces/workspace_001/
    """

    if not workspace_id:
        raise ValueError(
            "Workspace ID must not be empty."
        )

    safe_workspace_id = Path(
        workspace_id
    ).name

    if not safe_workspace_id:
        raise ValueError(
            "Invalid workspace ID."
        )

    workspace_path = (
        WORKSPACE_ROOT
        / safe_workspace_id
    )

    workspace_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return workspace_path


def get_workspace_path(
    workspace_id: str,
) -> Path:
    """
    Return the directory path for a workspace.
    """

    safe_workspace_id = Path(
        workspace_id
    ).name

    workspace_path = (
        WORKSPACE_ROOT
        / safe_workspace_id
    )

    return workspace_path


def save_workspace_file(
    file_content: bytes,
    original_filename: str,
    workspace_id: str,
) -> Path:
    """
    Save an uploaded CSV or Excel file
    inside a specific workspace.
    """

    if not original_filename:
        raise ValueError(
            "Uploaded file must have a filename."
        )

    if not is_supported_file(
        original_filename
    ):
        raise ValueError(
            "Unsupported file type. "
            "Supported formats: CSV, XLSX, XLS."
        )

    workspace_path = create_workspace(
        workspace_id
    )

    safe_filename = Path(
        original_filename
    ).name

    destination = (
        workspace_path
        / safe_filename
    )

    destination.write_bytes(
        file_content
    )

    return destination


def list_workspace_files(
    workspace_id: str,
) -> list[dict[str, Any]]:
    """
    Return supported files stored
    inside a workspace.
    """

    workspace_path = get_workspace_path(
        workspace_id
    )

    if not workspace_path.exists():
        return []

    files = []

    for file_path in sorted(
        workspace_path.iterdir()
    ):
        if (
            file_path.is_file()
            and is_supported_file(file_path)
        ):
            files.append(
                {
                    "filename": file_path.name,
                    "extension": file_path.suffix.lower(),
                    "size_bytes": file_path.stat().st_size,
                }
            )

    return files


def load_workspace_tables(
    workspace_id: str,
) -> dict[str, Any]:
    """
    Load all supported files in a workspace.

    Every CSV becomes one table.

    Every Excel worksheet becomes
    one table.

    Table names are made unique by
    including the filename when necessary.
    """

    workspace_path = get_workspace_path(
        workspace_id
    )

    if not workspace_path.exists():
        raise FileNotFoundError(
            f"Workspace was not found: "
            f"{workspace_id}"
        )

    workspace_files = []

    for file_path in sorted(
        workspace_path.iterdir()
    ):
        if (
            file_path.is_file()
            and is_supported_file(file_path)
        ):
            workspace_files.append(
                file_path
            )

    if not workspace_files:
        raise ValueError(
            "The workspace does not contain "
            "any supported datasets."
        )

    all_tables = {}

    for file_path in workspace_files:
        tables = load_dataset(
            file_path
        )

        for table_name, dataframe in (
            tables.items()
        ):
            if table_name not in all_tables:
                all_tables[
                    table_name
                ] = dataframe
                continue

            unique_table_name = (
                f"{file_path.stem}__"
                f"{table_name}"
            )

            counter = 2

            while (
                unique_table_name
                in all_tables
            ):
                unique_table_name = (
                    f"{file_path.stem}__"
                    f"{table_name}__"
                    f"{counter}"
                )
                counter += 1

            all_tables[
                unique_table_name
            ] = dataframe

    return all_tables