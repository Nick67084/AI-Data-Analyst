from typing import Any

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel, Field

from app.services.dataset_service import (
    is_supported_file,
)

from app.services.workspace_analyzer import (
    analyze_persistent_workspace,
)

from app.services.workspace_manager import (
    create_workspace,
    list_workspace_files,
    save_workspace_file,
)

from app.services.dashboard_generator import (
    generate_dashboard_plan,
)

from app.services.dashboard_executor import (
    execute_workspace_dashboard,
)


router = APIRouter(
    prefix="/api/workspaces",
    tags=["Workspace Analysis"],
)


# =========================================================
# Dashboard Request Schema
# =========================================================

class DashboardRequest(BaseModel):
    request: str = Field(
        default=(
            "Create a useful business dashboard "
            "from the uploaded data."
        ),
        min_length=1,
    )


# =========================================================
# Create Workspace
# =========================================================

@router.post("/{workspace_id}/create")
def create_new_workspace(
    workspace_id: str,
) -> dict[str, Any]:
    """
    Create a new persistent analysis workspace.
    """

    try:
        workspace_path = create_workspace(
            workspace_id
        )

        return {
            "workspace_id": workspace_id,
            "workspace_path": str(
                workspace_path
            ),
            "message": (
                "Workspace created successfully."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to create workspace: "
                f"{exc}"
            ),
        ) from exc


# =========================================================
# Upload Workspace File
# =========================================================

@router.post("/{workspace_id}/files")
async def upload_workspace_file(
    workspace_id: str,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Upload one CSV or Excel file into
    an existing analysis workspace.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file must have "
                "a filename."
            ),
        )

    if not is_supported_file(
        file.filename
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Supported formats: "
                "CSV, XLSX, XLS."
            ),
        )

    try:
        file_content = await file.read()

        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        saved_path = save_workspace_file(
            file_content=file_content,
            original_filename=file.filename,
            workspace_id=workspace_id,
        )

        return {
            "workspace_id": workspace_id,
            "filename": file.filename,
            "saved_path": str(
                saved_path
            ),
            "message": (
                "File uploaded successfully."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to upload file: "
                f"{exc}"
            ),
        ) from exc


# =========================================================
# List Workspace Files
# =========================================================

@router.get("/{workspace_id}/files")
def get_workspace_files(
    workspace_id: str,
) -> dict[str, Any]:
    """
    List all uploaded files in a workspace.
    """

    try:
        files = list_workspace_files(
            workspace_id
        )

        return {
            "workspace_id": workspace_id,
            "file_count": len(files),
            "files": files,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to list workspace files: "
                f"{exc}"
            ),
        ) from exc


# =========================================================
# Analyze Existing Workspace
# =========================================================

@router.post("/{workspace_id}/analyze")
def analyze_existing_workspace(
    workspace_id: str,
) -> dict[str, Any]:
    """
    Analyze all datasets currently stored
    inside a persistent workspace.
    """

    try:
        result = analyze_persistent_workspace(
            workspace_id
        )

        return result

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Workspace analysis failed: "
                f"{exc}"
            ),
        ) from exc


# =========================================================
# Generate Dynamic AI Dashboard
# =========================================================

@router.post("/{workspace_id}/dashboard")
def generate_workspace_dashboard(
    workspace_id: str,
    dashboard_request: DashboardRequest,
) -> dict[str, Any]:
    """
    Analyze the workspace, generate an AI dashboard plan,
    validate the plan, and execute it against the real data.
    """

    # -----------------------------------------------------
    # 1. Validate workspace ID
    # -----------------------------------------------------

    if not workspace_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Workspace ID must not be empty.",
        )

    # -----------------------------------------------------
    # 2. Validate user dashboard request
    # -----------------------------------------------------

    user_request = dashboard_request.request.strip()

    if not user_request:
        raise HTTPException(
            status_code=400,
            detail=(
                "Dashboard request must not "
                "be empty."
            ),
        )

    try:
        # -------------------------------------------------
        # 3. Analyze workspace
        # -------------------------------------------------

        analysis_result = (
            analyze_persistent_workspace(
                workspace_id
            )
        )

        # -------------------------------------------------
        # 4. Get AI summary and LLM context
        # -------------------------------------------------

        ai_summary = analysis_result[
            "ai_summary"
        ]

        llm_context = analysis_result[
            "llm_context"
        ]

        # -------------------------------------------------
        # 5. Add the user's dashboard request
        # -------------------------------------------------

        llm_context_with_request = (
            f"{llm_context}\n\n"
            "USER DASHBOARD REQUEST:\n"
            f"{user_request}\n"
        )

        # -------------------------------------------------
        # 6. Generate dashboard plan
        # -------------------------------------------------

        dashboard_plan = (
            generate_dashboard_plan(
                ai_summary=ai_summary,
                llm_context=(
                    llm_context_with_request
                ),
            )
        )

        # -------------------------------------------------
        # 7. Execute the validated dashboard
        #
        # IMPORTANT:
        # The existing executor accepts the dashboard
        # plan as its second positional argument.
        # -------------------------------------------------

        executed_dashboard = (
            execute_workspace_dashboard(
                workspace_id,
                dashboard_plan["dashboard"],
            )
        )

        # -------------------------------------------------
        # 8. Return complete dynamic dashboard result
        # -------------------------------------------------

        return {
            "workspace_id": workspace_id,

            "user_request": user_request,

            "dashboard": (
                dashboard_plan["dashboard"]
            ),

            "validation": (
                dashboard_plan["validation"]
            ),

            "rejected_kpis": (
                dashboard_plan["rejected_kpis"]
            ),

            "rejected_charts": (
                dashboard_plan["rejected_charts"]
            ),

            "rejected_filters": (
                dashboard_plan["rejected_filters"]
            ),

            "execution": (
                executed_dashboard.get(
                    "execution",
                    {},
                )
            ),

            "executed_kpis": (
                executed_dashboard.get(
                    "kpis",
                    [],
                )
            ),

            "executed_dashboard": (
                executed_dashboard
            ),
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to generate "
                f"dashboard: {exc}"
            ),
        ) from exc