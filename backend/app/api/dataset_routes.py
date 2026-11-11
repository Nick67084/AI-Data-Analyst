from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.schemas.dataset_schema import DatasetProfile

from app.services.dataset_service import (
    is_supported_file,
    profile_dataset,
    save_uploaded_file,
)

from app.services.workspace_analyzer import (
    analyze_persistent_workspace,
)

from app.services.dashboard_generator import (
    generate_dashboard_plan,
)

from app.services.dashboard_executor import (
    execute_workspace_dashboard,
)


router = APIRouter(
    prefix="/api",
    tags=["Dataset Workspace"],
)


# =========================================================
# Request schema for AI dashboard generation
# =========================================================

class DashboardRequest(BaseModel):
    request: str = Field(
        default="Create a useful business dashboard from the uploaded data.",
        min_length=1,
    )


# =========================================================
# Upload Dataset
# =========================================================

@router.post(
    "/datasets/upload",
    response_model=DatasetProfile,
)
async def upload_dataset(
    file: UploadFile = File(...),
):
    """
    Upload a CSV or Excel dataset,
    save it to the workspace,
    and generate its dataset profile.
    """

    # -----------------------------------------------------
    # 1. Check filename
    # -----------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename.",
        )

    # -----------------------------------------------------
    # 2. Check file extension
    # -----------------------------------------------------

    if not is_supported_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Supported formats: CSV, XLSX, XLS."
            ),
        )

    try:
        # -------------------------------------------------
        # 3. Read uploaded file as bytes
        # -------------------------------------------------

        file_content = await file.read()

        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        # -------------------------------------------------
        # 4. Save bytes to workspace
        # -------------------------------------------------

        saved_path = save_uploaded_file(
            file_content=file_content,
            original_filename=file.filename,
        )

        # -------------------------------------------------
        # 5. Convert saved path to Path object
        # -------------------------------------------------

        saved_path = Path(saved_path)

        # -------------------------------------------------
        # 6. Profile the saved dataset
        # -------------------------------------------------

        profile = profile_dataset(
            saved_path
        )

        return profile

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process dataset: {exc}",
        ) from exc


# =========================================================
# Generate Dynamic AI Dashboard
# =========================================================

@router.post(
    "/workspace/{workspace_id}/dashboard",
)
async def generate_workspace_dashboard(
    workspace_id: str,
    dashboard_request: DashboardRequest,
):
    """
    Analyze a workspace, generate an AI dashboard plan,
    validate it, and execute it against the real data.
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
    # 2. Validate dashboard request
    # -----------------------------------------------------

    user_request = dashboard_request.request.strip()

    if not user_request:
        raise HTTPException(
            status_code=400,
            detail="Dashboard request must not be empty.",
        )

    try:
        # -------------------------------------------------
        # 3. Analyze the workspace
        # -------------------------------------------------

        analysis_result = analyze_persistent_workspace(
            workspace_id
        )

        # -------------------------------------------------
        # 4. Get AI summary and compact LLM context
        # -------------------------------------------------

        ai_summary = analysis_result["ai_summary"]
        llm_context = analysis_result["llm_context"]

        # -------------------------------------------------
        # 5. Add the user's dashboard request to the
        #    existing LLM context.
        #
        #    The current dashboard generator accepts
        #    llm_context, so we can pass the request through
        #    without changing the generator yet.
        # -------------------------------------------------

        llm_context_with_request = (
            f"{llm_context}\n\n"
            "USER DASHBOARD REQUEST:\n"
            f"{user_request}\n"
        )

        # -------------------------------------------------
        # 6. Generate dashboard plan
        # -------------------------------------------------

        dashboard_plan = generate_dashboard_plan(
            ai_summary=ai_summary,
            llm_context=llm_context_with_request,
        )

        # -------------------------------------------------
        # 7. Execute validated dashboard against real data
        # -------------------------------------------------

        executed_dashboard = execute_workspace_dashboard(
            workspace_id=workspace_id,
            dashboard=dashboard_plan["dashboard"],
        )

        # -------------------------------------------------
        # 8. Return complete result
        # -------------------------------------------------

        return {
            "workspace_id": workspace_id,
            "user_request": user_request,

            "dashboard": dashboard_plan["dashboard"],

            "validation": dashboard_plan["validation"],

            "rejected_kpis": dashboard_plan[
                "rejected_kpis"
            ],

            "rejected_charts": dashboard_plan[
                "rejected_charts"
            ],

            "rejected_filters": dashboard_plan[
                "rejected_filters"
            ],

            "execution": executed_dashboard.get(
                "execution",
                {},
            ),

            "executed_kpis": executed_dashboard.get(
                "kpis",
                [],
            ),

            "executed_dashboard": executed_dashboard,
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
            detail=f"Failed to generate dashboard: {exc}",
        ) from exc