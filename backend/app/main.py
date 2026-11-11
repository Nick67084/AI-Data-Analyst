from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.api.dataset_routes import router as dataset_router
from app.api.workspace_routes import router as workspace_router
from app.services.analytics import (
    get_kpis,
    get_monthly_revenue,
    get_top_products,
)

from app.services.ml_services import (
    get_forecast,
    get_anomalies,
    get_customer_segments,
    get_ml_model_status,
)

from app.services.rag_service import (
    process_document,
    ask_question,
    get_rag_status,
    DOCUMENTS_DIR,
)

from app.services.analyst_service import (
    ask_combined_analyst,
)


app = FastAPI(
    title="AI Data Analyst API",
    description=(
        "AI-powered business data analysis, machine learning, "
        "RAG, and generative AI API"
    ),
    version="3.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# DATASET WORKSPACE
# ==================================================

app.include_router(dataset_router)
app.include_router(workspace_router)


# ==================================================
# REQUEST MODELS
# ==================================================

class RAGQuestion(BaseModel):
    question: str
    top_k: int = 5


# ==================================================
# ROOT / HEALTH
# ==================================================

@app.get("/")
def root():
    return {
        "message": "AI Data Analyst API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ==================================================
# ANALYTICS
# ==================================================

@app.get("/api/kpis")
def kpis():
    return get_kpis()


@app.get("/api/monthly-revenue")
def monthly_revenue():
    return get_monthly_revenue()


@app.get("/api/top-products")
def top_products():
    return get_top_products()


# ==================================================
# MACHINE LEARNING
# ==================================================

@app.get("/api/forecast")
def forecast():
    return get_forecast()


@app.get("/api/anomalies")
def anomalies():
    return get_anomalies()


@app.get("/api/customer-segments")
def customer_segments():
    return get_customer_segments()


@app.get("/api/ml-status")
def ml_status():
    return get_ml_model_status()


# ==================================================
# RAG
# ==================================================

@app.post("/api/rag/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    allowed_extensions = {
        ".pdf",
        ".txt",
        ".docx",
    }

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename was provided.",
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Please upload PDF, TXT, or DOCX."
            ),
        )

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_filename = Path(
        file.filename
    ).name

    file_path = DOCUMENTS_DIR / safe_filename

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded document is empty.",
        )

    with open(
        file_path,
        "wb",
    ) as output_file:
        output_file.write(contents)

    try:
        result = process_document(
            file_path
        )

        return {
            "message": (
                "Document processed successfully."
            ),
            **result,
        }

    except Exception as error:
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


@app.post("/api/rag/ask")
def rag_question(
    request: RAGQuestion,
):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        return ask_question(
            question=question,
            top_k=request.top_k,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


@app.get("/api/rag/status")
def rag_status():
    return get_rag_status()


# ==================================================
# COMBINED AI DATA ANALYST
# ==================================================

@app.post("/api/analyst/ask")
def analyst_question(
    request: RAGQuestion,
):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        return ask_combined_analyst(
            question=question,
            top_k=request.top_k,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error