import requests

from app.services.analytics import (
    get_kpis,
    get_monthly_revenue,
    get_top_products,
)

from app.services.ml_services import (
    get_forecast,
    get_anomalies,
    get_customer_segments,
)

from app.services.rag_service import (
    retrieve_context,
    load_vector_store,
)


OLLAMA_URL = (
    "http://localhost:11434/api/generate"
)

OLLAMA_MODEL = "llama3.2:3b"


def build_structured_context():
    """
    Collect the existing analytics and ML results
    that the AI Analyst can use.
    """

    monthly_revenue = get_monthly_revenue()

    top_products = get_top_products(
        limit=10
    )

    forecast = get_forecast()

    anomalies = get_anomalies()

    segments = get_customer_segments()

    return {
        "kpis": get_kpis(),

        "recent_monthly_revenue": (
            monthly_revenue[-6:]
        ),

        "top_products": top_products,

        "forecast_next_30_days": (
            forecast[:30]
        ),

        "top_anomalies": (
            anomalies[:10]
        ),

        "customer_segments": segments,
    }


def ask_combined_analyst(
    question: str,
    top_k: int = 5,
):
    """
    Combine structured business analytics
    with retrieved document information.
    """

    structured = (
        build_structured_context()
    )

    # ----------------------------------------------
    # Retrieve relevant document information
    # ----------------------------------------------

    index, _ = load_vector_store()

    retrieved_documents = []

    if index is not None:
        retrieved_documents = (
            retrieve_context(
                question,
                top_k=top_k,
            )
        )

    # ----------------------------------------------
    # Build document context
    # ----------------------------------------------

    document_parts = []

    for index_number, document in enumerate(
        retrieved_documents,
        start=1,
    ):
        document_parts.append(
            f"""
DOCUMENT SOURCE {index_number}
File: {document["source"]}
Similarity: {document["score"]:.4f}

{document["text"]}
"""
        )

    document_context = "\n".join(
        document_parts
    )

    if not document_context:
        document_context = (
            "No business document has been "
            "uploaded or no relevant document "
            "context was found."
        )

    # ----------------------------------------------
    # Prompt for Llama
    # ----------------------------------------------

    prompt = f"""
You are an AI Data Analyst Assistant
for a retail business.

Your job is to analyze business information
and provide useful, evidence-based answers.

You have access to TWO information sources:

1. STRUCTURED BUSINESS DATA
   - KPIs
   - Monthly revenue
   - Top products
   - Sales forecast
   - Detected anomalies
   - Customer segments

2. BUSINESS DOCUMENTS
   - Company reports
   - Business strategies
   - Policies
   - Marketing information
   - Other uploaded documents

Use both sources when they are relevant.

IMPORTANT RULES:

- Use structured business data for numerical
  and analytical claims.

- Use business document context for company
  strategies, policies, explanations, and
  information contained in documents.

- Do not invent numbers.

- Do not invent business strategies.

- Do not invent causes for changes in revenue.

- If the available information does not prove
  something, clearly say that the available
  data does not establish it.

- When useful, distinguish between:
  "According to the sales data..."
  and
  "According to the company document..."

- Give a clear and concise business-oriented
  explanation.

- Do not mention RAG, embeddings, FAISS,
  vector databases, or internal implementation
  unless the user specifically asks about them.

STRUCTURED BUSINESS DATA:

{structured}


RETRIEVED BUSINESS DOCUMENT CONTEXT:

{document_context}


USER QUESTION:

{question}


ANSWER:
"""

    # ----------------------------------------------
    # Send request to Ollama
    # ----------------------------------------------

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()

    answer = result.get(
        "response",
        "The AI model did not return an answer.",
    ).strip()

    # ----------------------------------------------
    # Return result
    # ----------------------------------------------

    return {
        "question": question,

        "answer": answer,

        "data_context": {
            "kpis": structured["kpis"],

            "recent_months": (
                structured[
                    "recent_monthly_revenue"
                ]
            ),

            "top_products_count": len(
                structured["top_products"]
            ),

            "forecast_days": len(
                structured[
                    "forecast_next_30_days"
                ]
            ),

            "anomalies_count": len(
                structured[
                    "top_anomalies"
                ]
            ),

            "customer_segments_count": len(
                structured[
                    "customer_segments"
                ]
            ),
        },

        "document_sources": [
            {
                "source": document["source"],

                "score": round(
                    document["score"],
                    4,
                ),

                "text": document["text"],
            }

            for document in retrieved_documents
        ],
    }