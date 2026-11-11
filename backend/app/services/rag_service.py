from pathlib import Path
import requests

from app.rag.document_loader import load_document
from app.rag.chunker import chunk_text
from app.rag.embeddings import create_embeddings
from app.rag.vector_store import (
    create_vector_store,
    search_vector_store,
    load_vector_store
)


BASE_DIR = Path(__file__).resolve().parents[3]

DOCUMENTS_DIR = BASE_DIR / "documents"

OLLAMA_URL = "http://localhost:11434/api/generate"

OLLAMA_MODEL = "llama3.2:3b"


def ensure_documents_directory():
    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def process_document(file_path: Path):
    ensure_documents_directory()

    text = load_document(file_path)

    if not text.strip():
        raise ValueError(
            "The uploaded document does not contain readable text."
        )

    chunks = chunk_text(text)

    if not chunks:
        raise ValueError(
            "Could not create text chunks from the document."
        )

    embeddings = create_embeddings(chunks)

    create_vector_store(
        embeddings=embeddings,
        chunks=chunks,
        source_name=file_path.name
    )

    return {
        "filename": file_path.name,
        "characters": len(text),
        "chunks": len(chunks)
    }


def retrieve_context(question, top_k=5):
    query_embedding = create_embeddings(
        [question]
    )

    results = search_vector_store(
        query_embedding,
        top_k=top_k
    )

    return results


def generate_answer(question, retrieved_documents):
    if not retrieved_documents:
        return (
            "I could not find relevant information "
            "in the uploaded documents."
        )

    context_parts = []

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):
        context_parts.append(
            f"""
SOURCE {i}
File: {document["source"]}

{document["text"]}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are an AI Data Analyst Assistant.

Answer the user's question using the provided document context.

Rules:
- Use the provided context as the primary source.
- Do not invent facts that are not supported by the context.
- If the context does not contain enough information, clearly say so.
- Give a clear and concise business-oriented answer.
- Mention the source file when useful.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    return result.get(
        "response",
        "The AI model did not return an answer."
    ).strip()


def ask_question(question, top_k=5):
    retrieved_documents = retrieve_context(
        question,
        top_k=top_k
    )

    answer = generate_answer(
        question,
        retrieved_documents
    )

    sources = []

    for document in retrieved_documents:
        sources.append(
            {
                "source": document["source"],
                "score": round(
                    document["score"],
                    4
                ),
                "text": document["text"]
            }
        )

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }


def get_rag_status():
    index, metadata = load_vector_store()

    return {
        "ollama_model": OLLAMA_MODEL,
        "ollama_url": OLLAMA_URL,
        "vector_store_ready": index is not None,
        "stored_chunks": len(metadata)
    }