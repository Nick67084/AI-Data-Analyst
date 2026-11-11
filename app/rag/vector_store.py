from pathlib import Path
import json
import faiss
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[3]

RAG_DATA_DIR = BASE_DIR / "data" / "rag"

INDEX_PATH = RAG_DATA_DIR / "faiss.index"
METADATA_PATH = RAG_DATA_DIR / "metadata.json"


def ensure_directory():
    RAG_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def create_vector_store(
    embeddings,
    chunks,
    source_name
):
    ensure_directory()

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    metadata = []

    for i, chunk in enumerate(chunks):
        metadata.append(
            {
                "id": i,
                "source": source_name,
                "text": chunk
            }
        )

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )


def load_vector_store():
    if not INDEX_PATH.exists():
        return None, []

    if not METADATA_PATH.exists():
        return None, []

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    return index, metadata


def search_vector_store(
    query_embedding,
    top_k=5
):
    index, metadata = load_vector_store()

    if index is None:
        return []

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(
            1,
            -1
        )

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):
        if index_id < 0:
            continue

        item = metadata[index_id].copy()

        item["score"] = float(score)

        results.append(item)

    return results