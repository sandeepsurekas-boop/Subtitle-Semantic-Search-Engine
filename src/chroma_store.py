"""ChromaDB persistence for semantic subtitle chunks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.env_setup import configure_safe_runtime

configure_safe_runtime()

import chromadb
from chromadb.config import Settings

import config
from src.chunker import TextChunk


def get_client(persist_path: Path) -> chromadb.PersistentClient:
    persist_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_path),
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collections(client: chromadb.PersistentClient):
    chunks = client.get_or_create_collection(
        name=config.COLLECTION_SEMANTIC,
        metadata={"hnsw:space": "cosine"},
    )
    filenames = client.get_or_create_collection(
        name=config.COLLECTION_FILENAMES,
        metadata={"hnsw:space": "cosine"},
    )
    return chunks, filenames


def add_chunks_batch(
    collection,
    chunks: list[TextChunk],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]] | None = None,
    batch_size: int = 500,
) -> None:
    if not chunks:
        return
    if metadatas is None:
        metadatas = [
            {
                "subtitle_id": c.subtitle_id,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

    for i in range(0, len(chunks), batch_size):
        batch_c = chunks[i : i + batch_size]
        collection.add(
            ids=[c.chunk_id for c in batch_c],
            documents=[c.text for c in batch_c],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )


def upsert_filename_map(
    collection,
    subtitle_ids: list[str],
    filenames: list[str],
    embedding_dim: int = 8,
) -> None:
    """Map subtitle_id → filename for result display (lookup by id only)."""
    dummy = [[0.0] * embedding_dim for _ in subtitle_ids]
    collection.upsert(
        ids=subtitle_ids,
        documents=filenames,
        embeddings=dummy,
    )


def query_semantic(
    collection,
    query_embedding: list[float],
    n_results: int = 10,
) -> dict:
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )


def get_filename(collection, subtitle_id: str) -> str | None:
    try:
        res = collection.get(ids=[subtitle_id], include=["documents"])
        docs = res.get("documents") or []
        if not docs:
            return None
        doc = docs[0]
        if isinstance(doc, list):
            doc = doc[0] if doc else None
        return doc or None
    except Exception:
        pass
    return None


def upsert_filename_map_batched(
    collection,
    subtitle_ids: list[str],
    filenames: list[str],
    batch_size: int = 500,
    embedding_dim: int = 8,
) -> None:
    for i in range(0, len(subtitle_ids), batch_size):
        upsert_filename_map(
            collection,
            subtitle_ids[i : i + batch_size],
            filenames[i : i + batch_size],
            embedding_dim=embedding_dim,
        )


def extract_subtitle_id(chunk_id: str) -> str:
    """chunk id format: {subtitle_id}-{chunk_index}"""
    if "-" in chunk_id:
        return chunk_id.rsplit("-", 1)[0]
    return chunk_id


def semantic_collection_exists(client: chromadb.PersistentClient) -> bool:
    try:
        coll = client.get_collection(config.COLLECTION_SEMANTIC)
        return coll.count() > 0
    except Exception:
        return False


def get_index_status(
    chroma_path: Path = config.CHROMA_PATH,
    artifacts_dir: Path = config.ARTIFACTS_DIR,
    db_path: Path = config.DB_PATH,
) -> dict:
    """Report what is ready so UIs can show setup instructions."""
    status = {
        "database": db_path.exists(),
        "semantic_index": False,
        "semantic_chunks": 0,
        "keyword_index": (artifacts_dir / "tfidf_vectorizer.joblib").exists(),
    }
    if chroma_path.exists():
        try:
            client = get_client(chroma_path)
            coll = client.get_collection(config.COLLECTION_SEMANTIC)
            status["semantic_chunks"] = coll.count()
            status["semantic_index"] = status["semantic_chunks"] > 0
            fn = client.get_collection(config.COLLECTION_FILENAMES)
            status["indexed_subtitles"] = fn.count()
        except Exception:
            pass
    return status


INGEST_INSTRUCTIONS = """
Index not built yet. Run ingest first (from project root, venv active):

  python scripts/download_data.py --check
  python -m src.ingest --mode both --sample 0.1    # quick test
  python -m src.ingest --mode both                 # full index

Then restart Streamlit.
""".strip()
