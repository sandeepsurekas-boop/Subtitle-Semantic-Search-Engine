"""SentenceTransformer embeddings for semantic search."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.env_setup import configure_safe_runtime, configure_torch_threads
from src.logging_config import get_logger

configure_safe_runtime()

logger = get_logger("embeddings")


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str, device: str = "cpu"):
    from sentence_transformers import SentenceTransformer

    configure_torch_threads()
    logger.info("Loading embedding model: %s on %s", model_name, device)
    return SentenceTransformer(model_name, device=device)


def encode_texts(
    texts: list[str],
    model_name: str,
    batch_size: int = 8,
    show_progress: bool = False,
    device: str = "cpu",
) -> np.ndarray:
    if not texts:
        return np.array([])

    model = get_embedding_model(model_name, device=device)
    logger.debug("Encoding %d texts (batch_size=%d)", len(texts), batch_size)
    vectors = model.encode(
        texts,
        batch_size=min(batch_size, len(texts)),
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if len(texts) == 1:
        norm = float(np.linalg.norm(vectors))
        logger.debug("Query vector L2 norm=%.4f (expect ~1.0 if normalized)", norm)
    return vectors


def encode_query(query: str, model_name: str, device: str = "cpu") -> list[float]:
    vec = encode_texts([query], model_name, device=device)[0]
    return vec.tolist()


def cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Both sides assumed L2-normalized → dot product equals cosine similarity."""
    return doc_matrix @ query_vec
