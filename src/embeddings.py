"""SentenceTransformer embeddings for semantic search."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str, device: str | None = None) -> SentenceTransformer:
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def encode_texts(
    texts: list[str],
    model_name: str,
    batch_size: int = 64,
    show_progress: bool = False,
    device: str | None = None,
) -> np.ndarray:
    model = get_embedding_model(model_name, device=device)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def encode_query(query: str, model_name: str, device: str | None = None) -> list[float]:
    vec = encode_texts([query], model_name, device=device)[0]
    return vec.tolist()


def cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Both sides assumed L2-normalized → dot product equals cosine similarity."""
    return doc_matrix @ query_vec
