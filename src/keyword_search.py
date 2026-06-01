"""TF-IDF keyword-based search (sparse vectors, cosine similarity)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.logging_config import get_logger

logger = get_logger("keyword")


@dataclass
class KeywordHit:
    chunk_id: str
    subtitle_id: str
    filename: str
    score: float
    snippet: str


class KeywordSearchEngine:
    """Bag-of-words / TF-IDF index over subtitle chunks."""

    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix: sparse.csr_matrix | None = None
        self.meta: pd.DataFrame | None = None

    def fit(self, chunk_ids: list[str], texts: list[str], meta: pd.DataFrame) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1, 2),
            stop_words=None,  # "I", "a" matter for movie quotes
            sublinear_tf=True,
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(texts)
        self.meta = meta.copy()
        self.meta["chunk_id"] = chunk_ids

    def save(self, vectorizer_path: Path, matrix_path: Path, meta_path: Path) -> None:
        vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, vectorizer_path)
        sparse.save_npz(str(matrix_path), self.matrix)
        self.meta.to_parquet(meta_path, index=False)

    def load(
        self,
        vectorizer_path: Path,
        matrix_path: Path,
        meta_path: Path,
    ) -> None:
        self.vectorizer = joblib.load(vectorizer_path)
        self.matrix = sparse.load_npz(str(matrix_path))
        self.meta = pd.read_parquet(meta_path)

    def search(self, query: str, top_k: int = 10, log_scores: bool = False) -> list[KeywordHit]:
        if self.vectorizer is None or self.matrix is None or self.meta is None:
            raise RuntimeError("Keyword index not built. Run ingest with --mode keyword.")

        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).ravel()
        top_idx = np.argsort(scores)[::-1][:top_k]

        if log_scores:
            logger.info("TF-IDF cosine similarity (top %d of %d chunks):", top_k, len(scores))
            for rank, idx in enumerate(top_idx, start=1):
                row = self.meta.iloc[int(idx)]
                logger.info(
                    "  #%02d idx=%d chunk_id=%s cosine_sim=%.6f | %s",
                    rank,
                    idx,
                    row["chunk_id"],
                    float(scores[idx]),
                    str(row.get("text", ""))[:100],
                )

        hits: list[KeywordHit] = []
        for idx in top_idx:
            row = self.meta.iloc[int(idx)]
            hits.append(
                KeywordHit(
                    chunk_id=row["chunk_id"],
                    subtitle_id=str(row["subtitle_id"]),
                    filename=str(row.get("filename", "")),
                    score=float(scores[idx]),
                    snippet=str(row.get("text", ""))[:300],
                )
            )
        return hits
