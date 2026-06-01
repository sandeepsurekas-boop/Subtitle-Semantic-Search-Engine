"""Part 2: Retrieve subtitles by text or audio query (cosine similarity)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import chromadb

import config
from src.audio_query import transcribe_audio
from src.chroma_store import (
    INGEST_INSTRUCTIONS,
    extract_subtitle_id,
    get_client,
    get_filename,
    get_index_status,
    query_semantic,
    semantic_collection_exists,
)
from src.cleaning import clean_query
from src.embeddings import encode_query
from src.keyword_search import KeywordSearchEngine


class IndexNotReadyError(RuntimeError):
    """Raised when ChromaDB or TF-IDF index has not been built via ingest."""

    def __init__(self, message: str, status: dict | None = None):
        super().__init__(message)
        self.status = status or {}


@dataclass
class SearchResult:
    subtitle_id: str
    filename: str
    score: float
    snippet: str
    opensubtitles_url: str


def _dedupe_by_subtitle(hits: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    out: list[SearchResult] = []
    for h in sorted(hits, key=lambda x: x.score, reverse=True):
        if h.subtitle_id in seen:
            continue
        seen.add(h.subtitle_id)
        out.append(h)
    return out


class SubtitleSearchEngine:
    def __init__(
        self,
        chroma_path: Path = config.CHROMA_PATH,
        model_name: str = config.EMBEDDING_MODEL,
        artifacts_dir: Path = config.ARTIFACTS_DIR,
        require_semantic: bool = True,
        require_keyword: bool = False,
    ):
        self.model_name = model_name
        self.chroma_path = chroma_path
        self.artifacts_dir = artifacts_dir
        self.status = get_index_status(chroma_path, artifacts_dir)

        self.client = get_client(chroma_path)
        self.chunks = None
        if semantic_collection_exists(self.client):
            self.chunks = self.client.get_collection(config.COLLECTION_SEMANTIC)

        if require_semantic and self.chunks is None:
            raise IndexNotReadyError(
                f"Semantic index missing (collection '{config.COLLECTION_SEMANTIC}').\n"
                f"{INGEST_INSTRUCTIONS}",
                status=self.status,
            )

        try:
            self.filenames = self.client.get_collection(config.COLLECTION_FILENAMES)
        except Exception:
            self.filenames = None

        self.keyword: KeywordSearchEngine | None = None
        v_path = artifacts_dir / "tfidf_vectorizer.joblib"
        if v_path.exists():
            self.keyword = KeywordSearchEngine()
            self.keyword.load(
                v_path,
                artifacts_dir / "tfidf_matrix.npz",
                artifacts_dir / "tfidf_metadata.parquet",
            )

        if require_keyword and self.keyword is None:
            raise IndexNotReadyError(
                "Keyword (TF-IDF) index missing. Run: python -m src.ingest --mode keyword",
                status=self.status,
            )

    @classmethod
    def create_if_ready(cls, mode: str = "semantic", **kwargs) -> "SubtitleSearchEngine":
        return cls(
            require_semantic=mode == "semantic",
            require_keyword=mode == "keyword",
            **kwargs,
        )

    def search_semantic(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if self.chunks is None:
            raise IndexNotReadyError(
                f"Semantic index not built.\n{INGEST_INSTRUCTIONS}",
                status=self.status,
            )
        q = clean_query(query)
        emb = encode_query(q, self.model_name)
        raw = query_semantic(self.chunks, emb, n_results=top_k * 3)

        hits: list[SearchResult] = []
        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        metas = (raw.get("metadatas") or [[]])[0]
        for i, (cid, doc, dist) in enumerate(zip(ids, docs, dists)):
            sid = extract_subtitle_id(cid)
            fname = ""
            if metas and i < len(metas) and metas[i]:
                fname = metas[i].get("filename") or ""
            if not fname and self.filenames:
                fname = get_filename(self.filenames, sid) or ""
            # Chroma cosine distance: lower is better → similarity ≈ 1 - dist
            score = 1.0 - float(dist) if dist is not None else 0.0
            hits.append(
                SearchResult(
                    subtitle_id=sid,
                    filename=fname,
                    score=score,
                    snippet=(doc or "")[:400],
                    opensubtitles_url=f"https://www.opensubtitles.org/en/subtitles/{sid}",
                )
            )
        return _dedupe_by_subtitle(hits)[:top_k]

    def search_keyword(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if self.keyword is None:
            raise RuntimeError("TF-IDF index missing. Run: python -m src.ingest --mode keyword")
        q = clean_query(query)
        raw = self.keyword.search(q, top_k=top_k * 3)
        hits = [
            SearchResult(
                subtitle_id=h.subtitle_id,
                filename=h.filename,
                score=h.score,
                snippet=h.snippet,
                opensubtitles_url=f"https://www.opensubtitles.org/en/subtitles/{h.subtitle_id}",
            )
            for h in raw
        ]
        return _dedupe_by_subtitle(hits)[:top_k]

    def search_audio(
        self,
        audio_path: Path,
        mode: str = "semantic",
        top_k: int = 10,
        whisper_model: str = config.WHISPER_MODEL,
    ) -> tuple[str, list[SearchResult]]:
        transcript = transcribe_audio(audio_path, model_size=whisper_model)
        if mode == "keyword":
            return transcript, self.search_keyword(transcript, top_k)
        return transcript, self.search_semantic(transcript, top_k)


def _print_results(results: list[SearchResult], transcript: str | None = None) -> None:
    if transcript:
        print(f"\nTranscript ({len(transcript)} chars):\n{transcript[:500]}...\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.score:.4f}] {r.filename or r.subtitle_id}")
        print(f"   {r.opensubtitles_url}")
        print(f"   {r.snippet[:200]}...\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search subtitle index")
    parser.add_argument("--query", type=str, help="Text query (dialogue)")
    parser.add_argument("--audio", type=Path, help="Audio query (~2 min clip)")
    parser.add_argument(
        "--mode",
        choices=["semantic", "keyword"],
        default="semantic",
    )
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    args = parser.parse_args()

    engine = SubtitleSearchEngine.create_if_ready(mode=args.mode)

    if args.audio:
        transcript, results = engine.search_audio(args.audio, mode=args.mode, top_k=args.top_k)
        _print_results(results, transcript)
    elif args.query:
        if args.mode == "keyword":
            results = engine.search_keyword(args.query, args.top_k)
        else:
            results = engine.search_semantic(args.query, args.top_k)
        _print_results(results)
    else:
        parser.error("Provide --query or --audio")


if __name__ == "__main__":
    main()
