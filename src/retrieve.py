"""Part 2: Retrieve subtitles by text or audio query (cosine similarity)."""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

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
from src.logging_config import get_logger, setup_logging

logger = get_logger("retrieve")


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
    chunk_id: str = ""
    cosine_distance: float | None = None


def _phrase_boost(query: str, chunk_text: str) -> float:
    q = re.sub(r"\s+", " ", query.lower()).strip()
    t = re.sub(r"\s+", " ", chunk_text.lower()).strip()
    if not q or not t:
        return 0.0
    if q in t:
        return config.PHRASE_EXACT_BOOST
    words = [w for w in q.split() if len(w) > 2]
    if words and all(w in t for w in words):
        return config.PHRASE_WORD_BOOST
    return 0.0


def _rerank_and_dedupe(
    query: str, hits: list[SearchResult], top_k: int
) -> list[SearchResult]:
    """Vector score + exact-phrase boost, then one result per subtitle file."""
    boosted: list[SearchResult] = []
    for h in hits:
        boost = _phrase_boost(query, h.snippet)
        if boost > 0:
            logger.info(
                "Phrase boost +%.2f on chunk=%s | %r",
                boost,
                h.chunk_id,
                h.snippet[:100],
            )
        boosted.append(
            SearchResult(
                subtitle_id=h.subtitle_id,
                filename=h.filename,
                score=min(1.0, h.score + boost),
                snippet=h.snippet,
                opensubtitles_url=h.opensubtitles_url,
                chunk_id=h.chunk_id,
                cosine_distance=h.cosine_distance,
            )
        )
    boosted.sort(key=lambda x: x.score, reverse=True)
    seen: set[str] = set()
    out: list[SearchResult] = []
    for h in boosted:
        if h.subtitle_id in seen:
            continue
        seen.add(h.subtitle_id)
        out.append(h)
        if len(out) >= top_k:
            break
    return out


def _log_semantic_raw(query: str, cleaned: str, raw: dict, top_k: int) -> None:
    ids = (raw.get("ids") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    logger.info("=" * 60)
    logger.info("SEMANTIC SEARCH")
    logger.info("  Raw query     : %r", query)
    logger.info("  Cleaned query : %r", cleaned)
    logger.info("  Chroma hits   : %d (before dedupe, requested %d)", len(ids), top_k * 3)
    logger.info("  Score note    : similarity = 1 - cosine_distance (Chroma hnsw:space=cosine)")
    for rank, (cid, dist, doc) in enumerate(zip(ids, dists, docs), start=1):
        sim = 1.0 - float(dist) if dist is not None else 0.0
        snippet = (doc or "")[:120].replace("\n", " ")
        logger.info(
            "  #%02d chunk_id=%s | cosine_distance=%.6f | cosine_similarity=%.6f | %s",
            rank,
            cid,
            float(dist) if dist is not None else -1.0,
            sim,
            snippet,
        )
    logger.info("=" * 60)


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

        logger.info(
            "Engine init: model=%s chroma_chunks=%s keyword_index=%s",
            model_name,
            self.status.get("semantic_chunks"),
            self.status.get("keyword_index"),
        )

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

        raw_query = query
        q = clean_query(query)
        logger.debug("Encoding query with model=%s", self.model_name)
        emb = encode_query(q, self.model_name)

        n_fetch = min(top_k * 3, max(self.chunks.count(), 1))
        raw = query_semantic(self.chunks, emb, n_results=n_fetch)
        _log_semantic_raw(raw_query, q, raw, top_k)

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
            dist_f = float(dist) if dist is not None else None
            score = 1.0 - dist_f if dist_f is not None else 0.0
            hits.append(
                SearchResult(
                    subtitle_id=sid,
                    filename=fname,
                    score=score,
                    snippet=(doc or "")[:400],
                    opensubtitles_url=f"https://www.opensubtitles.org/en/subtitles/{sid}",
                    chunk_id=cid,
                    cosine_distance=dist_f,
                )
            )

        final = _rerank_and_dedupe(q, hits, top_k)
        logger.info("Returning %d results after rerank+dedupe (top_k=%d)", len(final), top_k)
        for rank, r in enumerate(final, 1):
            logger.info(
                "  FINAL #%d subtitle_id=%s similarity=%.4f file=%s",
                rank,
                r.subtitle_id,
                r.score,
                r.filename or "(unknown)",
            )
        return final

    def search_keyword(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if self.keyword is None:
            raise RuntimeError("TF-IDF index missing. Run: python -m src.ingest --mode keyword")

        raw_query = query
        q = clean_query(query)
        logger.info("KEYWORD SEARCH raw=%r cleaned=%r", raw_query, q)
        raw_hits = self.keyword.search(q, top_k=top_k * 3, log_scores=True)
        hits = [
            SearchResult(
                subtitle_id=h.subtitle_id,
                filename=h.filename,
                score=h.score,
                snippet=h.snippet,
                opensubtitles_url=f"https://www.opensubtitles.org/en/subtitles/{h.subtitle_id}",
                chunk_id=h.chunk_id,
            )
            for h in raw_hits
        ]
        final = _rerank_and_dedupe(q, hits, top_k)
        logger.info("Returning %d keyword results after rerank", len(final))
        return final

    def search_audio(
        self,
        audio_path: Path,
        mode: str = "semantic",
        top_k: int = 10,
        whisper_model: str = config.WHISPER_MODEL,
    ) -> tuple[str, list[SearchResult]]:
        logger.info("Audio search: file=%s mode=%s", audio_path, mode)
        transcript = transcribe_audio(audio_path, model_size=whisper_model)
        logger.info("Whisper transcript (%d chars): %r", len(transcript), transcript[:200])
        if mode == "keyword":
            return transcript, self.search_keyword(transcript, top_k)
        return transcript, self.search_semantic(transcript, top_k)


def _print_results(results: list[SearchResult], transcript: str | None = None) -> None:
    if transcript:
        print(f"\nTranscript ({len(transcript)} chars):\n{transcript[:500]}...\n")
    for i, r in enumerate(results, 1):
        dist = f" dist={r.cosine_distance:.4f}" if r.cosine_distance is not None else ""
        print(f"{i}. [sim={r.score:.4f}{dist}] {r.filename or r.subtitle_id}")
        print(f"   chunk={r.chunk_id}")
        print(f"   {r.opensubtitles_url}")
        print(f"   {r.snippet[:200]}...\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search subtitle index")
    parser.add_argument("--query", type=str, help="Text query (dialogue)")
    parser.add_argument("--audio", type=Path, help="Audio query (~2 min clip)")
    parser.add_argument("--mode", choices=["semantic", "keyword"], default="semantic")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose logs (cosine distance per chunk) → console + logs/retrieval.log",
    )
    args = parser.parse_args()

    setup_logging("DEBUG" if args.debug else config.LOG_LEVEL)

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
