"""Part 1: Ingest subtitle database → clean → chunk → embed → ChromaDB / TF-IDF."""
from __future__ import annotations

import argparse
import gc
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Must run before heavy native libs load
from src.env_setup import configure_safe_runtime

configure_safe_runtime()

import config  # noqa: E402
from src.logging_config import get_logger, setup_logging  # noqa: E402

logger = get_logger("ingest")
from src.chunker import chunk_corpus  # noqa: E402
from src.cleaning import clean_subtitle_text  # noqa: E402
from src.database import iter_subtitle_batches  # noqa: E402
from src.embeddings import encode_texts  # noqa: E402
from src.chroma_store import (  # noqa: E402
    add_chunks_batch,
    get_client,
    get_or_create_collections,
    upsert_filename_map_batched,
)
from src.keyword_search import KeywordSearchEngine  # noqa: E402


def prepare_records(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Return (id, filename, raw_subtitle_text) — chunker extracts dialogue cues."""
    records = []
    for _, row in df.iterrows():
        raw = row["content"]
        if clean_subtitle_text(raw):
            records.append((str(row["num"]), str(row["name"]), raw))
    return records


def ingest_semantic_batches(
    db_path: Path,
    chroma_path: Path,
    model_name: str,
    chunk_size: int,
    overlap: int,
    sample_fraction: float,
    batch_size: int,
    batch_embed: int = 8,
    device: str = "cpu",
    chroma_write_batch: int = 100,
) -> int:
    client = get_client(chroma_path)
    coll_chunks, coll_names = get_or_create_collections(client)

    total_chunks = 0
    filename_ids: list[str] = []
    filename_names: list[str] = []
    seen_ids: set[str] = set()

    batch_iter = iter_subtitle_batches(
        db_path, batch_size=batch_size, sample_fraction=sample_fraction
    )

    for df in tqdm(batch_iter, desc="Ingest semantic"):
        records = prepare_records(df)
        del df
        if not records:
            continue

        chunks = chunk_corpus(records, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            continue

        texts = [c.text for c in chunks]
        embeddings = encode_texts(
            texts,
            model_name,
            batch_size=batch_embed,
            show_progress=False,
            device=device,
        )

        id_to_name = {sid: name for sid, name, _ in records}
        metadatas = [
            {
                "subtitle_id": c.subtitle_id,
                "chunk_index": c.chunk_index,
                "filename": id_to_name.get(c.subtitle_id, ""),
            }
            for c in chunks
        ]
        add_chunks_batch(
            coll_chunks,
            chunks,
            embeddings.tolist(),
            metadatas=metadatas,
            batch_size=chroma_write_batch,
        )

        for sid, name, _ in records:
            if sid not in seen_ids:
                seen_ids.add(sid)
                filename_ids.append(sid)
                filename_names.append(name)

        total_chunks += len(chunks)
        logger.info("Batch done: +%d chunks (running total=%d)", len(chunks), total_chunks)
        del chunks, texts, embeddings, records
        gc.collect()

    if filename_ids:
        upsert_filename_map_batched(coll_names, filename_ids, filename_names)

    return total_chunks


def ingest_keyword_batches(
    db_path: Path,
    artifacts_dir: Path,
    chunk_size: int,
    overlap: int,
    sample_fraction: float,
    batch_size: int,
) -> int:
    all_chunk_ids: list[str] = []
    all_texts: list[str] = []
    meta_rows: list[dict] = []

    for df in tqdm(
        iter_subtitle_batches(db_path, batch_size=batch_size, sample_fraction=sample_fraction),
        desc="Ingest keyword",
    ):
        records = prepare_records(df)
        chunks = chunk_corpus(records, chunk_size=chunk_size, overlap=overlap)
        id_to_name = {sid: name for sid, name, _ in records}
        for c in chunks:
            all_chunk_ids.append(c.chunk_id)
            all_texts.append(c.text)
            meta_rows.append(
                {
                    "chunk_id": c.chunk_id,
                    "subtitle_id": c.subtitle_id,
                    "filename": id_to_name.get(c.subtitle_id, ""),
                    "text": c.text,
                }
            )
        del df, records, chunks
        gc.collect()

    if not all_texts:
        return 0

    meta = pd.DataFrame(meta_rows)
    print(f"Fitting TF-IDF on {len(all_texts)} chunks...")
    engine = KeywordSearchEngine()
    engine.fit(all_chunk_ids, all_texts, meta)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    engine.save(
        artifacts_dir / "tfidf_vectorizer.joblib",
        artifacts_dir / "tfidf_matrix.npz",
        artifacts_dir / "tfidf_metadata.parquet",
    )
    return len(all_texts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest subtitle database")
    parser.add_argument("--mode", choices=["semantic", "keyword", "both"], default="both")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    parser.add_argument("--sample", type=float, default=config.SAMPLE_FRACTION)
    parser.add_argument("--batch-size", type=int, default=50, help="Subtitles per DB batch")
    parser.add_argument("--embed-batch", type=int, default=8, help="Embedding mini-batch size")
    parser.add_argument("--device", default="cpu", choices=["cpu"], help="Use cpu on Mac")
    parser.add_argument("--chroma-path", type=Path, default=config.CHROMA_PATH)
    parser.add_argument("--artifacts", type=Path, default=config.ARTIFACTS_DIR)
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE_TOKENS)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP_TOKENS)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--debug", action="store_true", help="DEBUG logging")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.debug else config.LOG_LEVEL)

    if sys.platform == "darwin":
        print("macOS: using CPU-only, single-threaded mode (avoids bus errors).")

    if args.reset:
        if args.chroma_path.exists():
            shutil.rmtree(args.chroma_path)
        if args.artifacts.exists():
            shutil.rmtree(args.artifacts)
        print("Cleared chroma_db/ and artifacts/")

    print(
        f"Ingest from {args.db} (sample={args.sample}, "
        f"batch_size={args.batch_size}, embed_batch={args.embed_batch})"
    )

    if args.mode in ("semantic", "both"):
        n = ingest_semantic_batches(
            args.db,
            args.chroma_path,
            args.model,
            args.chunk_size,
            args.overlap,
            args.sample,
            args.batch_size,
            batch_embed=args.embed_batch,
            device=args.device,
        )
        print(f"Semantic ingest complete: {n} chunks in ChromaDB.")

    if args.mode in ("keyword", "both"):
        n = ingest_keyword_batches(
            args.db,
            args.artifacts,
            args.chunk_size,
            args.overlap,
            args.sample,
            args.batch_size,
        )
        print(f"Keyword ingest complete: {n} chunks in TF-IDF index.")


if __name__ == "__main__":
    main()
