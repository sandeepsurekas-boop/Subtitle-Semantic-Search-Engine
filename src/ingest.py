"""Part 1: Ingest subtitle database → clean → chunk → embed → ChromaDB / TF-IDF."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from src.chunker import TextChunk, chunk_corpus
from src.cleaning import clean_subtitle_text
from src.database import load_subtitles
from src.embeddings import encode_texts
from src.chroma_store import (
    add_chunks_batch,
    get_client,
    get_or_create_collections,
    upsert_filename_map,
)
from src.keyword_search import KeywordSearchEngine


def prepare_records(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    records = []
    for _, row in df.iterrows():
        cleaned = clean_subtitle_text(row["content"])
        if cleaned:
            records.append((str(row["num"]), str(row["name"]), cleaned))
    return records


def ingest_semantic(
    df: pd.DataFrame,
    chroma_path: Path,
    model_name: str,
    chunk_size: int,
    overlap: int,
    batch_embed: int = 64,
) -> int:
    records = prepare_records(df)
    chunks = chunk_corpus(records, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        print("No chunks produced.")
        return 0

    print(f"Encoding {len(chunks)} chunks with {model_name}...")
    texts = [c.text for c in chunks]
    embeddings = encode_texts(
        texts, model_name, batch_size=batch_embed, show_progress=True
    )

    client = get_client(chroma_path)
    coll_chunks, coll_names = get_or_create_collections(client)

    metadatas = [
        {
            "subtitle_id": c.subtitle_id,
            "chunk_index": c.chunk_index,
            "filename": next(
                (n for sid, n, _ in records if sid == c.subtitle_id), ""
            ),
        }
        for c in chunks
    ]

    print("Writing to ChromaDB...")
    add_chunks_batch(
        coll_chunks,
        chunks,
        embeddings.tolist(),
        metadatas=metadatas,
    )

    unique = df.drop_duplicates("num")
    upsert_filename_map(
        coll_names,
        unique["num"].astype(str).tolist(),
        unique["name"].astype(str).tolist(),
    )
    return len(chunks)


def ingest_keyword(
    df: pd.DataFrame,
    artifacts_dir: Path,
    chunk_size: int,
    overlap: int,
) -> int:
    records = prepare_records(df)
    chunks = chunk_corpus(records, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return 0

    id_to_name = {sid: name for sid, name, _ in records}
    meta = pd.DataFrame(
        {
            "chunk_id": [c.chunk_id for c in chunks],
            "subtitle_id": [c.subtitle_id for c in chunks],
            "filename": [id_to_name.get(c.subtitle_id, "") for c in chunks],
            "text": [c.text for c in chunks],
        }
    )

    print(f"Fitting TF-IDF on {len(chunks)} chunks...")
    engine = KeywordSearchEngine()
    engine.fit(
        meta["chunk_id"].tolist(),
        meta["text"].tolist(),
        meta,
    )
    engine.save(
        artifacts_dir / "tfidf_vectorizer.joblib",
        artifacts_dir / "tfidf_matrix.npz",
        artifacts_dir / "tfidf_metadata.parquet",
    )
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest subtitle database")
    parser.add_argument(
        "--mode",
        choices=["semantic", "keyword", "both"],
        default="both",
        help="semantic=Chroma+SentenceTransformers, keyword=TF-IDF",
    )
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    parser.add_argument("--sample", type=float, default=config.SAMPLE_FRACTION)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--chroma-path", type=Path, default=config.CHROMA_PATH)
    parser.add_argument("--artifacts", type=Path, default=config.ARTIFACTS_DIR)
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE_TOKENS)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP_TOKENS)
    args = parser.parse_args()

    print(f"Loading subtitles from {args.db} (sample={args.sample})...")
    df = load_subtitles(args.db, sample_fraction=args.sample, random_seed=args.seed)
    print(f"Loaded {len(df)} subtitle files.")

    if args.mode in ("semantic", "both"):
        n = ingest_semantic(
            df,
            args.chroma_path,
            args.model,
            args.chunk_size,
            args.overlap,
        )
        print(f"Semantic ingest complete: {n} chunks indexed.")

    if args.mode in ("keyword", "both"):
        n = ingest_keyword(df, args.artifacts, args.chunk_size, args.overlap)
        print(f"Keyword ingest complete: {n} chunks in TF-IDF index.")


if __name__ == "__main__":
    main()
