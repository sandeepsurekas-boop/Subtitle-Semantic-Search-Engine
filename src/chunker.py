"""Document chunker with overlapping token windows."""
from __future__ import annotations

from dataclasses import dataclass

import tiktoken


@dataclass
class TextChunk:
    chunk_id: str
    subtitle_id: str
    chunk_index: int
    text: str


def _get_encoder(model_name: str = "cl100k_base"):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def chunk_document(
    text: str,
    subtitle_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
    encoding_name: str = "cl100k_base",
) -> list[TextChunk]:
    """
    Split a long subtitle into token windows with overlap.

    Overlap preserves context at chunk boundaries (avoids cutting dialogue
    mid-scene). Stride = chunk_size - overlap.
    """
    if not text or not text.strip():
        return []

    enc = _get_encoder(encoding_name)
    tokens = enc.encode(text)
    if not tokens:
        return []

    if len(tokens) <= chunk_size:
        return [
            TextChunk(
                chunk_id=f"{subtitle_id}-0",
                subtitle_id=subtitle_id,
                chunk_index=0,
                text=text.strip(),
            )
        ]

    stride = max(1, chunk_size - overlap)
    chunks: list[TextChunk] = []
    idx = 0
    start = 0

    while start < len(tokens):
        window = tokens[start : start + chunk_size]
        chunk_text = enc.decode(window).strip()
        if chunk_text:
            chunks.append(
                TextChunk(
                    chunk_id=f"{subtitle_id}-{idx}",
                    subtitle_id=subtitle_id,
                    chunk_index=idx,
                    text=chunk_text,
                )
            )
            idx += 1
        if start + chunk_size >= len(tokens):
            break
        start += stride

    return chunks


def chunk_corpus(
    records: list[tuple[str, str, str]],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[TextChunk]:
    """
    Chunk many subtitles.

    records: list of (subtitle_id, filename, cleaned_text)
    """
    all_chunks: list[TextChunk] = []
    for sid, _name, text in records:
        all_chunks.extend(
            chunk_document(text, sid, chunk_size=chunk_size, overlap=overlap)
        )
    return all_chunks
