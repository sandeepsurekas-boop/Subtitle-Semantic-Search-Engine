"""Document chunker — by subtitle cues first, then token windows with overlap."""
from __future__ import annotations

from dataclasses import dataclass

import tiktoken

import config
from src.cleaning import extract_dialogue_cues


@dataclass
class TextChunk:
    chunk_id: str
    subtitle_id: str
    chunk_index: int
    text: str


def _get_encoder(encoding_name: str = "cl100k_base"):
    try:
        return tiktoken.encoding_for_model(encoding_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _pack_cues_into_token_chunks(
    cues: list[str],
    subtitle_id: str,
    chunk_size: int,
    overlap: int,
    lowercase: bool,
) -> list[TextChunk]:
    """Group short cue lines into token-bounded chunks (dialogue-aware)."""
    if not cues:
        return []

    enc = _get_encoder()
    chunks: list[TextChunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    chunk_idx = 0
    stride = max(1, chunk_size - overlap)

    def flush():
        nonlocal buffer, buffer_tokens, chunk_idx
        if not buffer:
            return
        text = " ".join(buffer).strip()
        if lowercase:
            text = text.lower()
        if text:
            chunks.append(
                TextChunk(
                    chunk_id=f"{subtitle_id}-{chunk_idx}",
                    subtitle_id=subtitle_id,
                    chunk_index=chunk_idx,
                    text=text,
                )
            )
            chunk_idx += 1
        buffer = []
        buffer_tokens = 0

    for cue in cues:
        cue_text = cue.lower() if lowercase else cue
        cue_tokens = len(enc.encode(cue_text))
        if cue_tokens > chunk_size:
            flush()
            toks = enc.encode(cue_text)
            stride = max(1, chunk_size - overlap)
            start = 0
            while start < len(toks):
                piece = enc.decode(toks[start : start + chunk_size]).strip()
                if piece:
                    chunks.append(
                        TextChunk(
                            chunk_id=f"{subtitle_id}-{chunk_idx}",
                            subtitle_id=subtitle_id,
                            chunk_index=chunk_idx,
                            text=piece,
                        )
                    )
                    chunk_idx += 1
                if start + chunk_size >= len(toks):
                    break
                start += stride
            continue

        if buffer_tokens + cue_tokens > chunk_size and buffer:
            flush()
        buffer.append(cue_text)
        buffer_tokens += cue_tokens

    flush()
    return chunks


def chunk_document(
    text: str,
    subtitle_id: str,
    chunk_size: int = 150,
    overlap: int = 30,
    lowercase: bool = True,
    by_cues: bool = True,
) -> list[TextChunk]:
    if not text or not text.strip():
        return []

    if by_cues:
        cues = extract_dialogue_cues(text)
        if cues:
            packed = _pack_cues_into_token_chunks(
                cues, subtitle_id, chunk_size, overlap, lowercase
            )
            if packed:
                return packed

    enc = _get_encoder()
    if lowercase:
        text = text.lower()
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
    chunk_size: int | None = None,
    overlap: int | None = None,
    by_cues: bool = True,
) -> list[TextChunk]:
    chunk_size = chunk_size or config.CHUNK_SIZE_TOKENS
    overlap = overlap or config.CHUNK_OVERLAP_TOKENS
    all_chunks: list[TextChunk] = []
    for sid, _name, raw_text in records:
        all_chunks.extend(
            chunk_document(
                raw_text,
                sid,
                chunk_size=chunk_size,
                overlap=overlap,
                by_cues=by_cues,
            )
        )
    return all_chunks
