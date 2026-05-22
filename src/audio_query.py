"""Transcribe audio queries with Whisper (Shazam-style dialogue search)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.cleaning import clean_query


@lru_cache(maxsize=1)
def _load_whisper(model_size: str):
    import whisper

    return whisper.load_model(model_size)


def transcribe_audio(
    audio_path: Path,
    model_size: str = "base",
    language: str | None = "en",
) -> str:
    """
    Convert ~2 minute TV/movie audio clip to text for semantic/keyword search.

    Use a clip from content that exists in eng_subtitles_database.db.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _load_whisper(model_size)
    result = model.transcribe(
        str(audio_path),
        language=language,
        fp16=False,
        verbose=False,
    )
    raw = (result.get("text") or "").strip()
    return clean_query(raw)
