"""Transcribe audio queries with Whisper (Shazam-style dialogue search)."""
from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

from src.cleaning import clean_query


@lru_cache(maxsize=1)
def _load_whisper(model_size: str):
    import whisper

    return whisper.load_model(model_size)


def _check_ffmpeg() -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg is required for audio search. Install: brew install ffmpeg"
        ) from e


def transcribe_audio(
    audio_path: Path,
    model_size: str = "base",
    language: str | None = "en",
) -> str:
    """
    Convert a TV/movie audio clip (~2 min) to text for semantic search.

    The clip should be from content in eng_subtitles_database.db.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    _check_ffmpeg()
    model = _load_whisper(model_size)
    result = model.transcribe(
        str(audio_path),
        language=language,
        fp16=False,
        verbose=False,
    )
    raw = (result.get("text") or "").strip()
    if not raw:
        raise ValueError("Whisper returned empty transcript. Try a longer or clearer clip.")
    return clean_query(raw)


def save_audio_bytes(data: bytes, suffix: str = ".wav") -> Path:
    """Write uploaded or recorded audio to a temp file for Whisper."""
    import tempfile

    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    fd, name = tempfile.mkstemp(suffix=suffix)
    path = Path(name)
    with open(fd, "wb") as f:
        f.write(data)
    return path
