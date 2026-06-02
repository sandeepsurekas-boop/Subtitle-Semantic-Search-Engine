"""Movie / show names from OpenSubtitles filenames."""
from __future__ import annotations

import re


def display_movie_name(filename: str, subtitle_id: str = "") -> str:
    """
    Human-readable title from filename, e.g. broker.(2022).eng.1cd → Broker (2022).
    """
    if not filename:
        return f"Film #{subtitle_id}" if subtitle_id else "Unknown film"

    base = filename.lower()
    for suffix in (".eng.1cd", ".eng.2cd", ".eng.3cd", ".sub", ".srt"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    m = re.match(r"^(.+?)\.\((\d{4})\)$", base)
    if m:
        words = m.group(1).replace(".", " ").replace("_", " ").strip()
        return f"{words.title()} ({m.group(2)})"

    return base.replace(".", " ").replace("_", " ").strip().title()


def filename_matches_query(filename: str, query: str) -> tuple[bool, float]:
    """True if query matches this subtitle filename; score 0–1."""
    if not filename or not query:
        return False, 0.0

    q = query.lower().strip()
    fn = filename.lower()
    flat = fn.replace(".", " ").replace("_", " ")

    if q in fn or q in flat:
        return True, 1.0

    q_tokens = [t for t in re.split(r"\W+", q) if len(t) > 1]
    if not q_tokens:
        return False, 0.0

    if all(t in flat for t in q_tokens):
        return True, 0.92

    return False, 0.0
