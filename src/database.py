"""Read and decode subtitle archives from the SQLite database."""
from __future__ import annotations

import io
import sqlite3
import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd


def _to_bytes(raw) -> bytes:
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, memoryview):
        return raw.tobytes()
    return raw.encode("latin-1")


def decompress_subtitle(raw) -> str:
    """Extract first file from ZIP blob and decode as latin-1."""
    data = _to_bytes(raw)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        if not names:
            return ""
        return zf.read(names[0]).decode("latin-1", errors="replace")


def load_subtitles(
    db_path: Path,
    sample_fraction: float = 1.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Load zipfiles table; optionally sample rows for limited compute.

    Returns DataFrame with columns: num, name, content (decoded text).
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            "Download eng_subtitles_database.db into data/ — see data/README.txt"
        )

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT num, name, content FROM zipfiles", conn)
    finally:
        conn.close()

    if 0 < sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=random_seed).reset_index(
            drop=True
        )

    df["content"] = df["content"].map(decompress_subtitle)
    df["num"] = df["num"].astype(str)
    return df


def iter_subtitles(
    db_path: Path,
    batch_size: int = 500,
    sample_fraction: float = 1.0,
    random_seed: int = 42,
) -> Iterator[pd.DataFrame]:
    """Yield decoded subtitle batches without loading full DB into memory at once."""
    df = load_subtitles(db_path, sample_fraction, random_seed)
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size].copy()
