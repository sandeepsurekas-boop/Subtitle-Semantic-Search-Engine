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
    """
    Extract first file from ZIP blob and decode as latin-1.

    Matches course notebook `decode_method` in project_hint_reading_the_data.ipynb.
    """
    try:
        data = _to_bytes(raw)
        with io.BytesIO(data) as f:
            with zipfile.ZipFile(f, "r") as zip_file:
                names = zip_file.namelist()
                if not names:
                    return ""
                subtitle_content = zip_file.read(names[0])
        return subtitle_content.decode("latin-1", errors="replace")
    except Exception:
        return ""


def _decode_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["content"] = out["content"].map(decompress_subtitle)
    out["num"] = out["num"].astype(str)
    return out


def iter_subtitle_batches(
    db_path: Path,
    batch_size: int = 50,
    sample_fraction: float = 1.0,
    random_seed: int = 42,
) -> Iterator[pd.DataFrame]:
    """
    Yield subtitle batches; decode ZIP per batch to limit RAM use.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            "Place eng_subtitles_database.db in data/ — see data/README.txt"
        )

    conn = sqlite3.connect(db_path)
    try:
        if 0 < sample_fraction < 1.0:
            total = conn.execute("SELECT COUNT(*) FROM zipfiles").fetchone()[0]
            limit = max(1, int(total * sample_fraction))
            # Fixed sample: pick IDs once, then fetch/decode in batches
            id_df = pd.read_sql_query(
                f"""
                SELECT num FROM zipfiles
                ORDER BY RANDOM()
                LIMIT {limit}
                """,
                conn,
            )
            nums = id_df["num"].tolist()
            for start in range(0, len(nums), batch_size):
                batch_nums = nums[start : start + batch_size]
                placeholders = ",".join("?" * len(batch_nums))
                chunk = pd.read_sql_query(
                    f"SELECT num, name, content FROM zipfiles WHERE num IN ({placeholders})",
                    conn,
                    params=batch_nums,
                )
                if not chunk.empty:
                    yield _decode_frame(chunk)
            return

        offset = 0
        while True:
            chunk = pd.read_sql_query(
                """
                SELECT num, name, content FROM zipfiles
                ORDER BY num
                LIMIT ? OFFSET ?
                """,
                conn,
                params=(batch_size, offset),
            )
            if chunk.empty:
                break
            yield _decode_frame(chunk)
            offset += batch_size
    finally:
        conn.close()


def load_subtitles(
    db_path: Path,
    sample_fraction: float = 1.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load subtitles into one DataFrame — only for small samples."""
    parts = list(
        iter_subtitle_batches(
            db_path,
            batch_size=100,
            sample_fraction=sample_fraction,
            random_seed=random_seed,
        )
    )
    if not parts:
        return pd.DataFrame(columns=["num", "name", "content"])
    return pd.concat(parts, ignore_index=True)
