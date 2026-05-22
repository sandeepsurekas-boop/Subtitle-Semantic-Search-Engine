#!/usr/bin/env python3
"""Print instructions and optionally verify the subtitle database is present."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB = DATA / "eng_subtitles_database.db"
README = DATA / "README.txt"

INSTRUCTIONS = """
Subtitle database setup
=======================

1. Obtain `eng_subtitles_database.db` (~1–2 GB, 82,498 subtitles).
2. Copy it to: {db_path}
3. Read schema notes in: {readme}

Common sources:
  - LearnBay / course data bundle
  - Community mirrors linked from Semantic-Based-Video-Subtitle-Search-engine

Quick check after download:
  python scripts/download_data.py --check
"""


def check_db() -> bool:
    if not DB.exists():
        print(f"MISSING: {DB}")
        return False
    import sqlite3

    conn = sqlite3.connect(DB)
    try:
        cur = conn.execute("SELECT COUNT(*) FROM zipfiles")
        count = cur.fetchone()[0]
        print(f"OK: {DB} ({count:,} rows in zipfiles)")
        return True
    finally:
        conn.close()


def main():
    if "--check" in sys.argv:
        sys.exit(0 if check_db() else 1)
    print(INSTRUCTIONS.format(db_path=DB, readme=README))
    if DB.exists():
        check_db()


if __name__ == "__main__":
    main()
