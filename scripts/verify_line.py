#!/usr/bin/env python3
"""
Check if a dialogue line exists in the DB and whether search finds it.

Example:
  python scripts/verify_line.py --query "may the force be with you"
  python scripts/verify_line.py --query "i am your father" --subtitle-id 123456
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.cleaning import clean_subtitle_text, extract_dialogue_cues  # noqa: E402
from src.database import decompress_subtitle  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402
from src.retrieve import SubtitleSearchEngine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--subtitle-id", help="OpenSubtitles num to check in DB")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.debug else "INFO")
    q = args.query.lower()

    print("\n=== 1) Database text search ===")
    conn = sqlite3.connect(config.DB_PATH)
    cur = conn.execute("SELECT num, name, content FROM zipfiles")
    found_in_db = []
    for num, name, blob in cur:
        raw = decompress_subtitle(blob)
        cues = extract_dialogue_cues(raw)
        blob_l = " ".join(c.lower() for c in cues)
        if q in blob_l or q in clean_subtitle_text(raw):
            found_in_db.append((num, name))
            if len(found_in_db) <= 5:
                print(f"  FOUND in DB: {num} | {name}")
    conn.close()
    print(f"  Total DB files containing phrase: {len(found_in_db)}")
    if not found_in_db:
        print("  ⚠ Line not in full database — search cannot find it.")

    if args.subtitle_id:
        in_list = any(str(n) == str(args.subtitle_id) for n, _ in found_in_db)
        print(f"  Subtitle {args.subtitle_id} in DB with phrase: {in_list}")

    print("\n=== 2) Vector index search ===")
    try:
        engine = SubtitleSearchEngine.create_if_ready("semantic")
        results = engine.search_semantic(args.query, top_k=5)
        if not results:
            print("  No results — re-ingest with: python -m src.ingest --mode semantic --reset")
        for i, r in enumerate(results, 1):
            mark = "✓" if q in r.snippet.lower() else " "
            print(f"  {mark} {i}. {r.filename} sim={r.score:.4f} id={r.subtitle_id}")
            print(f"      {r.snippet[:120]}...")
    except Exception as e:
        print(f"  Error: {e}")
        return 1

    print("\nDone. Re-ingest required after chunking fixes: rm -rf chroma_db && python -m src.ingest ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
