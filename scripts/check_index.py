#!/usr/bin/env python3
"""Check whether database and search indexes are ready."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chroma_store import get_index_status  # noqa: E402


def main() -> int:
    status = get_index_status()
    print(json.dumps(status, indent=2))

    if not status["database"]:
        print("\n❌ Missing: data/eng_subtitles_database.db")
        return 1
    if not status["semantic_index"]:
        print("\n❌ Semantic index empty. Run: python -m src.ingest --mode semantic")
        return 1
    print("\n✅ Ready to search (semantic).")
    if status["keyword_index"]:
        print("✅ Keyword index also ready.")
    else:
        print("ℹ️  Keyword index optional: python -m src.ingest --mode keyword")
    return 0


if __name__ == "__main__":
    sys.exit(main())
