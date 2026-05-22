eng_subtitles_database.db — Schema & Usage
==========================================

This SQLite database contains English subtitle files sampled from OpenSubtitles.org.

TABLE: zipfiles
----------------
Columns:
  - num (INTEGER): Unique subtitle ID on www.opensubtitles.org
  - name (TEXT):   Subtitle filename (e.g. broker.(2022).eng.1cd)
  - content (BLOB): Subtitle stored as ZIP-compressed binary, encoded as latin-1 string in SQLite

DECODING PIPELINE
-----------------
1. Read `content` column (bytes or latin-1 string).
2. Treat as raw bytes and open with zipfile.ZipFile(io.BytesIO(data)).
3. Extract the first file in the archive.
4. Decode text with latin-1 (subtitle encodings vary; latin-1 avoids decode errors).

SUBTITLE FORMAT
---------------
Files are typically SubRip (.srt) with:
  - Index numbers
  - Timestamps: HH:MM:SS,mmm --> HH:MM:SS,mmm
  - Dialogue lines
  - OpenSubtitles promotional links

DOWNLOAD
--------
Place `eng_subtitles_database.db` in this directory (`data/`).

Sources (course / community mirrors):
  - LearnBay / Innomatics project drive (provided with assignment)
  - Reference implementation data folder:
    https://github.com/HannahIgboke/Semantic-Based-Video-Subtitle-Search-engine
  - Search: "eng_subtitles_database.db opensubtitles"

Expected size: ~1–2 GB (82,498 subtitle archives).

OPEN SUBTITLES LINK
-------------------
https://www.opensubtitles.org/en/subtitles/{num}
