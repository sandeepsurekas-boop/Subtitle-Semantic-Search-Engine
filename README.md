# Subtitle Semantic Search Engine

A **Shazam-style** search tool for movie and TV subtitles. Type a quote you remember, or play/record a short audio clip — the system finds the closest matching subtitles from OpenSubtitles data using **vector search** and **cosine similarity**.

Built for the LearnBay / Innomatics project: *Enhancing Search Engine Relevance for Video Subtitles*.

---

## About the project

### Problem

Users often remember **a line of dialogue**, not the movie title. Classic search matches titles and metadata, not what was actually **said** in the video.

### Solution


| Stage      | What happens                                                                                                        |
| ---------- | ------------------------------------------------------------------------------------------------------------------- |
| **Ingest** | Read `eng_subtitles_database.db` → unzip subtitles → clean text → split into chunks → embed → store in **ChromaDB** |
| **Search** | Your query (text or audio→text) → same embedding model → **cosine similarity** vs all chunks → top matches          |


### Two search modes


| Mode                   | How it works                                       | Best for                     |
| ---------------------- | -------------------------------------------------- | ---------------------------- |
| **Semantic** (default) | SentenceTransformer embeddings + cosine similarity | Similar meaning, paraphrases |
| **Keyword**            | TF-IDF sparse vectors + cosine similarity          | Exact words in the subtitle  |


### Tech stack

- Python 3.11+
- SQLite + ZIP subtitle archives (course database)
- [SentenceTransformers](https://www.sbert.net/) (`paraphrase-MiniLM-L3-v2`)
- [ChromaDB](https://www.trychroma.com/) (cosine HNSW)
- [OpenAI Whisper](https://github.com/openai/whisper) (audio queries)
- [Streamlit](https://streamlit.io/) (web UI)

---

## Project structure

```
subtitle-semantic-search/
├── data/
│   ├── README.txt                      # Database schema
│   └── eng_subtitles_database.db       # YOU provide (~1.8 GB)
├── src/
│   ├── database.py                     # Read & decode ZIP subtitles
│   ├── cleaning.py                     # Remove timestamps, ads
│   ├── chunker.py                      # 500-token windows, 50 overlap
│   ├── embeddings.py                   # SentenceTransformers
│   ├── chroma_store.py                 # Vector DB
│   ├── keyword_search.py               # TF-IDF index
│   ├── ingest.py                       # Part 1 — build index
│   ├── retrieve.py                     # Part 2 — search + logging
│   ├── audio_query.py                  # Whisper transcription
│   └── logging_config.py               # Debug retrieval scores
├── scripts/
│   ├── download_data.py                # Check DB is present
│   └── check_index.py                  # Check index is built
├── streamlit_app.py                    # Main UI (text + voice)
├── config.py                           # Paths & settings
├── logs/retrieval.log                  # Created when you search (DEBUG detail)
└── requirements.txt
```

---

## Prerequisites

1. **Python 3.11+**
2. **ffmpeg** (for audio search): `brew install ffmpeg`
3. **Database file**: `data/eng_subtitles_database.db` (82,498 English subtitles from OpenSubtitles)

---

## Setup (from scratch)

```bash
cd subtitle-semantic-search

# 1. Virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Place database in data/
python scripts/download_data.py --check
# Expected: OK: ... (82,498 rows in zipfiles)

# 3. Clear old indexes (fresh start)
rm -rf chroma_db artifacts logs

# 4. Build search index (start small on Mac)
python -m src.ingest --mode semantic --sample 0.05 --reset

# 5. Verify
python scripts/check_index.py
```

### macOS ingest (if you see `bus error`)

```bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
python -m src.ingest --mode semantic --sample 0.02 --reset --batch-size 25 --embed-batch 4
```

---

## How to run

### Web app (recommended)

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

- **Type a quote** — text search  
- **Use your voice** — record → listen → convert to words → search

### Command line search

```bash
# Semantic
python -m src.retrieve --mode semantic --query "may the force be with you"

# Keyword (needs keyword ingest)
python -m src.retrieve --mode keyword --query "winter is coming"

# Audio file
python -m src.retrieve --mode semantic --audio path/to/clip.wav
```

---

## Logging & debugging retrieval

To see **exact cosine distances**, cleaned queries, and per-chunk scores:

```bash
# Console + logs/retrieval.log
export LOG_LEVEL=DEBUG
python -m src.retrieve --mode semantic --query "i am your father" --debug

# Or tail the log while using Streamlit
export LOG_LEVEL=DEBUG
streamlit run streamlit_app.py
# In another terminal:
tail -f logs/retrieval.log
```

### What the logs show

```
SEMANTIC SEARCH
  Raw query     : 'I am your father'
  Cleaned query : 'i am your father'
  #01 chunk_id=12345-2 | cosine_distance=0.412000 | cosine_similarity=0.588000 | ...
```

- **cosine_distance** — from ChromaDB (lower = closer)  
- **cosine_similarity** — we use `1 - distance` in the UI (higher = closer)  
- **Dedupe** — only one result per subtitle file (best chunk wins)

---

## Important: re-ingest after retrieval fixes

If search felt completely wrong before, **delete the old index and rebuild**:

```bash
rm -rf chroma_db artifacts
python -m src.ingest --mode both --reset --sample 0.1
```

Fixes applied: **per-cue chunking** (not whole-movie blobs), **smaller chunks** (150 tokens), **phrase reranking**, **keyword stop-word fix**.

Verify a line:

```bash
python scripts/verify_line.py --query "your exact dialogue line" --debug
```

## Optional REST API (`app/main.py`)

Removed earlier to keep the repo minimal; **restored** as an optional API (same logic as Streamlit):

```bash
uvicorn app.main:app --reload --port 8000
```

Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Why retrieval accuracy may be lower than expected

Understanding these limits helps you interpret scores and improve results.


| Reason                    | Explanation                                                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Partial index**         | If you used `--sample 0.05`, only ~5% of movies are searchable. The right film may not be in the index at all.           |
| **Old chunking (fixed)**  | Whole movies were one blob per chunk — bad for single lines. **Re-ingest** with cue-based ~150-token chunks.             |
| **Partial index**         | If only part of the DB was ingested, your movie may not be in the index at all.                                          |
| **Cleaning**              | Timestamps and ads are stripped; **lowercasing** can slightly change matching. Very short queries are ambiguous.         |
| **Small embedding model** | `paraphrase-MiniLM-L3-v2` is fast but less accurate than larger models (e.g. `all-MiniLM-L6-v2` or `all-mpnet-base-v2`). |
| **Approximate search**    | Chroma uses **HNSW** — fast but not exact nearest-neighbor; rank 1–3 can occasionally be wrong.                          |
| **Audio path**            | Whisper may mis-hear words → wrong query embedding → wrong matches.                                                      |
| **Language**              | Database is mostly **English** subtitles. Other languages match poorly.                                                  |
| **Semantic vs literal**   | Semantic search finds **similar meaning**, not always the exact film you want. Use **keyword** mode for exact phrases.   |
| **Dedupe by file**        | We return one hit per subtitle **file**; the best chunk might not be the scene you imagined.                             |


### How to improve accuracy

1. Ingest more data: `--sample 0.2` or full `--sample 1.0`
2. Use a stronger model in `config.py`: `EMBEDDING_MODEL = "all-MiniLM-L6-v2"` then re-ingest
3. Search with **longer, distinctive** phrases (not single common words)
4. For audio: clear recording, English dialogue, enable DEBUG and check the **transcript**
5. Compare modes: semantic vs keyword
6. Read `logs/retrieval.log` — if top `cosine_similarity` is below ~0.4, matches are weak

---

## Ingest options

```bash
python -m src.ingest --help

# Examples
python -m src.ingest --mode semantic --sample 0.1 --reset
python -m src.ingest --mode both --sample 0.1 --reset
python -m src.ingest --mode both --reset                    # full — hours, high RAM
python -m src.ingest --mode semantic --debug                # verbose ingest logs
```


| Flag              | Meaning                                    |
| ----------------- | ------------------------------------------ |
| `--sample 0.05`   | Use 5% of database                         |
| `--reset`         | Delete `chroma_db/` and `artifacts/` first |
| `--batch-size 50` | Subtitles per DB batch                     |
| `--embed-batch 8` | Embedding mini-batch size                  |
| `--debug`         | DEBUG logging                              |


---

## Course data hint notebook

The file `project_hint_reading_the_data.ipynb` (from your course folder) shows Steps 1–6:

1. Table `zipfiles`
2. Columns `num`, `name`, `content`
3. Load into pandas
4. `content` is ZIP binary
5. Decode with `zipfile` + `latin-1`
6. `decode_method` on all rows

Our `src/database.py` → `decompress_subtitle()` implements the same logic, in batches for large files.

---

## Troubleshooting


| Issue                                         | Fix                                                                |
| --------------------------------------------- | ------------------------------------------------------------------ |
| `Collection [subtitle_chunks] does not exist` | Run ingest first                                                   |
| `bus error` on Mac                            | Smaller `--sample`, `--embed-batch 4`, CPU-only ingest (see above) |
| Empty / weak results                          | Increase `--sample`; check DEBUG logs for similarity scores        |
| No live mic in UI                             | `pip install 'streamlit>=1.46'`                                    |
| Whisper fails                                 | Install `ffmpeg`                                                   |


---

## License & data

Subtitle data from [OpenSubtitles.org](https://www.opensubtitles.org). Use per their terms and your course guidelines.