# Subtitle Semantic Search Engine (Shazam for Subtitles)

Enhance search relevance for **video subtitles** using NLP and vector retrieval. Compare **keyword (TF-IDF)** vs **semantic (SentenceTransformers)** search, with **audio queries** transcribed via Whisper.

## What this demonstrates


| Concept         | Implementation                               |
| --------------- | -------------------------------------------- |
| Document ingest | SQLite `zipfiles` → ZIP decode → clean       |
| Chunking        | Overlapping **500-token** windows (tiktoken) |
| Keyword search  | TF-IDF + cosine similarity                   |
| Semantic search | `paraphrase-MiniLM-L6-v2` embeddings         |
| Vector DB       | **ChromaDB** (cosine HNSW)                   |
| Audio → text    | **OpenAI Whisper**                           |
| API             | **FastAPI** + **Streamlit** UI               |


## Project layout

```
subtitle-semantic-search/
├── config.py              # Paths & hyperparameters
├── data/
│   ├── README.txt         # Database schema
│   └── eng_subtitles_database.db   
├── src/
│   ├── database.py        # Part 1: read & decode DB
│   ├── cleaning.py        # Remove timestamps, ads, noise
│   ├── chunker.py         # Overlapping token chunks (required)
│   ├── embeddings.py      # SentenceTransformers
│   ├── chroma_store.py    # ChromaDB ingest/query
│   ├── keyword_search.py  # TF-IDF index
│   ├── ingest.py          # Full ingest pipeline
│   ├── retrieve.py        # Text + audio search
│   └── audio_query.py     # Whisper transcription
├── app/main.py            # FastAPI
├── streamlit_app.py       # Web UI
├── scripts/download_data.py
└── requirements.txt
```

## Quick start

### 1. Environment

```bash
cd subtitle-semantic-search
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download data

Place `**eng_subtitles_database.db**` in `data/` (see `data/README.txt`).

```bash
python scripts/download_data.py
python scripts/download_data.py --check
```

**Limited compute?** Start with 5% sample:

```bash
python -m src.ingest --mode semantic --sample 0.05 --reset
```

Install **ffmpeg** for audio search: `brew install ffmpeg`

### 3. Ingest (Part 1)

Build indexes (batched — stable on large DB):

```bash
python -m src.ingest --mode semantic --sample 0.05 --reset
python -m src.ingest --mode both --sample 0.1
```

Or separately:

```bash
python -m src.ingest --mode semantic
python -m src.ingest --mode keyword
```

### 4. Search (Part 2)

**Text query** (semantic):

```bash
python -m src.retrieve --mode semantic --query "i am your father"
```

**Keyword search:**

```bash
python -m src.retrieve --mode keyword --query "winter is coming"
```

**Audio query** (~2 min TV/movie clip from the database):

```bash
python -m src.retrieve --mode semantic --audio queries/audio/your_clip.wav
```

### 5. Run API / UI

```bash
uvicorn app.main:app --reload --port 8000
```

```bash
streamlit run streamlit_app.py
```

**Live audio search:** open Streamlit → **Audio search** → use **Record live audio** (~2 min clip) → **Search audio**.

## Core algorithm

1. **Preprocess** — decode ZIP subtitles, remove SRT timestamps and OpenSubtitles boilerplate.
2. **Chunk** — split long subtitles into 500-token windows with 50-token overlap.
3. **Vectorize**
  - Keyword: TF-IDF sparse vectors
  - Semantic: SentenceTransformer dense embeddings (L2-normalized)
4. **Store** — ChromaDB (`cosine` space) for semantic chunks; joblib/npz for TF-IDF.
5. **Query** — embed (or TF-IDF transform) the query; rank by **cosine similarity**.
6. **Audio** — Whisper transcript → same pipeline as text.

## Configuration

Edit `config.py` or `.env` (from `.env.example`):


| Setting                | Default                      |
| ---------------------- | ---------------------------- |
| `SAMPLE_FRACTION`      | `1.0` (use `0.3` on low RAM) |
| `CHUNK_SIZE_TOKENS`    | `500`                        |
| `CHUNK_OVERLAP_TOKENS` | `50`                         |
| `EMBEDDING_MODEL`      | `paraphrase-MiniLM-L6-v2`    |
| `WHISPER_MODEL`        | `base`                       |


## API examples

```bash
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "may the force be with you", "mode": "semantic", "top_k": 5}'
```

```bash
curl -X POST http://localhost:8000/search/audio \
  -F "file=@queries/audio/clip.wav" \
  -F "mode=semantic"
```

## Resume / portfolio bullets

- Built a **semantic retrieval pipeline** over 80k+ subtitle documents with **chunking**, **ChromaDB**, and **cosine similarity**.
- Implemented **dual search**: sparse **TF-IDF** (keyword) vs dense **SentenceTransformer** (semantic).
- Added **Shazam-style audio search** using **Whisper** → embedding → top-k subtitle match with OpenSubtitles deep links.

## License & data

Subtitle data is from [OpenSubtitles.org](https://www.opensubtitles.org). Use in accordance with their terms and your course guidelines.