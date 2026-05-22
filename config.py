"""Project configuration — paths and hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Data
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "eng_subtitles_database.db"
README_PATH = DATA_DIR / "README.txt"

# Vector store & keyword index
CHROMA_PATH = ROOT / "chroma_db"
ARTIFACTS_DIR = ROOT / "artifacts"
TFIDF_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
TFIDF_MATRIX_PATH = ARTIFACTS_DIR / "tfidf_matrix.npz"
TFIDF_META_PATH = ARTIFACTS_DIR / "tfidf_metadata.parquet"

# Chroma collection names
COLLECTION_SEMANTIC = "subtitle_chunks"
COLLECTION_FILENAMES = "subtitle_filenames"

# Embedding model (semantic search)
EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"

# Chunking (token windows with overlap — assignment requirement)
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# Ingest sampling (use 0.3 for limited compute)
SAMPLE_FRACTION = 1.0
RANDOM_SEED = 42

# Retrieval
DEFAULT_TOP_K = 10

# Audio
AUDIO_QUERY_DIR = ROOT / "queries" / "audio"
WHISPER_MODEL = "base"
