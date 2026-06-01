"""Optional REST API (same search logic as Streamlit)."""
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from src.chroma_store import get_index_status
from src.logging_config import setup_logging
from src.retrieve import IndexNotReadyError, SearchResult, SubtitleSearchEngine

setup_logging(config.LOG_LEVEL)

app = FastAPI(title="Subtitle Search API", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_engine: SubtitleSearchEngine | None = None


def get_engine(mode: str = "semantic") -> SubtitleSearchEngine:
    global _engine
    if _engine is None or mode == "keyword":
        _engine = SubtitleSearchEngine.create_if_ready(mode=mode)
    return _engine


class TextSearchRequest(BaseModel):
    query: str
    mode: Literal["semantic", "keyword"] = "semantic"
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=50)


class HitResponse(BaseModel):
    subtitle_id: str
    filename: str
    score: float
    snippet: str
    opensubtitles_url: str
    chunk_id: str = ""


@app.get("/health")
def health():
    return get_index_status()


@app.post("/search/text")
def search_text(body: TextSearchRequest):
    try:
        engine = get_engine(body.mode)
        if body.mode == "keyword":
            hits = engine.search_keyword(body.query, body.top_k)
        else:
            hits = engine.search_semantic(body.query, body.top_k)
    except IndexNotReadyError as e:
        raise HTTPException(503, str(e)) from e
    return [_hit(h) for h in hits]


@app.post("/search/audio")
async def search_audio(
    file: UploadFile = File(...),
    top_k: int = Form(config.DEFAULT_TOP_K),
):
    dest = config.AUDIO_QUERY_DIR / f"upload_{file.filename or 'clip.wav'}"
    config.AUDIO_QUERY_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())
    try:
        engine = get_engine("semantic")
        transcript, hits = engine.search_audio(dest, top_k=top_k)
    except IndexNotReadyError as e:
        raise HTTPException(503, str(e)) from e
    return {"transcript": transcript, "results": [_hit(h) for h in hits]}


def _hit(r: SearchResult) -> HitResponse:
    return HitResponse(
        subtitle_id=r.subtitle_id,
        filename=r.filename,
        score=r.score,
        snippet=r.snippet,
        opensubtitles_url=r.opensubtitles_url,
        chunk_id=r.chunk_id,
    )
