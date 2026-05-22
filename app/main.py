"""FastAPI service for subtitle semantic / keyword search."""
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from src.retrieve import SearchResult, SubtitleSearchEngine

app = FastAPI(
    title="Subtitle Semantic Search",
    description="Shazam-style search over video subtitles (semantic + keyword + audio)",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: SubtitleSearchEngine | None = None


def get_engine() -> SubtitleSearchEngine:
    global _engine
    if _engine is None:
        _engine = SubtitleSearchEngine()
    return _engine


class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: Literal["semantic", "keyword"] = "semantic"
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=50)


class HitResponse(BaseModel):
    subtitle_id: str
    filename: str
    score: float
    snippet: str
    opensubtitles_url: str


class TextSearchResponse(BaseModel):
    results: list[HitResponse]


def _to_response(results: list[SearchResult]) -> list[HitResponse]:
    return [
        HitResponse(
            subtitle_id=r.subtitle_id,
            filename=r.filename,
            score=r.score,
            snippet=r.snippet,
            opensubtitles_url=r.opensubtitles_url,
        )
        for r in results
    ]


@app.get("/health")
def health():
    engine = get_engine()
    return {
        "status": "ok",
        "semantic_index": engine.chunks.count() if engine.chunks else 0,
        "keyword_index": engine.keyword is not None,
    }


@app.post("/search/text", response_model=TextSearchResponse)
def search_text(body: TextSearchRequest):
    engine = get_engine()
    try:
        if body.mode == "keyword":
            results = engine.search_keyword(body.query, body.top_k)
        else:
            results = engine.search_semantic(body.query, body.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return TextSearchResponse(results=_to_response(results))


@app.post("/search/audio", response_model=TextSearchResponse)
async def search_audio(
    file: UploadFile = File(...),
    mode: Literal["semantic", "keyword"] = Form("semantic"),
    top_k: int = Form(config.DEFAULT_TOP_K),
):
    suffix = Path(file.filename or "clip.wav").suffix or ".wav"
    dest = config.AUDIO_QUERY_DIR / f"upload{suffix}"
    config.AUDIO_QUERY_DIR.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    dest.write_bytes(content)

    engine = get_engine()
    try:
        _transcript, results = engine.search_audio(dest, mode=mode, top_k=top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return TextSearchResponse(results=_to_response(results))
