"""
Movie Subtitle Search — simple UI for text and voice search.
"""
from __future__ import annotations

import hashlib

import streamlit as st

import importlib

import config
from src.logging_config import setup_logging
from src.audio_query import save_audio_bytes, transcribe_audio

setup_logging(config.LOG_LEVEL)

# Always use latest retrieve module (avoids stale SearchResult class in long-lived Streamlit)
import src.retrieve as _retrieve_mod

importlib.reload(_retrieve_mod)
from src.chroma_store import get_index_status
from src.retrieve import IndexNotReadyError, SubtitleSearchEngine
from src.titles import display_movie_name

st.set_page_config(
    page_title="Find Movie Lines",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding-top: 2rem; max-width: 42rem; }
        .hero { text-align: center; margin-bottom: 1.5rem; }
        .hero h1 { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.25rem; }
        .hero p { color: #5c6370; font-size: 1.05rem; line-height: 1.5; margin: 0; }
        .step {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 0.85rem;
        }
        .step-title { font-size: 1.05rem; font-weight: 600; color: #0f172a; margin-bottom: 0.25rem; }
        .step-hint { font-size: 0.9rem; color: #64748b; margin: 0; line-height: 1.45; }
        .result-card {
            background: #fff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.6rem;
        }
        .match-badge {
            display: inline-block;
            background: #dbeafe;
            color: #1d4ed8;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            margin-bottom: 0.35rem;
        }
        .result-card a { color: #1d4ed8; text-decoration: none; font-weight: 600; }
        .result-card a:hover { text-decoration: underline; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🎬 Find Movie & TV Lines</h1>
        <p>Type a quote you remember, or play a clip from a show —
        we’ll find which film or series it came from.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Session state
_defaults = {
    "audio_bytes": None,
    "audio_suffix": ".wav",
    "audio_format": "audio/wav",
    "transcript": None,
    "record_key": 0,
    "audio_hash": None,
    "search_mode": "semantic",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

_FORMAT = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
}


def _start_over() -> None:
    st.session_state.audio_bytes = None
    st.session_state.transcript = None
    st.session_state.audio_hash = None
    st.session_state.record_key += 1


def _audio_fingerprint(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _store_audio(live, uploaded) -> None:
    data, suffix = None, ".wav"
    if uploaded is not None:
        data = uploaded.getvalue()
        if uploaded.name and "." in uploaded.name:
            suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower()
    elif live is not None:
        data = live.getvalue() if hasattr(live, "getvalue") else live.read()
    if data and data != st.session_state.audio_bytes:
        st.session_state.audio_bytes = data
        st.session_state.audio_suffix = suffix
        st.session_state.audio_format = _FORMAT.get(suffix, "audio/wav")
        st.session_state.transcript = None
        st.session_state.audio_hash = None


def _hint_step(title: str, hint: str) -> None:
    st.markdown(
        f'<div class="step"><div class="step-title">{title}</div>'
        f'<p class="step-hint">{hint}</p></div>',
        unsafe_allow_html=True,
    )


def _show_results(results) -> None:
    if not results:
        st.info("No close matches. Try a movie name, a quote, or the other search style.")
        return
    st.markdown("### 🎯 Best matches")
    for i, r in enumerate(results, 1):
        pct = max(0, min(100, int(r.score * 100)))
        movie = display_movie_name(r.filename, r.subtitle_id)
        match_kind = getattr(r, "match_type", "dialogue")
        badge = "Name match" if match_kind == "title" else f"{pct}% match"
        st.markdown(
            f'<div class="result-card">'
            f'<span class="match-badge">{badge}</span><br>'
            f'<strong>{i}. <a href="{r.opensubtitles_url}" target="_blank" '
            f'rel="noopener">{movie}</a></strong></div>',
            unsafe_allow_html=True,
        )
        if match_kind != "title" and r.snippet:
            st.caption(f"“{r.snippet[:280]}…”" if len(r.snippet) > 280 else f"“{r.snippet}”")


def _run_search(query: str, mode: str, engine: SubtitleSearchEngine):
    if hasattr(engine, "search_combined"):
        return engine.search_combined(query, mode=mode, top_k=config.DEFAULT_TOP_K)
    # Fallback if an old cached engine is still in memory
    title = getattr(engine, "search_by_title", lambda q, top_k=10: [])(query, config.DEFAULT_TOP_K)
    if mode == "keyword":
        dialogue = engine.search_keyword(query, config.DEFAULT_TOP_K)
    else:
        dialogue = engine.search_semantic(query, config.DEFAULT_TOP_K)
    seen = {h.subtitle_id for h in title}
    merged = list(title)
    for h in dialogue:
        if h.subtitle_id not in seen:
            merged.append(h)
        if len(merged) >= config.DEFAULT_TOP_K:
            break
    return merged[: config.DEFAULT_TOP_K]


def _auto_transcribe() -> None:
    """Transcribe as soon as audio is available — no extra button."""
    data = st.session_state.audio_bytes
    if not data:
        return
    fp = _audio_fingerprint(data)
    if st.session_state.audio_hash == fp and st.session_state.transcript is not None:
        return
    if st.session_state.audio_hash == fp:
        return  # already tried and failed

    path = save_audio_bytes(data, suffix=st.session_state.audio_suffix)
    try:
        with st.spinner("Listening and writing what we heard…"):
            st.session_state.transcript = transcribe_audio(
                path, model_size=config.WHISPER_MODEL
            )
        st.session_state.audio_hash = fp
    except RuntimeError as err:
        st.error(str(err))
        st.session_state.audio_hash = fp
    except ValueError as err:
        st.warning(str(err))
        st.session_state.audio_hash = fp


# ── Setup checks ─────────────────────────────────────────────────────────────
status = get_index_status()
film_count = status.get("indexed_subtitles", 0)

if not status["database"]:
    st.error("**Setup needed:** the subtitle database file is missing.")
    with st.expander("How to fix (for your tech person)"):
        st.code("Place eng_subtitles_database.db in the data/ folder", language="text")
    st.stop()

if not status["semantic_index"]:
    st.warning("**Almost ready** — subtitles still need to be indexed once.")
    with st.expander("How to fix (for your tech person)"):
        st.code("python -m src.ingest --mode semantic --sample 0.05 --reset", language="bash")
    st.stop()

# ── Search style (tooltips on hover only) ────────────────────────────────────
st.markdown("**How should we search?**")
c_meaning, c_exact = st.columns(2)

with c_meaning:
    if st.button(
        "By meaning",
        use_container_width=True,
        type="primary" if st.session_state.search_mode == "semantic" else "secondary",
        help="Similar ideas — finds lines that mean something similar, even with different words.",
        key="btn_meaning",
    ):
        st.session_state.search_mode = "semantic"
        st.rerun()

with c_exact:
    keyword_ok = status.get("keyword_index", False)
    if st.button(
        "By exact words",
        use_container_width=True,
        type="primary" if st.session_state.search_mode == "keyword" else "secondary",
        help="Same words in the subtitle — best when you know the exact phrase.",
        key="btn_exact",
        disabled=not keyword_ok,
    ):
        st.session_state.search_mode = "keyword"
        st.rerun()

mode = st.session_state.search_mode
if mode == "keyword" and not status.get("keyword_index"):
    st.session_state.search_mode = "semantic"
    mode = "semantic"


# Do not cache the engine object — Streamlit cache can keep an old class instance
# missing methods after code updates (e.g. search_combined).
def _get_engine(m: str) -> SubtitleSearchEngine:
    return SubtitleSearchEngine.create_if_ready(mode=m)


try:
    engine = _get_engine(mode)
except IndexNotReadyError:
    st.error("Search isn’t ready yet. Ask whoever installed the app to run the index step.")
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_words, tab_voice = st.tabs(["✏️ Type a quote", "🎤 Use your voice"])

with tab_words:
    _hint_step(
        "Search by quote or movie name",
        "Type a line you remember, or a film/show name like Avatar or Broker.",
    )
    with st.form("text_search_form", clear_on_submit=False):
        quote = st.text_input(
            "quote",
            placeholder='e.g. "Avatar", "I\'ll be back", or "winter is coming"',
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "Find film or show",
            type="primary",
            use_container_width=True,
        )
    if submitted:
        if not quote.strip():
            st.warning("Please type a movie name or a line first.")
        else:
            with st.spinner("Looking through films and shows…"):
                _show_results(_run_search(quote.strip(), mode, engine))

with tab_voice:
    _hint_step(
        "Heard it on TV?",
        "Record or upload a short clip. We’ll write what we hear, then search — "
        "using the same style you picked above.",
    )

    mic = None
    if hasattr(st, "audio_input"):
        mic = st.audio_input("Tap to record", key=f"mic_{st.session_state.record_key}")
    else:
        st.info("To record from your mic: `pip install 'streamlit>=1.46'`")

    upload = st.file_uploader(
        "Or choose an audio file",
        type=["wav", "mp3", "m4a", "ogg", "webm"],
        key=f"up_{st.session_state.record_key}",
    )
    _store_audio(mic, upload)

    if not st.session_state.audio_bytes:
        st.caption("👆 Record or upload a clip first (up to ~2 minutes works best).")
    else:
        st.markdown("**Your recording**")
        st.audio(st.session_state.audio_bytes, format=st.session_state.audio_format)

        _auto_transcribe()

        if st.session_state.transcript is not None:
            st.markdown("**What we heard** — edit if needed, then press Enter or tap Find:")

            submitted_audio = False
            redo_clicked = False
            with st.form("audio_search_form", clear_on_submit=False):
                st.session_state.transcript = st.text_area(
                    "transcript_edit",
                    value=st.session_state.transcript,
                    height=100,
                    label_visibility="collapsed",
                )
                find_col, redo_col = st.columns([2, 1])
                with find_col:
                    submitted_audio = st.form_submit_button(
                        "Find film or show",
                        type="primary",
                        use_container_width=True,
                    )
                with redo_col:
                    redo_clicked = st.form_submit_button(
                        "Start over",
                        use_container_width=True,
                    )

            if redo_clicked:
                _start_over()
                st.rerun()
            elif submitted_audio:
                text = (st.session_state.transcript or "").strip()
                if not text:
                    st.warning("We couldn’t hear any words — try **Start over**.")
                else:
                    with st.spinner("Looking through films and shows…"):
                        _show_results(_run_search(text, mode, engine))
        elif st.session_state.audio_hash:
            st.caption("Couldn’t make out speech. Try a clearer clip or **Start over**.")
        else:
            st.caption("Writing what we heard from your clip…")

# Footer: films/shows count
if film_count > 0:
    st.caption(
        f"Searching across **{film_count:,}** films & shows · Links open OpenSubtitles.org"
    )
else:
    st.caption("Links open OpenSubtitles.org")
