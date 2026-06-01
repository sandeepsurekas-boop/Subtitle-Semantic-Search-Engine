"""
Movie Subtitle Search — simple UI for text and voice search.
"""
from __future__ import annotations

import streamlit as st

import config
from src.logging_config import setup_logging
from src.audio_query import save_audio_bytes, transcribe_audio

setup_logging(config.LOG_LEVEL)
from src.chroma_store import get_index_status
from src.retrieve import IndexNotReadyError, SubtitleSearchEngine

# ── Page setup ──────────────────────────────────────────────────────────────
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
        .step-label {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #2563eb;
            margin-bottom: 0.35rem;
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🎬 Find Movie & TV Lines</h1>
        <p>Type a quote you remember, or hum / play a clip from a show —
        we’ll match it to subtitles from thousands of films.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Session state (voice flow) ───────────────────────────────────────────────
for key, default in (
    ("audio_bytes", None),
    ("audio_suffix", ".wav"),
    ("audio_format", "audio/wav"),
    ("transcript", None),
    ("record_key", 0),
):
    if key not in st.session_state:
        st.session_state[key] = default

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
    st.session_state.record_key += 1


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


def _step(label: str, title: str, hint: str) -> None:
    st.markdown(
        f'<div class="step"><div class="step-label">{label}</div>'
        f'<div class="step-title">{title}</div>'
        f'<p class="step-hint">{hint}</p></div>',
        unsafe_allow_html=True,
    )


def _show_results(results) -> None:
    if not results:
        st.info("No close matches. Try different words or the other search style above.")
        return
    st.markdown("### 🎯 Best matches")
    for i, r in enumerate(results, 1):
        pct = max(0, min(100, int(r.score * 100)))
        title = r.filename.replace(".", " ").replace("_", " ") if r.filename else f"Subtitle #{r.subtitle_id}"
        st.markdown(
            f'<div class="result-card">'
            f'<span class="match-badge">{pct}% match</span><br>'
            f'<strong>{i}. {title}</strong></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"“{r.snippet[:280]}…”" if len(r.snippet) > 280 else f"“{r.snippet}”")
        st.link_button("Open on OpenSubtitles ↗", r.opensubtitles_url, use_container_width=True)


# ── Ready check ─────────────────────────────────────────────────────────────
status = get_index_status()

if not status["database"]:
    st.error("**Setup needed:** the subtitle database file is missing.")
    with st.expander("How to fix (for your tech person)"):
        st.code("Place eng_subtitles_database.db in the data/ folder", language="text")
    st.stop()

if not status["semantic_index"]:
    st.warning("**Almost ready** — the app still needs to index subtitles once (this can take a while).")
    with st.expander("How to fix (for your tech person)"):
        st.code("python -m src.ingest --mode semantic --sample 0.05 --reset", language="bash")
    st.stop()

# ── How to search (one simple choice) ─────────────────────────────────────────
st.markdown("**How should we search?**")
_opts = ["By meaning", "By exact words"]
if hasattr(st, "segmented_control"):
    search_style = st.segmented_control(
        "search_style",
        options=_opts,
        default="By meaning",
        label_visibility="collapsed",
    )
else:
    search_style = st.radio(
        "search_style",
        options=_opts,
        horizontal=True,
        label_visibility="collapsed",
    )
st.caption("**By meaning** — similar ideas · **By exact words** — same words in the subtitle")
mode = "semantic" if search_style == "By meaning" else "keyword"

if mode == "keyword" and not status["keyword_index"]:
    st.warning("Exact-word search isn’t set up yet. Use **By meaning** for now.")
    mode = "semantic"


@st.cache_resource
def _engine(m: str):
    return SubtitleSearchEngine.create_if_ready(mode=m)


try:
    engine = _engine(mode)
except IndexNotReadyError:
    st.error("Search isn’t ready yet. Ask whoever installed the app to run the index step.")
    st.stop()

# ── Two ways to search ────────────────────────────────────────────────────────
tab_words, tab_voice = st.tabs(["✏️ Type a quote", "🎤 Use your voice"])

with tab_words:
    _step("Option A", "Remember the line?", "Type what you heard — even part of a sentence works.")
    quote = st.text_input(
        "quote",
        placeholder='e.g. "I\'ll be back" or "winter is coming"',
        label_visibility="collapsed",
    )
    if st.button("Find subtitles", type="primary", use_container_width=True, key="go_text"):
        if not quote.strip():
            st.warning("Please type a line first.")
        else:
            with st.spinner("Looking through subtitles…"):
                hits = (
                    engine.search_keyword(quote, config.DEFAULT_TOP_K)
                    if mode == "keyword"
                    else engine.search_semantic(quote, config.DEFAULT_TOP_K)
                )
            _show_results(hits)

with tab_voice:
    _step(
        "Option B",
        "Heard it on TV?",
        "Record up to ~2 minutes from a movie or show, listen back, turn it into words, then search.",
    )

    mic = None
    if hasattr(st, "audio_input"):
        mic = st.audio_input("Tap to record", key=f"mic_{st.session_state.record_key}")
    else:
        st.info("To record from your mic, update Streamlit: `pip install 'streamlit>=1.46'`")
    upload = st.file_uploader(
        "Or choose an audio file",
        type=["wav", "mp3", "m4a", "ogg", "webm"],
        key=f"up_{st.session_state.record_key}",
    )
    _store_audio(mic, upload)

    # Step 1 — has audio?
    if not st.session_state.audio_bytes:
        st.caption("👆 Record or upload first. Tip: hold the mic near the speaker for a clear clip.")
    else:
        # Step 2 — listen
        st.markdown(
            '<div class="step"><div class="step-label">Step 1</div>'
            '<div class="step-title">Listen to your clip</div></div>',
            unsafe_allow_html=True,
        )
        st.audio(st.session_state.audio_bytes, format=st.session_state.audio_format)

        # Step 3 — convert (one button until transcript exists)
        if st.session_state.transcript is None:
            if st.button("Turn speech into words", type="primary", use_container_width=True):
                path = save_audio_bytes(
                    st.session_state.audio_bytes,
                    suffix=st.session_state.audio_suffix,
                )
                try:
                    with st.spinner("Converting speech to text…"):
                        st.session_state.transcript = transcribe_audio(
                            path, model_size=config.WHISPER_MODEL
                        )
                    st.rerun()
                except RuntimeError as err:
                    st.error(str(err))
                except ValueError as err:
                    st.warning(str(err))
        else:
            st.markdown(
                '<div class="step"><div class="step-label">Step 2</div>'
                '<div class="step-title">Check the words</div>'
                '<p class="step-hint">Fix anything the computer got wrong, then find matching subtitles.</p></div>',
                unsafe_allow_html=True,
            )
            st.session_state.transcript = st.text_area(
                "transcript_edit",
                value=st.session_state.transcript,
                height=100,
                label_visibility="collapsed",
                placeholder="Your words will appear here…",
            )

            find_col, redo_col = st.columns([2, 1])
            with find_col:
                find_clicked = st.button(
                    "Find matching subtitles",
                    type="primary",
                    use_container_width=True,
                )
            with redo_col:
                if st.button("Start over", use_container_width=True):
                    _start_over()
                    st.rerun()

            if find_clicked:
                text = (st.session_state.transcript or "").strip()
                if not text:
                    st.warning("Nothing to search — try **Start over** and record again.")
                else:
                    with st.spinner("Finding the best subtitle matches…"):
                        hits = engine.search_semantic(text, config.DEFAULT_TOP_K)
                    _show_results(hits)

st.caption(
    f"Searching {status['semantic_chunks']:,} subtitle pieces · "
    "Links open OpenSubtitles.org"
)
