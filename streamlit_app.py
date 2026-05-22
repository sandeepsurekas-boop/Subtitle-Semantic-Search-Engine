"""Streamlit UI for subtitle search (text + audio)."""
import tempfile
from pathlib import Path

import streamlit as st

import config
from src.retrieve import SubtitleSearchEngine

st.set_page_config(page_title="Subtitle Search", page_icon="🎬", layout="wide")
st.title("Subtitle Semantic Search Engine")
st.caption("Shazam-style retrieval over OpenSubtitles — semantic & keyword modes")

@st.cache_resource
def load_engine():
    return SubtitleSearchEngine()

engine = load_engine()

mode = st.radio("Search mode", ["semantic", "keyword"], horizontal=True)
top_k = st.slider("Results", 5, 30, config.DEFAULT_TOP_K)

tab_text, tab_audio = st.tabs(["Text query", "Audio query (~2 min)"])

with tab_text:
    query = st.text_input("Enter dialogue or phrase from a movie/TV show")
    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            if mode == "keyword":
                results = engine.search_keyword(query, top_k)
            else:
                results = engine.search_semantic(query, top_k)
        for r in results:
            st.markdown(f"**[{r.filename or r.subtitle_id}]({r.opensubtitles_url})** — score `{r.score:.4f}`")
            st.write(r.snippet)

with tab_audio:
    audio = st.file_uploader("Upload TV/movie audio clip", type=["wav", "mp3", "m4a", "ogg"])
    if st.button("Search audio", type="primary") and audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.name).suffix) as tmp:
            tmp.write(audio.getvalue())
            path = Path(tmp.name)
        with st.spinner("Transcribing with Whisper and searching..."):
            transcript, results = engine.search_audio(path, mode=mode, top_k=top_k)
        st.subheader("Transcript")
        st.write(transcript)
        st.subheader("Matches")
        for r in results:
            st.markdown(f"**[{r.filename or r.subtitle_id}]({r.opensubtitles_url})** — score `{r.score:.4f}`")
            st.write(r.snippet)
