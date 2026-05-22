"""Clean subtitle text and user queries before vectorization."""
import re

# SRT timestamps
_TIMESTAMP = re.compile(
    r"\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}"
)
_INDEX_LINE = re.compile(r"\n?\d+\r?")
_OS_LINKS = re.compile(
    r"(?:www\.)?osdb\.link/[\w\d]+|www\.OpenSubtitles\.org|"
    r"osdb\.link/ext|api\.OpenSubtitles\.org|OpenSubtitles\.com",
    re.IGNORECASE,
)
_HTML_TAGS = re.compile(r"</?i>|</?b>|</?u>", re.IGNORECASE)
_BOM = "\ufeff"


def clean_subtitle_text(text: str, lowercase: bool = True) -> str:
    """
    Remove timestamps, indices, ads, and noise while keeping dialogue.

    Lowercasing helps keyword (TF-IDF) consistency; semantic models are
    robust either way — we lowercase by default for both pipelines.
    """
    if not text:
        return ""

    text = text.replace(_BOM, "")
    text = _TIMESTAMP.sub(" ", text)
    text = _INDEX_LINE.sub("", text)
    text = re.sub(r"[\r\n]+", " ", text)
    text = _HTML_TAGS.sub("", text)
    text = _OS_LINKS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if lowercase:
        text = text.lower()
    return text


def clean_query(text: str) -> str:
    """Same cleaning as documents so query and index spaces align."""
    return clean_subtitle_text(text, lowercase=True)
