"""Clean subtitle text and user queries before vectorization."""
import re

_TIMESTAMP = re.compile(
    r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}"
)
_CUE_INDEX = re.compile(r"^\s*\d+\s*$")
_OS_LINKS = re.compile(
    r"(?:www\.)?osdb\.link/[\w\d]+|www\.OpenSubtitles\.org|"
    r"osdb\.link/ext|api\.OpenSubtitles\.org|OpenSubtitles\.com",
    re.IGNORECASE,
)
_HTML_TAGS = re.compile(r"</?i>|</?b>|</?u>", re.IGNORECASE)
_BOM = "\ufeff"


def extract_dialogue_cues(raw: str) -> list[str]:
    """
    Split SRT into one string per subtitle cue (dialogue block).

    Keeps lines separate per cue instead of one giant blob — critical for
    accurate line-level search.
    """
    if not raw:
        return []

    raw = raw.replace(_BOM, "")
    raw = _HTML_TAGS.sub("", raw)
    raw = _OS_LINKS.sub(" ", raw)

    cues: list[str] = []
    for block in re.split(r"\n\s*\n", raw):
        lines: list[str] = []
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if _TIMESTAMP.search(line) or _CUE_INDEX.match(line):
                continue
            lines.append(line)
        if lines:
            cues.append(" ".join(lines))
    return cues


def clean_subtitle_text(text: str, lowercase: bool = True) -> str:
    """Full-document clean (fallback). Prefer cue-based chunking for search."""
    if not text:
        return ""

    cues = extract_dialogue_cues(text)
    if cues:
        text = " ".join(cues)
    else:
        text = text.replace(_BOM, "")
        text = _TIMESTAMP.sub(" ", text)
        text = re.sub(r"[\r\n]+", " ", text)
        text = _HTML_TAGS.sub("", text)
        text = _OS_LINKS.sub(" ", text)

    text = re.sub(r"\s+", " ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def clean_query(text: str) -> str:
    return clean_subtitle_text(text, lowercase=True)
