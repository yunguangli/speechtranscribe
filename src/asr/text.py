"""Transcript text cleanup shared by the engine and the UI."""

from __future__ import annotations

import re

# CJK ideographs: no spaces between them, and no space around CJK punctuation.
_CJK = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CJK_PUNCT = "，。！？、；：）》」』】"
_CJK_GAP = re.compile(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])")
_CJK_PUNCT_GAP = re.compile(rf"\s+(?=[{_CJK_PUNCT}])")
_CJK_PUNCT_TRAIL = re.compile(rf"(?<=[{_CJK_PUNCT}])\s+")

# Runaway decode loops repeat one char or a short phrase dozens of times. The
# threshold is high enough that normal emphatic repetition ("no no no") stays.
_REPEAT_CHAR = re.compile(r"(.)\1{19,}")
_REPEAT_PHRASE = re.compile(
    r"\b((?:\S+\s+){0,3}\S+)(?:\s+\1){19,}\b", re.IGNORECASE
)


def normalize_text(text: str) -> str:
    """Collapse whitespace, fix CJK/Latin spacing, break repetition loops."""
    text = " ".join(text.split())
    text = _CJK_GAP.sub("", text)
    text = _CJK_PUNCT_GAP.sub("", text)
    text = _CJK_PUNCT_TRAIL.sub("", text)
    text = _REPEAT_CHAR.sub(r"\1", text)
    text = _REPEAT_PHRASE.sub(r"\1", text)
    return text.strip()