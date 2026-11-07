"""Language names understood by Qwen3-ASR.

The model takes a free-form English name in the assistant turn, e.g.
``language Chinese<asr_text>``. ``None`` means auto-detect: the model writes the
detected language before ``<asr_text>`` and we split it off.
"""

from __future__ import annotations

# (label shown in the UI, name passed to the model)
LANGUAGES: list[tuple[str, str]] = [
    ("Auto detect", ""),
    ("English", "English"),
    ("Chinese", "Chinese"),
    ("Cantonese", "Cantonese"),
    ("Japanese", "Japanese"),
    ("Korean", "Korean"),
    ("Spanish", "Spanish"),
    ("French", "French"),
    ("German", "German"),
    ("Italian", "Italian"),
    ("Portuguese", "Portuguese"),
    ("Russian", "Russian"),
    ("Arabic", "Arabic"),
    ("Hindi", "Hindi"),
    ("Thai", "Thai"),
    ("Vietnamese", "Vietnamese"),
    ("Indonesian", "Indonesian"),
    ("Malay", "Malay"),
    ("Turkish", "Turkish"),
    ("Dutch", "Dutch"),
    ("Polish", "Polish"),
    ("Czech", "Czech"),
    ("Swedish", "Swedish"),
    ("Danish", "Danish"),
    ("Finnish", "Finnish"),
    ("Greek", "Greek"),
    ("Hungarian", "Hungarian"),
    ("Romanian", "Romanian"),
    ("Filipino", "Filipino"),
    ("Persian", "Persian"),
    ("Macedonian", "Macedonian"),
]

LANGUAGE_CODES = {
    "English": "EN",
    "Chinese": "ZH",
    "Cantonese": "YUE",
    "Japanese": "JA",
    "Korean": "KO",
    "Spanish": "ES",
    "French": "FR",
    "German": "DE",
    "Italian": "IT",
    "Portuguese": "PT",
    "Russian": "RU",
    "Arabic": "AR",
    "Hindi": "HI",
    "Thai": "TH",
    "Vietnamese": "VI",
    "Indonesian": "ID",
    "Malay": "MS",
    "Turkish": "TR",
    "Dutch": "NL",
    "Polish": "PL",
    "Czech": "CS",
    "Swedish": "SV",
    "Danish": "DA",
    "Finnish": "FI",
    "Greek": "EL",
    "Hungarian": "HU",
    "Romanian": "RO",
    "Filipino": "FIL",
    "Persian": "FA",
    "Macedonian": "MK",
}

# Chinese dialects the model supports when it hears them; it reports the
# dialect name, so give them badges too.
DIALECTS = {
    "Sichuan": "SC",
    "Wu": "WU",
    "Minnan": "NAN",
    "Cantonese (Hong Kong accent)": "YUE",
    "Cantonese (Guangdong accent)": "YUE",
}


def badge(language: str | None) -> str:
    """Short uppercase code for the transcript chip."""
    if not language:
        return "?"
    key = language.strip()
    if key in LANGUAGE_CODES:
        return LANGUAGE_CODES[key]
    if key in DIALECTS:
        return DIALECTS[key]
    return key[:3].upper()
