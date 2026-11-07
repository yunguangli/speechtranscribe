"""Cheap ASR front/back-end helpers (no ONNX session needed)."""

import numpy as np
import pytest

from asr.qwen3 import Qwen3Asr, SpecialTokens, Transcript, _trim_silence
from asr.text import normalize_text


def tone(seconds: float, rate: int = 16000, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * 220 * t) * amplitude).astype(np.float32)


def test_trim_silence_drops_silent_edges():
    audio = np.concatenate(
        [
            np.zeros(16000, dtype=np.float32),
            tone(1.0),
            np.zeros(16000, dtype=np.float32),
        ]
    )
    trimmed = _trim_silence(audio)
    assert 16000 <= trimmed.size < audio.size
    assert abs(trimmed.size - 16000) < 4000  # speech + 100 ms padding each side


def test_trim_silence_keeps_all_speech():
    audio = tone(0.5)
    assert _trim_silence(audio).size == audio.size


def test_normalize_removes_spaces_between_cjk():
    assert normalize_text("你 好 ， 世 界 。") == "你好，世界。"
    assert normalize_text("我 用 iPhone 。") == "我用 iPhone。"


def test_normalize_collapses_latin_spaces():
    assert normalize_text("Hello,   world.\nNext line.") == "Hello, world. Next line."


def test_normalize_joins_cjk_segments():
    assert normalize_text("你好。 世界。") == "你好。世界。"


def test_normalize_breaks_runaway_repetition():
    assert normalize_text("a" * 40) == "a"
    assert normalize_text("the " * 25).strip() == "the"
    assert normalize_text("你" * 30) == "你"


def test_normalize_keeps_normal_emphasis():
    assert normalize_text("no no no no no") == "no no no no no"
    assert normalize_text("yeah yeah yeah") == "yeah yeah yeah"


def test_accept_draft_full_match():
    preds = np.array([6, 7, 8, 9])
    assert Qwen3Asr._accept_draft([5, 6, 7, 8], preds) == 4


def test_accept_draft_stops_at_first_mismatch():
    preds = np.array([6, 7, 9, 10])
    assert Qwen3Asr._accept_draft([5, 6, 7, 8], preds) == 3


def test_accept_draft_single_token():
    assert Qwen3Asr._accept_draft([5], np.array([9])) == 1


def test_real_time_factor_includes_prefill():
    """Prefill dominates short utterances; leaving it out overstated speed."""
    result = Transcript(
        text="x",
        audio_seconds=10.0,
        encode_seconds=1.0,
        prefill_seconds=2.0,
        decode_seconds=3.0,
    )
    assert result.real_time_factor == pytest.approx(0.6)


def test_real_time_factor_without_audio():
    assert Transcript(text="x").real_time_factor == 0.0


def test_special_tokens_read_from_config():
    tokens = SpecialTokens.from_config(
        {
            "special_tokens": {
                "im_start_token_id": 1,
                "im_end_token_id": 2,
                "audio_start_token_id": 3,
                "audio_end_token_id": 4,
                "audio_pad_token_id": 5,
                "asr_text_token_id": 6,
                "eos_token_ids": [7, 8],
            }
        }
    )
    assert (tokens.im_start, tokens.im_end) == (1, 2)
    assert (tokens.audio_start, tokens.audio_end) == (3, 4)
    assert (tokens.audio_pad, tokens.asr_text) == (5, 6)
    assert tokens.eos == frozenset({7, 8})


def test_special_tokens_fall_back_for_old_config():
    tokens = SpecialTokens.from_config({})
    assert tokens.im_start == 151644
    assert tokens.audio_pad == 151676
    assert tokens.asr_text == 151704
    assert tokens.eos == frozenset({151643, 151645})