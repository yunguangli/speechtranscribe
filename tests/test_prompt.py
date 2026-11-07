"""Prompt construction, without loading the neural nets."""

import pytest

from asr.qwen3 import Qwen3Asr

MODEL_DIR = "src/assets/models/qwen3-asr-0.6b-int4"


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from pathlib import Path

    if not Path(MODEL_DIR, "tokenizer.json").is_file():
        pytest.skip("model weights not downloaded")
    eng = Qwen3Asr(model_dir=MODEL_DIR)
    eng._tokenizer = __import__("tokenizers").Tokenizer.from_file(
        f"{MODEL_DIR}/tokenizer.json"
    )
    return eng


def test_system_turn_is_constant(engine):
    assert engine._system_ids() == [151644, 8948, 198, 151645, 198]


def test_prompt_has_one_pad_per_audio_token(engine):
    ids = engine._prompt_ids(37, None)
    assert ids.count(151676) == 37
    # Audio start/end wrap the pads, and the assistant turn is last.
    first = ids.index(151676)
    assert ids[first - 1] == 151669
    assert ids[first + 37] == 151670
    assert ids[-3:] == [151644, 77091, 198]


def test_language_suffix(engine):
    ids = engine._prompt_ids(8, "Chinese")
    assert ids[-1] == 151704  # <asr_text> closes the preamble
    decoded = engine._tokenizer.decode(ids[-4:], skip_special_tokens=True)
    assert "Chinese" in decoded


def test_auto_language_has_no_suffix(engine):
    ids = engine._prompt_ids(8, None)
    assert 151704 not in ids
