"""Transcript export helpers, without a running Flet page."""

import asyncio

import numpy as np
import pytest

import app as app_module
from app import Job, SpeechApp, Utterance


class FakeSaveButton:
    disabled = False


class FakeApp(SpeechApp):
    """SpeechApp methods only; skips __init__ so no page is needed."""


def make_app(*utterances: Utterance) -> FakeApp:
    app = FakeApp.__new__(FakeApp)
    app.utterances = {u.id: u for u in utterances}
    app.order = [u.id for u in utterances]
    app.save_btn = FakeSaveButton()
    app.recording = False
    app.transcribing = False
    app._jobs = []
    app._current_job = None
    app._language_name = ""
    app._drafts = {}
    app._sticky_uid = None
    app._sticky_language = None
    app._debug = False
    return app


def test_srt_timestamp():
    assert SpeechApp._srt_timestamp(0) == "00:00:00,000"
    assert SpeechApp._srt_timestamp(61.234) == "00:01:01,234"
    assert SpeechApp._srt_timestamp(3661.5) == "01:01:01,500"


def test_export_text_skips_empty_rows():
    app = make_app(
        Utterance(id=0, text="Hello there.", final=True, seconds=2.0),
        Utterance(id=1, text="", final=False),
        Utterance(id=2, text="Second line.", final=True, seconds=1.5),
    )
    assert SpeechApp._export_text(app) == "Hello there.\n\nSecond line.\n"


def test_export_text_empty():
    assert SpeechApp._export_text(make_app()) == ""


def test_export_srt_uses_absolute_start():
    app = make_app(
        Utterance(id=0, text="First.", final=True, start=0.0, seconds=2.0),
        Utterance(id=1, text="Second.", final=True, start=5.0, seconds=1.5),
    )
    out = SpeechApp._export_srt(app)
    assert out.startswith("1\n00:00:00,000 --> 00:00:02,000\nFirst.")
    assert "2\n00:00:05,000 --> 00:00:06,500\nSecond." in out
    assert out.endswith("\n")


def test_export_srt_empty():
    assert SpeechApp._export_srt(make_app()) == ""


def test_export_srt_emits_one_cue_per_chunk():
    """Imports queue many chunks into one paragraph; each keeps its timing."""
    utt = Utterance(id=0, final=True, start=0.0, seconds=9.0)
    utt.segments = {0.0: "First part.", 3.0: "Second part.", 7.5: "Last part."}
    utt.text = "First part. Second part. Last part."
    out = SpeechApp._export_srt(make_app(utt))
    assert "1\n00:00:00,000 --> 00:00:03,000\nFirst part." in out
    assert "2\n00:00:03,000 --> 00:00:07,500\nSecond part." in out
    assert "3\n00:00:07,500 --> 00:00:09,000\nLast part." in out
    assert out.count("-->") == 3


def test_save_enabled_only_when_idle_and_text_exists():
    app = make_app(Utterance(id=0, text="Hello.", final=True, seconds=1.0))

    SpeechApp._update_save_enabled(app)
    assert app.save_btn.disabled is False

    app.recording = True
    SpeechApp._update_save_enabled(app)
    assert app.save_btn.disabled is True

    app.recording = False
    app.transcribing = True
    SpeechApp._update_save_enabled(app)
    assert app.save_btn.disabled is True

    app.transcribing = False
    app._jobs.append(object())
    SpeechApp._update_save_enabled(app)
    assert app.save_btn.disabled is True


def test_save_disabled_without_text():
    app = make_app(Utterance(id=0, text="  ", final=True))
    SpeechApp._update_save_enabled(app)
    assert app.save_btn.disabled is True


@pytest.mark.asyncio
async def test_copy_puts_every_utterance_on_the_clipboard(monkeypatch):
    captured: dict[str, str] = {}

    class FakeClipboard:
        async def set(self, value: str) -> None:
            captured["value"] = value

    monkeypatch.setattr(app_module.ft, "Clipboard", FakeClipboard)

    app = make_app(
        Utterance(id=0, text="First sentence.", final=True, seconds=2.0),
        Utterance(id=1, text="Second sentence.", final=True, seconds=1.0),
    )
    app._snack = lambda _message: None

    await SpeechApp._copy_all(app, None)

    assert captured["value"] == "First sentence.\n\nSecond sentence.\n"


class FakePage:
    def update(self) -> None:
        pass


def test_apply_merges_segments_without_overwriting():
    """A new segment must append to the paragraph, never replace its text."""
    utt = Utterance(id=0, text="", final=True, start=0.0)
    app = make_app(utt)
    app.page = FakePage()
    app._render_utterance = lambda _utt: None
    app._show_empty = lambda _show: None

    def apply(text, final, seconds, start):
        SpeechApp._apply(
            app,
            Job(0, np.zeros(1, dtype=np.float32), final, seconds, start),
            text,
            "English",
            seconds,
        )

    apply("First.", True, 2.0, 0.0)
    apply("Second", False, 2.0, 2.4)
    assert utt.text == "First. Second"

    apply("Second.", True, 2.0, 2.4)
    assert utt.text == "First. Second."
    assert utt.seconds == pytest.approx(4.4)

    # An empty final for the last segment drops its stale partial text.
    apply("", True, 2.0, 2.4)
    assert utt.text == "First."


def test_empty_final_keeps_row_while_more_chunks_are_queued():
    """An import can have a silent chunk before speech; don't drop the row."""
    utt = Utterance(id=0, text="", final=True, start=0.0)
    app = make_app(utt)
    app.page = FakePage()
    app._render_utterance = lambda _utt: None
    app._show_empty = lambda _show: None
    app._paragraph_uid = None

    class FakeList:
        controls: list = []

    app.list_view = FakeList()
    app._jobs = [Job(0, np.zeros(1, dtype=np.float32), True, 1.0, 3.0)]

    SpeechApp._apply(
        app,
        Job(0, np.zeros(1, dtype=np.float32), True, 1.0, 0.0),
        "",
        "English",
        1.0,
    )
    assert app.utterances[0] is utt
    assert app.order == [0]

    # Nothing else queued: the empty row is removed.
    app._jobs.clear()
    SpeechApp._apply(
        app,
        Job(0, np.zeros(1, dtype=np.float32), True, 1.0, 3.0),
        "",
        "English",
        1.0,
    )
    assert 0 not in app.utterances
    assert app.order == []


def test_paragraph_groups_short_pauses_only():
    app = make_app()
    counter = {"n": 0}

    def new_utterance(start: float = 0.0) -> int:
        uid = counter["n"]
        counter["n"] += 1
        app.utterances[uid] = Utterance(id=uid, start=start)
        app.order.append(uid)
        return uid

    app._new_utterance = new_utterance
    app._paragraph_uid = None
    app._last_segment_end = 0.0

    uid = SpeechApp._paragraph_for(app, 0.19)
    app._last_segment_end = 2.30
    assert SpeechApp._paragraph_for(app, 2.30) == uid  # continuous speech
    app._last_segment_end = 5.98
    assert SpeechApp._paragraph_for(app, 5.98) == uid

    # A long pause opens a fresh paragraph and drops the sticky language.
    app._sticky_uid = uid
    app._sticky_language = "English"
    app._last_segment_end = 15.01
    new_uid = SpeechApp._paragraph_for(app, 20.0)
    assert new_uid != uid
    assert app._sticky_uid == new_uid
    assert app._sticky_language is None


class FakeResult:
    def __init__(self, text: str, token_ids: list[int]) -> None:
        self.text = text
        self.token_ids = token_ids


def test_partial_tokens_are_reused_as_the_next_draft():
    app = make_app(Utterance(id=0))
    partial = Job(0, np.zeros(1, dtype=np.float32), False, 1.0, 0.0)
    final = Job(0, np.zeros(1, dtype=np.float32), True, 1.2, 0.0)

    SpeechApp._remember_result(app, partial, "English", FakeResult("Hello", [1, 2, 3]))
    assert SpeechApp._draft_for(app, final, "English") == [1, 2, 3]

    SpeechApp._remember_result(
        app, final, "English", FakeResult("Hello.", [1, 2, 3, 4])
    )
    assert SpeechApp._draft_for(app, final, "English") is None


def test_sticky_language_applies_to_later_chunks():
    utt = Utterance(id=0, final=True)
    app = make_app(utt)
    app.page = FakePage()
    app._render_utterance = lambda _utt: None
    app._show_empty = lambda _show: None
    app._sticky_uid = 0  # set by _paragraph_for when the row is created

    SpeechApp._apply(
        app, Job(0, np.zeros(1, dtype=np.float32), True, 2.0, 0.0), "Hello.", "English", 2.0
    )
    assert app._sticky_uid == 0
    assert (
        SpeechApp._language_for(app, Job(0, np.zeros(1), True, 1.0, 2.0)) == "English"
    )
    # Another paragraph must not inherit it.
    assert SpeechApp._language_for(app, Job(1, np.zeros(1), True, 1.0, 9.0)) is None
    # A manual choice always wins.
    app._language_name = "Chinese"
    assert (
        SpeechApp._language_for(app, Job(0, np.zeros(1), True, 1.0, 2.0)) == "Chinese"
    )


def test_final_supersedes_queued_partials():
    app = make_app(Utterance(id=0, text="", final=False))
    app._wake = asyncio.Event()
    audio = np.zeros(16, dtype=np.float32)

    SpeechApp._submit(app, Job(0, audio, final=False, seconds=0.5, segment_start=0.0))
    SpeechApp._submit(app, Job(0, audio, final=False, seconds=1.0, segment_start=0.0))
    assert len(app._jobs) == 1  # newer partial replaced the older one

    SpeechApp._submit(app, Job(0, audio, final=True, seconds=1.2, segment_start=0.0))
    assert len(app._jobs) == 1
    assert app._jobs[0].final is True


class FakeEngine:
    def __init__(self) -> None:
        self.cancelled = 0

    def cancel_current(self) -> None:
        self.cancelled += 1


def test_final_cancels_running_partial():
    app = make_app(Utterance(id=0, text="", final=False))
    app._wake = asyncio.Event()
    app.engine = FakeEngine()
    audio = np.zeros(16, dtype=np.float32)

    app._current_job = Job(0, audio, final=False, seconds=1.0, segment_start=0.0)
    SpeechApp._submit(app, Job(0, audio, final=True, seconds=1.2, segment_start=0.0))
    assert app.engine.cancelled == 1

    # A running final must not be aborted by the next final.
    app._current_job = Job(0, audio, final=True, seconds=1.0, segment_start=0.0)
    SpeechApp._submit(app, Job(0, audio, final=True, seconds=1.2, segment_start=3.0))
    assert app.engine.cancelled == 1

