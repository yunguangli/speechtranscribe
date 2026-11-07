import numpy as np
import pytest

from audio.segmenter import Segmenter, split_at_pauses


class StubVad:
    """Speech is loud, silence is quiet; no model needed."""

    def __init__(self):
        self.reset()

    def reset(self):
        pass

    def prob(self, frame):
        return 1.0 if np.sqrt(np.mean(frame**2)) > 0.05 else 0.0


def make(rate=16000, **kwargs):
    partials = []
    finals = []
    kwargs.setdefault("partial_seconds", 0.5)
    kwargs.setdefault("end_silence_ms", 160)
    seg = Segmenter(
        rate,
        on_partial=lambda audio, seconds, start: partials.append((audio, seconds, start)),
        on_final=lambda audio, seconds, start: finals.append((audio, seconds, start)),
        vad=StubVad(),
        **kwargs,
    )
    return seg, partials, finals


def tone(seconds, rate=16000, amplitude=0.5):
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * 440 * t) * amplitude).astype(np.float32)


def silence(seconds, rate=16000):
    return np.zeros(int(seconds * rate), dtype=np.float32)


def test_silence_produces_nothing():
    seg, partials, finals = make()
    for _ in range(50):
        seg.feed(silence(0.032))
    assert not partials and not finals


def test_speech_then_silence_finalises():
    seg, partials, finals = make()
    seg.feed(tone(1.2))
    seg.feed(silence(0.5))
    assert partials, "partial decode should fire while speaking"
    assert finals, "final segment should fire after the silence"
    audio, seconds, _start = finals[0]
    assert audio.size == int(seconds * 16000)
    assert seconds > 1.0


def test_short_blip_is_dropped():
    seg, _, finals = make()
    seg.feed(tone(0.05))
    seg.feed(silence(0.5))
    assert not finals


def test_reresamples_non_16k():
    seg, _, finals = make(rate=48000)
    seg.feed(tone(1.0, rate=48000))
    seg.feed(silence(0.5, rate=48000))
    assert finals
    audio, seconds, _start = finals[0]
    assert abs(audio.size - seconds * 16000) < 2


def test_segment_start_tracks_stream_position():
    seg, _, finals = make()
    seg.feed(tone(1.0))
    seg.feed(silence(0.5))
    seg.feed(tone(1.0))
    seg.feed(silence(0.5))
    assert len(finals) >= 2
    assert finals[1][2] > finals[0][2]


def test_max_segment_is_capped():
    seg, _, finals = make()
    seg.feed(tone(14.0))
    assert finals, "a long monologue must be cut at max_seconds"
    assert finals[0][1] <= 5.2


def test_long_segment_cuts_at_the_quietest_point():
    """A dip 0.6 s before the cap is a better cut than the clock boundary."""
    seg, _, finals = make()
    audio = tone(6.0)
    dip = int(4.4 * 16000)
    audio[dip : dip + int(0.12 * 16000)] *= 0.2  # quieter, but still speech
    seg.feed(audio)
    seg.flush()

    assert len(finals) >= 2
    assert abs(finals[0][1] - 4.4) < 0.2
    assert sum(item[1] for item in finals) == pytest.approx(6.0, abs=0.1)
    # The tail is carried into the next segment: no gap, no overlap.
    assert finals[1][2] == pytest.approx(finals[0][2] + finals[0][1], abs=1e-3)


def test_partials_stop_before_the_cut():
    seg, partials, _ = make(
        max_seconds=5.0, partial_seconds=0.2, first_partial_seconds=0.2
    )
    seg.feed(tone(4.8))
    assert partials
    assert max(item[1] for item in partials) <= 3.6


def test_flush_finalises_partial_speech():
    seg, _, finals = make()
    seg.feed(tone(1.0))
    seg.flush()
    assert finals


def test_split_at_pauses_keeps_timestamps():
    audio = np.concatenate(
        [tone(1.2), silence(0.5), tone(1.2), silence(0.5), tone(1.2)]
    )
    chunks = split_at_pauses(audio, 5.0, vad=StubVad())
    assert len(chunks) == 3
    starts = [start for start, _audio, _seconds in chunks]
    assert starts == sorted(starts)
    assert starts[0] == pytest.approx(0.0, abs=0.1)
    # Each chunk converts back to the segment's own duration.
    for _start, chunk, seconds in chunks:
        assert abs(chunk.size - seconds * 16000) < 2


def test_split_at_pauses_caps_long_speech():
    chunks = split_at_pauses(tone(12.0), 5.0, vad=StubVad())
    assert len(chunks) >= 3
    assert all(seconds <= 5.2 for _start, _audio, seconds in chunks)
    # Contiguous: no audio is dropped between chunks.
    assert sum(seconds for _start, _audio, seconds in chunks) == pytest.approx(
        12.0, abs=0.15
    )


def test_split_at_pauses_falls_back_when_silent():
    """No VAD hits must not mean one giant encoder pass."""
    chunks = split_at_pauses(silence(20.0), 5.0, vad=StubVad())
    assert len(chunks) == 4
    assert [start for start, _audio, _seconds in chunks] == [0.0, 5.0, 10.0, 15.0]
    assert all(seconds <= 5.2 for _start, _audio, seconds in chunks)
    assert sum(seconds for _start, _audio, seconds in chunks) == 20.0
