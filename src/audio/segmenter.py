"""Turn a live PCM stream into utterances.

The recorder hands us 32 ms-ish chunks. This class buffers them, runs VAD,
keeps a short pre-roll so the first syllable is not clipped, and emits:

* ``on_partial(audio_16k, seconds, start)``  - every ~1 s while talking
* ``on_final(audio_16k, seconds, start)``    - after ~400 ms of silence, or at 5 s

Long monologues are cut at a low-energy 100 ms window in the last ~1.2 s,
not at a hard clock boundary: a mid-word cut loses words, a low-energy cut
lands in a natural gap. Among the quietest windows, the latest one is chosen
so chunks stay close to the cap. The audio after the cut is carried into the
next segment, so no samples are dropped and the segments stay contiguous.

``start`` is the segment's position in the stream, in seconds, so the caller
can merge segments into paragraphs when the pause between them is short.

Audio is resampled to 16 kHz only when it is handed over, never per chunk.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np

from .resample import resample
from .vad import FRAME, RATE, make_vad

SAMPLE_RATE = RATE
FRAME_SECONDS = FRAME / RATE  # 32 ms
FRAME_SAMPLES = FRAME


class Segmenter:
    def __init__(
        self,
        rate: int,
        on_partial: Callable[[np.ndarray, float, float], None],
        on_final: Callable[[np.ndarray, float, float], None],
        *,
        onset_frames: int = 2,
        end_silence_ms: int = 400,
        pre_roll_ms: int = 320,
        partial_seconds: float = 1.0,
        first_partial_seconds: float = 0.6,
        max_seconds: float = 5.0,
        cut_search_seconds: float = 1.2,
        cut_window_ms: int = 100,
        min_cut_seconds: float = 2.5,
        min_tail_seconds: float = 0.25,
        partial_stop_ratio: float = 0.7,
        assets_dir=None,
        vad=None,
    ) -> None:
        self.rate = rate
        self.on_partial = on_partial
        self.on_final = on_final
        self.onset_frames = onset_frames
        self.end_frames = max(1, int(end_silence_ms / 1000 / FRAME_SECONDS))
        self.pre_roll_frames = max(0, int(pre_roll_ms / 1000 / FRAME_SECONDS))
        self.partial_seconds = partial_seconds
        self.first_partial_seconds = first_partial_seconds
        self.max_samples = int(max_seconds * rate)
        self.cut_search_samples = max(1, int(cut_search_seconds * rate))
        self.cut_window = max(4, int(cut_window_ms / 1000 * rate))
        self.min_cut_samples = max(0, int(min_cut_seconds * rate))
        self.min_tail_samples = max(1, int(min_tail_seconds * rate))
        self.partial_stop_samples = int(partial_stop_ratio * self.max_samples)

        self._frame_len = max(1, int(round(rate * FRAME_SECONDS)))
        self._pending = np.zeros(0, dtype=np.float32)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=self.pre_roll_frames)

        self._vad = vad or (
            make_vad(assets_dir) if assets_dir else make_vad(_default_assets())
        )
        self._active = False
        self._onset = 0
        self._silence = 0
        self._chunks: list[np.ndarray] = []
        self._samples = 0
        self._since_partial = 0.0
        self._partials = 0
        self._frames_seen = 0
        self._segment_start = 0.0

    # -- public ------------------------------------------------------------

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
        self._pre_roll.clear()
        self._vad.reset()
        self._active = False
        self._onset = 0
        self._silence = 0
        self._chunks = []
        self._samples = 0
        self._since_partial = 0.0
        self._partials = 0
        self._frames_seen = 0
        self._segment_start = 0.0

    def feed(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        self._pending = np.concatenate([self._pending, samples.astype(np.float32)])
        while self._pending.size >= self._frame_len:
            frame = self._pending[: self._frame_len]
            self._pending = self._pending[self._frame_len :]
            self._process(frame)

    def flush(self) -> None:
        """Finalise whatever is buffered (stop button pressed)."""
        if self._active and self._samples:
            self._emit_final()
        else:
            self._reset_segment()

    @property
    def speaking(self) -> bool:
        return self._active

    # -- internals ---------------------------------------------------------

    def _process(self, frame: np.ndarray) -> None:
        frame16 = resample(frame, self.rate, RATE)
        if frame16.size < FRAME_SAMPLES:
            frame16 = np.pad(frame16, (0, FRAME_SAMPLES - frame16.size))
        elif frame16.size > FRAME_SAMPLES:
            frame16 = frame16[:FRAME_SAMPLES]

        speech = self._vad.prob(frame16) >= 0.5

        if not self._active:
            self._pre_roll.append(frame.copy())
            if speech:
                self._onset += 1
                if self._onset >= self.onset_frames:
                    self._start()
            else:
                self._onset = 0
            self._frames_seen += 1
            return

        self._chunks.append(frame.copy())
        self._samples += frame.size

        if speech:
            self._silence = 0
        else:
            self._silence += 1

        self._since_partial += FRAME_SECONDS
        cadence = (
            self.first_partial_seconds if self._partials == 0 else self.partial_seconds
        )
        if (
            self._since_partial >= cadence
            and self._silence < self.end_frames
            and self._samples >= self.rate // 2
            and self._samples < self.partial_stop_samples
        ):
            self._since_partial = 0.0
            self._partials += 1
            self.on_partial(
                self._to_16k(), self._samples / self.rate, self._segment_start
            )

        if self._silence >= self.end_frames:
            self._emit_final()
        elif self._samples >= self.max_samples:
            self._cut_long_segment()
        self._frames_seen += 1

    def _start(self) -> None:
        self._active = True
        self._onset = 0
        self._silence = 0
        self._since_partial = 0.0
        self._partials = 0
        # The pre-roll frames come before the onset frames, so this is where
        # the segment really begins.
        self._segment_start = max(
            0.0, (self._frames_seen - len(self._pre_roll)) * FRAME_SECONDS
        )
        self._chunks = list(self._pre_roll)
        self._samples = sum(chunk.size for chunk in self._chunks)
        self._pre_roll.clear()

    def _to_16k(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._chunks)
        return resample(audio, self.rate, RATE)

    def _quietest_cut(self, audio: np.ndarray) -> int:
        """Sample index of the quietest 100 ms window near the end.

        Looks back from the segment cap so no extra audio has to be buffered
        before a cut. Always leaves ``min_tail_samples`` for the next segment.
        """
        right = audio.size - self.min_tail_samples
        if right <= 0:
            return audio.size

        left = max(self.min_cut_samples, right - self.cut_search_samples)
        starts = right - left + 1
        if starts < 1 or audio.size - left < self.cut_window:
            return max(0, min(right, audio.size))

        window = np.abs(audio[left:])
        csum = np.concatenate(([0.0], np.cumsum(window, dtype=np.float64)))
        sums = csum[self.cut_window :] - csum[: -self.cut_window]
        limit = min(sums.size, starts)
        if limit <= 0:
            return max(0, min(right, audio.size))
        energy = sums[:limit]
        # The latest window in the quietest 10 % keeps chunks close to the cap
        # while still landing in a dip rather than a vowel.
        threshold = float(np.quantile(energy, 0.10))
        quiet = np.flatnonzero(energy <= threshold)
        if quiet.size == 0:
            quiet = np.array([int(np.argmin(energy))])
        pos = int(quiet[-1])
        inner = int(np.argmin(window[pos : pos + self.cut_window]))
        return max(0, min(left + pos + inner, right))

    def _cut_long_segment(self) -> None:
        """Emit the head of an over-long segment and carry the tail forward."""
        audio = np.concatenate(self._chunks)
        # Search the exact cap-length window, not the extra frame that made
        # _samples cross the threshold: the cut must not jitter with framing.
        target = audio[: self.max_samples]
        cut = self._quietest_cut(target)
        if cut <= 0 or cut >= audio.size:
            self._emit_final()
            return

        head = resample(target[:cut], self.rate, RATE)
        seconds = head.size / RATE
        if seconds >= 0.25:
            self.on_final(head, seconds, self._segment_start)

        tail = audio[cut:]
        self._chunks = [tail]
        self._samples = tail.size
        self._segment_start += cut / self.rate
        self._since_partial = 0.0
        self._partials = 0
        self._active = True

    def _emit_final(self) -> None:
        audio = self._to_16k()
        seconds = audio.size / RATE
        if seconds >= 0.25:
            self.on_final(audio, seconds, self._segment_start)
        self._reset_segment()

    def _reset_segment(self) -> None:
        self._active = False
        self._onset = 0
        self._silence = 0
        self._chunks = []
        self._samples = 0
        self._since_partial = 0.0
        self._partials = 0


def _default_assets():
    from .vad import assets_dir

    return assets_dir()


def split_at_pauses(
    audio: np.ndarray,
    max_seconds: float,
    *,
    rate: int = RATE,
    block_seconds: float = 1.0,
    vad=None,
) -> list[tuple[float, np.ndarray, float]]:
    """Cut a whole recording at pauses, keeping timestamps.

    The live path calls :meth:`Segmenter.feed` as audio arrives; this is the
    same state machine driven from a buffer, so an imported file gets the same
    chunk boundaries and no single encoder pass over an hour of audio.

    Returns ``(start_seconds, chunk_16k, seconds)``. Audio with no detected
    speech is returned as one chunk so the caller can still decide what to do.
    """
    chunks: list[tuple[float, np.ndarray, float]] = []

    def on_final(chunk: np.ndarray, seconds: float, start: float) -> None:
        chunks.append((start, chunk, seconds))

    segmenter = Segmenter(
        rate,
        on_partial=lambda *_args: None,
        on_final=on_final,
        max_seconds=max_seconds,
        # There is no UI watching a file import, so partials are pure waste.
        partial_seconds=1e9,
        first_partial_seconds=1e9,
        vad=vad,
    )
    block = max(1, int(block_seconds * rate))
    for offset in range(0, audio.size, block):
        segmenter.feed(audio[offset : offset + block])
    segmenter.flush()

    if not chunks and audio.size:
        # Nothing the VAD recognises as speech (music, unusual noise, a very
        # quiet recording). Still cut to length: one 40-minute encoder pass
        # would truncate the transcript and blow up memory.
        audio16 = resample(np.asarray(audio, dtype=np.float32), rate, RATE)
        step = max(1, int(max_seconds * RATE))
        for offset in range(0, audio16.size, step):
            piece = audio16[offset : offset + step]
            chunks.append((offset / RATE, piece, piece.size / RATE))
    return chunks
