"""Voice activity detection.

Silero VAD v5 (2.3 MB ONNX) when ``assets/silero_vad.onnx`` is present; a plain
adaptive-energy detector otherwise. Both expose the same tiny interface:
``reset()`` and ``prob(frame_16k)`` where the frame is 512 float32 samples.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

FRAME = 512  # 32 ms at 16 kHz; Silero v5 is trained on exactly this size
RATE = 16_000


class SileroVad:
    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        x = np.asarray(frame, dtype=np.float32)[np.newaxis, :]
        out, self._state = self.session.run(
            None, {"input": x, "state": self._state, "sr": np.array(RATE, np.int64)}
        )
        return float(out[0, 0])


class EnergyVad:
    """Fallback: RMS against a slow-tracking noise floor."""

    # How fast the floor follows a quiet room versus a noisy one. Downward it
    # tracks quickly (a fan switching off should not keep the VAD open), upward
    # it moves ~60x slower so a long utterance cannot raise its own threshold.
    _FALL = 0.05
    _RISE = 0.0005
    _MAX_FLOOR = 0.02

    def __init__(self) -> None:
        self._floor = 1e-4

    def reset(self) -> None:
        self._floor = 1e-4

    def prob(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32))) + 1e-9)
        if rms < self._floor:
            self._floor += self._FALL * (rms - self._floor)
        else:
            self._floor += self._RISE * (rms - self._floor)
            self._floor = min(self._floor, self._MAX_FLOOR)
        threshold = max(self._floor * 4.0, 0.004)
        return 1.0 if rms > threshold else 0.0


def make_vad(assets_dir: Path) -> SileroVad | EnergyVad:
    path = Path(assets_dir) / "silero_vad.onnx"
    if path.is_file():
        try:
            return SileroVad(path)
        except Exception:
            pass
    return EnergyVad()


def assets_dir() -> Path:
    env = os.environ.get("FLET_ASSETS_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[1] / "assets"
