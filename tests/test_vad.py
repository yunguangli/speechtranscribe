"""Energy VAD fallback (used when silero_vad.onnx cannot be loaded)."""

import numpy as np

from audio.vad import EnergyVad


def noisy(rms: float, samples: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(samples) * rms).astype(np.float32)


def test_quiet_background_is_silence():
    vad = EnergyVad()
    assert vad.prob(noisy(1e-4)) == 0.0


def test_speech_over_quiet_background_is_detected():
    vad = EnergyVad()
    for _ in range(20):
        vad.prob(noisy(1e-4))
    assert vad.prob(noisy(0.05)) == 1.0


def test_floor_adapts_up_in_a_noisy_room():
    """A fan should not keep the VAD open forever."""
    vad = EnergyVad()
    fan = noisy(0.05, seed=1)
    assert vad.prob(fan) == 1.0  # loud relative to the starting floor
    for _ in range(1500):
        vad.prob(fan)
    assert vad.prob(fan) == 0.0


def test_speech_still_detected_after_floor_rises():
    vad = EnergyVad()
    fan = noisy(0.05, seed=1)
    for _ in range(1500):
        vad.prob(fan)
    assert vad.prob(noisy(0.3, seed=2)) == 1.0
