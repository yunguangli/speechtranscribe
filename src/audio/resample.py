"""Sample-rate conversion without scipy.

FFT resampling: transform the signal, keep or zero-pad the spectrum, invert at
the new length. Fine for speech-length buffers (a few seconds) and fast enough
that it never shows up in the latency budget.
"""

from __future__ import annotations

import numpy as np


def resample(samples: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    """Resample mono float32 audio. Returns the input unchanged if rates match."""
    if rate_in == rate_out or samples.size == 0:
        return np.asarray(samples, dtype=np.float32)

    samples = np.asarray(samples, dtype=np.float32)
    n_in = samples.size
    n_out = max(1, int(round(n_in * rate_out / rate_in)))

    spectrum = np.fft.rfft(samples)
    bins_out = n_out // 2 + 1
    if bins_out <= spectrum.size:
        trimmed = spectrum[:bins_out].copy()
    else:
        trimmed = np.zeros(bins_out, dtype=spectrum.dtype)
        trimmed[: spectrum.size] = spectrum

    out = np.fft.irfft(trimmed, n_out) * (n_out / n_in)
    return out.astype(np.float32)
