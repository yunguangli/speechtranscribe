"""Whisper-compatible log-mel front-end (numpy only).

Qwen3-ASR's encoder was trained on Whisper-style features: 16 kHz mono,
128 mel bins, n_fft 400, hop 160, Slaney-normalised filters, log10 with a
-8 dB floor, then (x + 4) / 4. Anything else and the transcript degrades
into guesses.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
FMIN = 0.0
FMAX = 8000.0

_FILTERBANK: np.ndarray | None = None


def mel_filterbank() -> np.ndarray:
    """Slaney-normalised mel filterbank, shape [N_MELS, 1 + N_FFT // 2]."""
    global _FILTERBANK
    if _FILTERBANK is not None:
        return _FILTERBANK

    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    def hz_to_mel(freq: np.ndarray) -> np.ndarray:
        mels = freq / f_sp
        above = freq >= min_log_hz
        mels[above] = min_log_mel + np.log(freq[above] / min_log_hz) / logstep
        return mels

    def mel_to_hz(mels: np.ndarray) -> np.ndarray:
        freqs = mels * f_sp
        above = mels >= min_log_mel
        freqs[above] = min_log_hz * np.exp(logstep * (mels[above] - min_log_mel))
        return freqs

    fft_freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
    points = np.linspace(
        hz_to_mel(np.array([FMIN]))[0], hz_to_mel(np.array([FMAX]))[0], N_MELS + 2
    )
    band_hz = mel_to_hz(points)

    diff = np.diff(band_hz)
    ramps = band_hz[:, np.newaxis] - fft_freqs[np.newaxis, :]
    weights = np.zeros((N_MELS, len(fft_freqs)), dtype=np.float32)
    for i in range(N_MELS):
        lower = -ramps[i] / diff[i]
        upper = ramps[i + 2] / diff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    weights *= (2.0 / (band_hz[2 : N_MELS + 2] - band_hz[:N_MELS]))[:, np.newaxis]
    weights = weights.astype(np.float32)
    _FILTERBANK = weights
    return weights


def log_mel(audio: np.ndarray) -> np.ndarray:
    """float32 mono 16 kHz samples in [-1, 1] -> [1, 128, frames]."""
    if audio.ndim != 1:
        raise ValueError("audio must be mono")
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.size < N_FFT:
        audio = np.pad(audio, (0, N_FFT - audio.size))

    padded = np.pad(audio, N_FFT // 2, mode="reflect")
    # Periodic Hann; np.hanning() is the symmetric one.
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)

    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    spec = np.fft.rfft(frames * window, n=N_FFT, axis=-1)
    magnitudes = (np.abs(spec) ** 2).T.astype(np.float32)

    mel = mel_filterbank() @ magnitudes
    log_spec = np.log10(np.clip(mel, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    log_spec = log_spec[:, :-1]  # WhisperFeatureExtractor drops the last frame
    return log_spec[np.newaxis, :, :].astype(np.float32)
