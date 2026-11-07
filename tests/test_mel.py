import numpy as np

from asr.mel import N_MELS, N_FFT, SAMPLE_RATE, log_mel, mel_filterbank


def test_filterbank_shape():
    filters = mel_filterbank()
    assert filters.shape == (N_MELS, N_FFT // 2 + 1)
    assert filters.min() >= 0.0
    # Every filter has weight: an all-zero row would silence a band.
    assert (filters.sum(axis=1) > 0).all()


def test_log_mel_shape():
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 s
    mel = log_mel(audio)
    assert mel.shape[0] == 1
    assert mel.shape[1] == N_MELS
    # 1 s -> 101 frames, minus the one Whisper drops.
    assert mel.shape[2] == 100
    assert mel.dtype == np.float32


def test_log_mel_normalized():
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(SAMPLE_RATE) * 0.1).astype(np.float32)
    mel = log_mel(audio)
    assert np.isfinite(mel).all()
    assert mel.max() <= 1.0 + 1e-5


def test_short_audio_is_padded():
    mel = log_mel(np.ones(100, dtype=np.float32) * 0.01)
    assert mel.shape[2] >= 1
