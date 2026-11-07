import io
import struct
import wave

import numpy as np

from audio.wav import decode_wav, pcm16_to_float32


def wav_bytes(samples: np.ndarray, rate=16000, channels=1, width=2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def float_wav_bytes(samples: np.ndarray, rate=16000, channels=1, width=4) -> bytes:
    """A WAV with format tag 3 (IEEE float), which ``wave`` cannot write."""
    raw = samples.astype(f"<f{width}").tobytes()
    block = channels * width
    fmt = struct.pack("<HHIIHH", 3, channels, rate, rate * block, block, width * 8)
    return (
        b"RIFF"
        + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(raw))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(raw))
        + raw
    )


def extensible_float_wav_bytes(samples: np.ndarray, rate=16000) -> bytes:
    """WAVE_FORMAT_EXTENSIBLE wrapping the IEEE-float sub-format (0x0003)."""
    raw = samples.astype("<f4").tobytes()
    guid = bytes.fromhex("0300000000001000800000aa00389b71")
    fmt = (
        struct.pack("<HHIIHH", 0xFFFE, 1, rate, rate * 4, 4, 32)
        + struct.pack("<HHI", 22, 32, 0x4)
        + guid
    )
    return (
        b"RIFF"
        + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(raw))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(raw))
        + raw
    )


def test_pcm16_stream_to_float():
    raw = np.array([0, 16384, -32768], dtype="<i2").tobytes()
    audio = pcm16_to_float32(raw)
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, [0.0, 0.5, -1.0])


def test_decode_16bit_mono():
    samples = np.array([0, 1000, -1000], dtype="<i2")
    audio, rate = decode_wav(wav_bytes(samples))
    assert rate == 16000
    assert audio.size == 3
    np.testing.assert_allclose(audio, samples / 32768.0, atol=1e-7)


def test_decode_stereo_is_mixed_down():
    interleaved = np.array([1000, -1000, 2000, -2000], dtype="<i2")
    audio, _ = decode_wav(wav_bytes(interleaved, channels=2))
    assert audio.size == 2
    np.testing.assert_allclose(audio, [0.0, 0.0], atol=1e-7)


def test_decode_24bit():
    samples = np.array([0, 4194304, -8388608], dtype=np.int32)
    raw = b"".join(int(v).to_bytes(3, "little", signed=True) for v in samples)
    block = 3
    fmt = struct.pack("<HHIIHH", 1, 1, 16000, 16000 * block, block, 24)
    data = (
        b"RIFF"
        + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(raw))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(raw))
        + raw
    )
    audio, rate = decode_wav(data)
    assert rate == 16000
    np.testing.assert_allclose(audio, [0.0, 0.5, -1.0], atol=1e-6)


def test_decode_float32():
    """The stdlib `wave` module rejects format 3; the importer must not."""
    samples = np.array([0.0, 0.5, -0.75, 1.5], dtype="<f4")
    audio, rate = decode_wav(float_wav_bytes(samples))
    assert rate == 16000
    np.testing.assert_allclose(audio, [0.0, 0.5, -0.75, 1.0], atol=1e-6)


def test_decode_float64():
    samples = np.array([0.0, 0.25, -1.0], dtype="<f8")
    audio, _ = decode_wav(float_wav_bytes(samples, width=8))
    np.testing.assert_allclose(audio, samples, atol=1e-6)


def test_decode_extensible_float():
    samples = np.array([0.0, -0.5], dtype="<f4")
    audio, _ = decode_wav(extensible_float_wav_bytes(samples))
    np.testing.assert_allclose(audio, samples)


def test_decode_float_stereo_downmix():
    interleaved = np.array([1.0, -1.0, 0.5, 0.5], dtype="<f4")
    audio, _ = decode_wav(float_wav_bytes(interleaved, channels=2))
    np.testing.assert_allclose(audio, [0.0, 0.5], atol=1e-6)


def test_rejects_non_riff():
    try:
        decode_wav(b"not a wav file")
    except ValueError as exc:
        assert "RIFF" in str(exc)
    else:
        raise AssertionError("expected ValueError")
