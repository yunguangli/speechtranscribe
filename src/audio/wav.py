"""WAV decoding and PCM helpers (stdlib only).

Only WAV import is supported on the phone, where ffmpeg does not exist. The
decoder parses RIFF itself because the stdlib ``wave`` module rejects float
PCM (``unknown format: 3``), which every DAW and most phone recorders can emit.

Supported: 8/16/24/32-bit integer PCM and 32/64-bit IEEE float, plain or
``WAVE_FORMAT_EXTENSIBLE``, mono or multi-channel (mixed down to mono).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

_RIFF = b"RIFF"
_WAVE = b"WAVE"

_FORMAT_PCM = 0x0001
_FORMAT_FLOAT = 0x0003
_FORMAT_EXTENSIBLE = 0xFFFE


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Raw little-endian PCM16 bytes (what AudioRecorder streams) -> [-1, 1]."""
    if not data:
        return np.zeros(0, dtype=np.float32)
    usable = len(data) - (len(data) % 2)
    return np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0


@dataclass(frozen=True, slots=True)
class _Format:
    tag: int
    channels: int
    rate: int
    block_align: int
    bits: int


def _parse_fmt(chunk: bytes) -> _Format:
    if len(chunk) < 16:
        raise ValueError("truncated WAV fmt chunk")
    tag, channels, rate, _byte_rate, block_align, bits = struct.unpack(
        "<HHIIHH", chunk[:16]
    )
    if tag == _FORMAT_EXTENSIBLE:
        # cbSize at 16, validBits at 18, channelMask at 20, then a 16-byte
        # GUID whose first two bytes are the real format tag.
        if len(chunk) < 26:
            raise ValueError("truncated WAVE_FORMAT_EXTENSIBLE header")
        tag = struct.unpack("<H", chunk[24:26])[0]
    if channels < 1 or rate < 1 or bits < 1:
        raise ValueError("invalid WAV format")
    width = bits // 8
    if block_align <= 0:
        block_align = channels * width
    return _Format(tag, channels, int(rate), int(block_align), int(bits))


def _split_chunks(data: bytes) -> tuple[_Format, bytes]:
    view = memoryview(data)
    if len(view) < 12 or bytes(view[:4]) != _RIFF or bytes(view[8:12]) != _WAVE:
        raise ValueError("not a RIFF/WAVE file")

    fmt: _Format | None = None
    payload: bytes | None = None
    pos = 12
    while pos + 8 <= len(view):
        chunk_id = bytes(view[pos : pos + 4])
        size = struct.unpack("<I", view[pos + 4 : pos + 8])[0]
        start = pos + 8
        end = min(start + size, len(view))
        if chunk_id == b"fmt " and fmt is None:
            fmt = _parse_fmt(bytes(view[start:end]))
        elif chunk_id == b"data" and payload is None:
            payload = bytes(view[start:end])
        if fmt is not None and payload is not None:
            break
        # Chunks are word-aligned: an odd size is followed by one pad byte.
        pos = start + size + (size & 1)

    if fmt is None:
        raise ValueError("WAV file has no fmt chunk")
    if payload is None:
        raise ValueError("WAV file has no data chunk")
    return fmt, payload


def _decode_pcm(fmt: _Format, raw: bytes) -> np.ndarray:
    width = fmt.bits // 8
    if fmt.tag == _FORMAT_FLOAT:
        if width == 4:
            audio = np.frombuffer(raw, dtype="<f4")
        elif width == 8:
            audio = np.frombuffer(raw, dtype="<f8").astype(np.float32)
        else:
            raise ValueError(f"Unsupported float WAV width: {fmt.bits} bits")
        return np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)

    if fmt.tag != _FORMAT_PCM:
        raise ValueError(f"Unsupported WAV format tag: 0x{fmt.tag:04x}")

    if width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        audio = ints.astype(np.float32) / 8388608.0
    elif width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {fmt.bits} bits")
    return audio


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """WAV bytes -> (mono float32 in [-1, 1], sample rate)."""
    fmt, raw = _split_chunks(data)

    width = fmt.bits // 8
    if fmt.block_align != fmt.channels * width:
        raise ValueError("unsupported WAV block alignment")
    frame_bytes = fmt.block_align
    raw = raw[: len(raw) - len(raw) % frame_bytes]

    audio = _decode_pcm(fmt, raw)
    if fmt.channels > 1:
        audio = audio.reshape(-1, fmt.channels).mean(axis=1)

    return audio.astype(np.float32), fmt.rate
