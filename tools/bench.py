#!/usr/bin/env python3
"""Transcribe WAV files from the command line.

    python tools/bench.py /tmp/asr_zh.wav
    python tools/bench.py clip1.wav clip2.wav --language Chinese

Prints the transcript, the detected language, and encode/prefill/decode
timings so the real-time budget is visible without opening the UI. Long files
are split at pauses exactly like the app does; pass ``--whole`` to feed the
whole file to the model in one pass instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr import SAMPLE_RATE, Qwen3Asr  # noqa: E402
from audio.resample import resample  # noqa: E402
from audio.segmenter import split_at_pauses  # noqa: E402
from audio.wav import decode_wav  # noqa: E402

DEFAULT_MAX_SECONDS = 5.0


def read(path: Path) -> tuple[np.ndarray, int]:
    if path.suffix.lower() != ".wav":
        raise SystemExit(f"{path}: only .wav is supported")
    return decode_wav(path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--language", default=None, help="force a language, e.g. Chinese")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--whole",
        action="store_true",
        help="do not split at pauses (raw model timing; can truncate long files)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help=f"chunk length when splitting (default {DEFAULT_MAX_SECONDS:g})",
    )
    args = parser.parse_args()

    engine = Qwen3Asr(threads=args.threads, on_progress=lambda m: print(f"  {m}"))
    engine.load()
    print(f"  providers: {', '.join(engine.providers)}")

    for path in args.files:
        audio, rate = read(path)
        if rate != SAMPLE_RATE:
            audio = resample(audio, rate, SAMPLE_RATE)

        if args.whole:
            chunks = [(0.0, audio, audio.size / SAMPLE_RATE)]
        else:
            chunks = split_at_pauses(audio, args.max_seconds)

        encode = prefill = decode = 0.0
        texts: list[str] = []
        detected: str | None = args.language
        for _start, chunk, _seconds in chunks:
            result = engine.transcribe(chunk, language=args.language)
            encode += result.encode_seconds
            prefill += result.prefill_seconds
            decode += result.decode_seconds
            detected = detected or result.language
            if result.text:
                texts.append(result.text)

        audio_seconds = audio.size / SAMPLE_RATE
        total = encode + prefill + decode
        rtf = total / audio_seconds if audio_seconds else 0.0
        print(
            f"\n{path.name}: {audio_seconds:.1f}s audio, {total:.2f}s compute "
            f"({rtf:.2f}x realtime), {len(chunks)} chunk(s), "
            f"lang={detected or '?'}"
        )
        print(
            f"  encode {encode:.2f}s  prefill {prefill:.2f}s  "
            f"decode {decode:.2f}s"
        )
        print(f"  {' '.join(texts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
