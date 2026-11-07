# SpeechTranscribe

On-device speech-to-text with [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
(int4 ONNX). Records from the microphone, segments speech with Silero VAD, and
prints text while you talk. 30+ languages, auto-detected or pinned.

No audio leaves the device.

## 1. Fetch the weights (~786 MB)

```bash
uv run python tools/fetch_weights.py
```

Downloads into `src/assets/models/qwen3-asr-0.6b-int4/` and verifies the
checksums. The folder is git-ignored.

## 2. Run

```bash
uv run flet run            # desktop
uv run flet run --web      # web (dynamic; Python runs on the server)
```

Static web builds (`flet build web`) cannot run the model — use desktop,
mobile, or dynamic web.

## 3. Build the APK

```bash
flet build apk -v
```

Android: `arm64-v8a` only (no int4 ONNX Runtime wheel for `armeabi-v7a`), and
the microphone permissions are already declared in `pyproject.toml`.

## What it does

- **Record** — streams 16 kHz PCM16, VAD cuts at ~400 ms silence. Continuous
  speech is chunked every ~5 s at the quietest point in the last second, so a
  fast talker still gets text while talking instead of waiting for the whole
  monologue. Partial transcripts refresh about once a second. Chunks separated
  by a short pause stay in the same paragraph; a new row starts only after
  roughly 1.5 s of silence.
- **Audio file** — import a `.wav` (8/16/24/32-bit integer or 32/64-bit float,
  mono or stereo) and transcribe it. Long files are split at pauses with the
  same logic as live audio, so memory stays flat and nothing is truncated.
- **Language** — auto-detect or pin one of the 30 supported names. The language
  detected at the start of a paragraph is reused for the rest of it, which
  stabilises short chunks.
- **Select** — the whole transcript sits in a `SelectionArea`; drag across
  paragraphs to copy. The list follows the newest text, pauses if you scroll
  up, and resumes when you return to the bottom.
- **Save** — copy all, or save `.txt` / `.srt` (one cue per chunk, with its
  position in the recording). Saving is enabled only once recording has
  stopped and the queue has finished; every save contains the whole session in
  a single file.

## Layout

```
src/asr/        mel front end, prompt, ONNX decoder
src/audio/      resampling, WAV decoding, VAD, segmenter
src/app.py      UI, recording loop, worker, export
src/assets/     Silero VAD + icons + model weights
tools/          weight fetcher, CLI benchmark
tests/          mel, prompt, segmenter, WAV, export
```

## Performance

- Long monologues are cut at a low-energy 100 ms window in the last ~1.2 s,
  preferring the latest quiet window, not on a clock boundary. Measured on
  fast English speech: 100 % of words kept vs 97 % for a hard 5 s cut. The
  tail carries into the next chunk, so no audio is dropped.
- Each partial decode is reused as a speculative draft for the final. The
  draft is verified in one batched forward and the output is token-identical
  to plain greedy decoding. Measured: ~25 % off a 5 s final, ~45 % off a
  14 s one.
- A final result cancels the partial that is already decoding and drops queued
  partials, so pausing produces the real transcript immediately.
- Partials stop at ~70 % of the chunk, because a final is about to land.
- Leading/trailing silence is trimmed before the encoder, and the token budget
  follows the audio duration. Short clips are padded to 0.5 s like the
  reference preprocessing.
- Imported files are split at pauses too (15 s cap by default), so a long
  recording neither grows encoder memory with its length nor gets cut off by
  the token budget. Each chunk repeats the short prompt prefill, which costs a
  few percent of compute; in exchange the transcript stays complete at any
  length.
- ONNX Runtime uses half the cores (capped at 4; two on this 4-core box).
  Override with `SPEECHTRANSCRIBE_THREADS=4`.
- Pick an execution provider with `SPEECHTRANSCRIBE_EP`: `cpu` (default),
  `xnnpack`, `nnapi`, or `coreml`. Unavailable providers fall back to CPU; the
  chosen list is printed by `tools/bench.py`. On a phone, try `xnnpack` and
  `nnapi` — they are compiled into the Flet Android/iOS wheels.
- Tune the chunk length with `SPEECHTRANSCRIBE_MAX_SECONDS=4` (2-15, default 5)
  and log per-decode timings with `SPEECHTRANSCRIBE_DEBUG=1`.
- Accuracy: int4 is the 0.6B model's trade-off. Pin the language when you know
  it, speak in phrases rather than single words, and keep the phone mic close.

## Benchmark

```bash
uv run python tools/bench.py clip.wav --language Chinese
```

On a 4-core CPU without a GPU: ~0.8x realtime (a 15 s clip takes ~12 s).
With the live segmenter the same audio is delivered in ~5 s chunks, and the
first text arrives after a few seconds instead of at the end.
