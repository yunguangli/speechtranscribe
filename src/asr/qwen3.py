"""Qwen3-ASR-0.6B (int4 ONNX) on CPU.

The export is three graphs plus an embedding table:

* ``encoder.int4.onnx``            mel -> audio features
* ``decoder_merged.int4.onnx``     KV-cache decoder over embeddings
* ``embed_tokens.bin``             fp16 [vocab, 1024], used by the caller

The decoder is a speech LLM: the prompt is a Qwen chat with the audio features
spliced in where ``<|audio_pad|>`` placeholders sit. Greedy decode until EOS.

The system turn precedes the audio, so its KV cache never changes between
utterances; we compute it once (a few tokens) and reuse it.

Adapted from the Apache-2.0 reference driver in
``OHF-Voice/wyoming-faster-whisper`` (``qwen3_asr_handler.py``).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .mel import SAMPLE_RATE, log_mel
from .text import normalize_text

MODEL_SUBDIR = Path("models") / "qwen3-asr-0.6b-int4"

_SYSTEM = 8948
_USER = 872
_ASSISTANT = 77091
_NEWLINE = 198

# Fallbacks for config.json files that predate the ``special_tokens`` block.
_DEFAULT_SPECIAL_TOKENS = {
    "im_start_token_id": 151644,
    "im_end_token_id": 151645,
    "audio_start_token_id": 151669,
    "audio_end_token_id": 151670,
    "audio_pad_token_id": 151676,
    "asr_text_token_id": 151704,
}
_DEFAULT_EOS_IDS = (151643, 151645)

_MASK_BLOCKED = np.finfo(np.float32).min

# Execution providers, in preference order, using the names that
# ``onnxruntime.get_available_providers()`` reports. NNAPI and XNNPACK are
# present in the Android/iOS wheels; on desktop only CPU is normally built in.
_PROVIDER_ALIASES = {
    "cpu": ("CPUExecutionProvider",),
    "xnnpack": ("XnnpackExecutionProvider", "CPUExecutionProvider"),
    "nnapi": ("NnapiExecutionProvider", "CPUExecutionProvider"),
    "coreml": ("CoreMLExecutionProvider", "CPUExecutionProvider"),
}


@dataclass(frozen=True, slots=True)
class SpecialTokens:
    im_start: int
    im_end: int
    audio_start: int
    audio_end: int
    audio_pad: int
    asr_text: int
    eos: frozenset[int]

    @classmethod
    def from_config(cls, config: dict) -> "SpecialTokens":
        known = config.get("special_tokens") or {}
        values: dict[str, int] = {}
        for name, default in _DEFAULT_SPECIAL_TOKENS.items():
            value = known.get(name)
            values[name] = default if value is None else int(value)
        eos = known.get("eos_token_ids") or _DEFAULT_EOS_IDS
        return cls(
            im_start=values["im_start_token_id"],
            im_end=values["im_end_token_id"],
            audio_start=values["audio_start_token_id"],
            audio_end=values["audio_end_token_id"],
            audio_pad=values["audio_pad_token_id"],
            asr_text=values["asr_text_token_id"],
            eos=frozenset(int(t) for t in eos),
        )


def _read_special_tokens(path: Path) -> SpecialTokens:
    """Read the prompt/stop tokens from ``config.json``.

    They are duplicated in the export and in this file historically; taking
    them from the config means a re-export cannot silently desynchronise the
    prompt from the weights.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            return SpecialTokens.from_config(json.load(fh))
    except (OSError, ValueError, KeyError, TypeError):
        return SpecialTokens.from_config({})


def _selected_providers() -> list[str]:
    """Execution providers for this process (``SPEECHTRANSCRIBE_EP``)."""
    requested = (os.environ.get("SPEECHTRANSCRIBE_EP") or "cpu").strip().lower()
    wanted = _PROVIDER_ALIASES.get(requested, _PROVIDER_ALIASES["cpu"])
    available = set(ort.get_available_providers())
    chosen = [name for name in wanted if name in available]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def _trim_silence(audio: np.ndarray) -> np.ndarray:
    """Drop leading/trailing non-speech so the encoder does not run on it.

    VAD segments carry ~320 ms of pre-roll and up to ~400 ms of trailing
    silence. On a phone that is a real slice of the compute budget, and long
    silent tails also invite hallucinated words.
    """
    window = 320  # 20 ms
    usable = audio.size // window
    if usable < 3:
        return audio

    frames = audio[: usable * window].reshape(usable, window)
    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1) + 1e-12)
    peak = float(np.max(np.abs(audio)))
    # A conservative threshold: quiet consonants at a chunk boundary must not be
    # mistaken for silence.
    threshold = max(peak * 0.008, 0.0015)
    voiced = np.flatnonzero(rms > threshold)
    if voiced.size == 0:
        return audio

    pad = 1600  # keep 100 ms around the speech
    start = max(0, voiced[0] * window - pad)
    end = min(audio.size, (voiced[-1] + 1) * window + pad)
    return audio[start:end]


def default_assets_dir() -> Path:
    """Assets directory, honouring the path `flet run` / `flet build` export."""
    env = os.environ.get("FLET_ASSETS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "assets"


@dataclass(slots=True)
class Transcript:
    text: str
    language: str | None = None
    audio_seconds: float = 0.0
    encode_seconds: float = 0.0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    cancelled: bool = False
    # Generated ids (language preamble included, EOS excluded). Re-used as a
    # draft for the next partial/final of the same segment.
    token_ids: list[int] = field(default_factory=list)

    @property
    def real_time_factor(self) -> float:
        total = self.encode_seconds + self.prefill_seconds + self.decode_seconds
        return total / self.audio_seconds if self.audio_seconds else 0.0


class Qwen3Asr:
    """Loads once, transcribes many utterances."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        threads: int | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else default_assets_dir() / MODEL_SUBDIR
        if threads is None:
            env = os.environ.get("SPEECHTRANSCRIBE_THREADS")
            if env and env.isdigit():
                threads = max(1, int(env))
        if threads is None:
            # int4 decode is memory-bandwidth-bound: on a 4-core box two threads
            # are as fast as four, and leave headroom for the UI/audio threads.
            threads = min(4, max(1, (os.cpu_count() or 4) // 2))
        self.threads = threads
        self.tokens = _read_special_tokens(self.model_dir / "config.json")
        # Filled in by `load`; useful for the UI/debug log.
        self.providers = _selected_providers()
        self._on_progress = on_progress or (lambda _msg: None)
        self._loaded = False
        self._encoder: ort.InferenceSession | None = None
        self._decoder: ort.InferenceSession | None = None
        self._embed: np.ndarray | None = None
        self._tokenizer: Tokenizer | None = None
        self._empty_kv: np.ndarray | None = None
        self._prefix_key: tuple[int, ...] | None = None
        self._prefix_kv: tuple[np.ndarray, np.ndarray] | None = None
        # A partial decode can be aborted from the UI thread when a final
        # result arrives; `terminate` is read by ONNX Runtime mid-run.
        self._run_options = ort.RunOptions()
        self._cancel_requested = False
        self._active = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def missing_files(self) -> list[str]:
        return [
            name
            for name in (
                "encoder.int4.onnx",
                "decoder_merged.int4.onnx",
                "embed_tokens.bin",
                "tokenizer.json",
                "config.json",
            )
            if not (self.model_dir / name).is_file()
        ]

    def load(self) -> None:
        if self._loaded:
            return

        missing = self.missing_files()
        if missing:
            raise FileNotFoundError(
                f"Model files missing from {self.model_dir}: {', '.join(missing)}. "
                "Run: python tools/fetch_weights.py"
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.providers = _selected_providers()

        self._on_progress("Loading encoder")
        self._encoder = self._session(self.model_dir / "encoder.int4.onnx", options)

        self._on_progress("Loading decoder")
        self._decoder = self._session(
            self.model_dir / "decoder_merged.int4.onnx", options
        )

        kv_shape = self._decoder.get_outputs()[1].shape
        self._empty_kv = np.zeros(
            (kv_shape[0], 1, kv_shape[2], 0, kv_shape[4]), dtype=np.float32
        )

        self._on_progress("Loading vocabulary")
        self._tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))

        with open(self.model_dir / "config.json", encoding="utf-8") as fh:
            config = json.load(fh)
        self.tokens = SpecialTokens.from_config(config)
        hidden = config["decoder"]["hidden_size"]
        # fp16 on disk, one row cast per token. A memmap keeps ~300 MB out of RSS.
        element_size = np.dtype(np.float16).itemsize
        rows = (self.model_dir / "embed_tokens.bin").stat().st_size // (
            hidden * element_size
        )
        self._embed = np.memmap(
            self.model_dir / "embed_tokens.bin",
            dtype=np.float16,
            mode="r",
            shape=(rows, hidden),
        )
        self._loaded = True
        self._on_progress("Ready")

    def close(self) -> None:
        self._encoder = None
        self._decoder = None
        self._tokenizer = None
        embed, self._embed = self._embed, None
        if embed is not None:
            mmap = getattr(embed, "_mmap", None)
            if mmap is not None:
                try:
                    mmap.close()
                except (BufferError, ValueError):
                    pass
        self._prefix_kv = None
        self._prefix_key = None
        self._loaded = False

    # -- inference ---------------------------------------------------------

    def _session(self, path: Path, options: ort.SessionOptions) -> ort.InferenceSession:
        """Create a session, falling back to CPU if the chosen EP refuses.

        ``SPEECHTRANSCRIBE_EP`` is an experiment knob; a phone whose NNAPI
        driver rejects the graph must not leave the app unable to start.
        """
        try:
            return ort.InferenceSession(
                str(path), sess_options=options, providers=self.providers
            )
        except Exception as exc:  # noqa: BLE001 - provider errors are opaque
            if self.providers == ["CPUExecutionProvider"]:
                raise
            self._on_progress(f"{self.providers[0]} unavailable ({exc}); using CPU")
            self.providers = ["CPUExecutionProvider"]
            return ort.InferenceSession(
                str(path), sess_options=options, providers=self.providers
            )

    def _system_ids(self) -> list[int]:
        return [
            self.tokens.im_start,
            _SYSTEM,
            _NEWLINE,
            self.tokens.im_end,
            _NEWLINE,
        ]

    def _language_ids(self, language: str | None) -> list[int]:
        if not language:
            return []
        return self._tokenizer.encode(
            f"language {language}", add_special_tokens=False
        ).ids + [self.tokens.asr_text]

    def _prompt_ids(self, n_audio: int, language: str | None) -> list[int]:
        return (
            self._system_ids()
            + [self.tokens.im_start, _USER, _NEWLINE, self.tokens.audio_start]
            + [self.tokens.audio_pad] * n_audio
            + [
                self.tokens.audio_end,
                self.tokens.im_end,
                _NEWLINE,
                self.tokens.im_start,
                _ASSISTANT,
                _NEWLINE,
            ]
            + self._language_ids(language)
        )

    @staticmethod
    def _causal_mask(q_len: int, past_len: int) -> np.ndarray:
        new = np.triu(np.full((q_len, q_len), _MASK_BLOCKED, dtype=np.float32), k=1)
        if past_len == 0:
            return new[np.newaxis, np.newaxis]
        past = np.zeros((q_len, past_len), dtype=np.float32)
        return np.concatenate([past, new], axis=-1)[np.newaxis, np.newaxis]

    def _run(
        self,
        embeds: np.ndarray,
        positions: Sequence[int] | np.ndarray,
        past_keys: np.ndarray,
        past_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._decoder.run(
            ["logits", "present_keys", "present_values"],
            {
                "input_embeds": embeds,
                "position_ids": np.asarray(positions, dtype=np.int64)[np.newaxis, :],
                "attention_mask": self._causal_mask(embeds.shape[1], past_keys.shape[3]),
                "past_keys": past_keys,
                "past_values": past_values,
            },
            self._run_options,
        )

    def _prefix(self) -> tuple[np.ndarray, np.ndarray]:
        ids = self._system_ids()
        key = tuple(ids)
        if self._prefix_key == key and self._prefix_kv is not None:
            return self._prefix_kv
        _, keys, values = self._run(
            self._embed[ids].astype(np.float32)[np.newaxis, :, :],
            np.arange(len(ids)),
            self._empty_kv,
            self._empty_kv,
        )
        self._prefix_key = key
        self._prefix_kv = (keys, values)
        return keys, values

    def cancel_current(self) -> None:
        """Abort the decode that is currently running (thread-safe)."""
        if self._active:
            self._cancel_requested = True
            self._run_options.terminate = True

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        max_new_tokens: int = 256,
        draft_ids: Sequence[int] | None = None,
    ) -> Transcript:
        """Transcribe float32 mono 16 kHz audio in [-1, 1]."""
        if not self._loaded:
            self.load()

        audio = _trim_silence(np.asarray(audio, dtype=np.float32))
        duration = audio.size / SAMPLE_RATE
        if audio.size < SAMPLE_RATE // 10:
            return Transcript(text="", language=language, audio_seconds=duration)

        # The reference preprocessing pads clips shorter than 0.5 s.
        if audio.size < SAMPLE_RATE // 2:
            audio = np.pad(audio, (0, SAMPLE_RATE // 2 - audio.size))

        # Nothing sane needs more tokens than this for the given duration.
        max_new_tokens = min(max_new_tokens, max(64, int(duration * 12) + 24))

        self._cancel_requested = False
        self._run_options.terminate = False
        self._active = True
        try:
            return self._transcribe(
                audio, language, max_new_tokens, duration, draft_ids
            )
        except Exception:
            if self._cancel_requested:
                return Transcript(
                    text="",
                    language=language,
                    audio_seconds=duration,
                    cancelled=True,
                )
            raise
        finally:
            self._active = False

    @staticmethod
    def _accept_draft(draft: Sequence[int], preds: np.ndarray) -> int:
        """How many draft tokens the verifier confirms (at least one)."""
        for i in range(len(draft) - 1):
            if int(preds[i]) != int(draft[i + 1]):
                return i + 1
        return len(draft)

    def _transcribe(
        self,
        audio: np.ndarray,
        language: str | None,
        max_new_tokens: int,
        duration: float,
        draft_ids: Sequence[int] | None = None,
    ) -> Transcript:
        started = time.perf_counter()
        mel = log_mel(audio)
        features = self._encoder.run(None, {"mel": mel}, self._run_options)[0]
        encode_seconds = time.perf_counter() - started
        if self._cancel_requested:
            return Transcript(
                text="", language=language, audio_seconds=duration, cancelled=True
            )

        prompt = self._prompt_ids(features.shape[1], language)
        n_system = len(self._system_ids())
        past_keys, past_values = self._prefix()

        rest = prompt[n_system:]
        embeds = self._embed[rest].astype(np.float32)[np.newaxis, :, :]
        audio_at = prompt.index(self.tokens.audio_pad) - n_system
        embeds[0, audio_at : audio_at + features.shape[1]] = features[0]

        prefill_started = time.perf_counter()
        logits, past_keys, past_values = self._run(
            embeds, np.arange(n_system, len(prompt)), past_keys, past_values
        )
        prefill_seconds = time.perf_counter() - prefill_started

        decode_started = time.perf_counter()
        token = int(np.argmax(logits[0, -1, :]))
        tokens: list[int] = [token]
        position = len(prompt)

        # A previous decode of this segment is a strong draft: verify it in one
        # batched forward instead of re-generating it token by token. Accepted
        # tokens are exactly what greedy decoding would have produced.
        draft = [int(t) for t in (draft_ids or ())][: max(0, max_new_tokens - 1)]
        if draft and draft[0] == token:
            verified, vk, vv = self._run(
                self._embed[draft].astype(np.float32)[np.newaxis, :, :],
                np.arange(position, position + len(draft)),
                past_keys,
                past_values,
            )
            if self._cancel_requested:
                return Transcript(
                    text="", language=language, audio_seconds=duration, cancelled=True
                )
            preds = np.argmax(verified[0], axis=-1)
            confirmed = self._accept_draft(draft, preds)
            token = int(preds[confirmed - 1])
            tokens = draft[:confirmed] + [token]
            keep = position + confirmed
            # The verifier also cached the rejected draft tokens; drop them.
            past_keys = np.ascontiguousarray(vk[:, :, :, :keep])
            past_values = np.ascontiguousarray(vv[:, :, :, :keep])
            position = keep

        while token not in self.tokens.eos and len(tokens) < max_new_tokens:
            if self._cancel_requested:
                return Transcript(
                    text="", language=language, audio_seconds=duration, cancelled=True
                )
            logits, past_keys, past_values = self._run(
                self._embed[token].astype(np.float32)[None, None, :],
                [position],
                past_keys,
                past_values,
            )
            token = int(np.argmax(logits[0, -1, :]))
            tokens.append(token)
            position += 1
        decode_seconds = time.perf_counter() - decode_started

        detected: str | None = language or None
        spoken = tokens
        if self.tokens.asr_text in tokens:
            split = tokens.index(self.tokens.asr_text)
            preamble = self._tokenizer.decode(
                tokens[:split], skip_special_tokens=True
            ).strip()
            if preamble.lower().startswith("language "):
                preamble = preamble[len("language ") :].strip()
            if preamble:
                detected = preamble
            spoken = tokens[split + 1 :]

        text = self._tokenizer.decode(spoken, skip_special_tokens=True)
        return Transcript(
            text=normalize_text(text),
            language=detected,
            audio_seconds=duration,
            encode_seconds=encode_seconds,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            token_ids=[t for t in tokens if t not in self.tokens.eos],
        )
