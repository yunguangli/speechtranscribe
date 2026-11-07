"""SpeechTranscribe UI.

Recording is a stream of raw PCM16 chunks. A VAD turns them into utterances;
each utterance is transcribed in a worker thread (ONNX Runtime is CPU-bound),
with a partial decode while you talk and a speculative draft for the final.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque

import flet as ft
import flet_audio_recorder as far
import numpy as np

from asr.languages import LANGUAGES, badge
from asr.text import normalize_text
from audio import Segmenter, decode_wav, resample, split_at_pauses

RECORD_RATE = 16_000
MAX_UTTERANCES = 200
# Pause between segments before a new paragraph starts. The segmenter already
# burns ~400 ms of trailing silence and ~320 ms of pre-roll, so this is roughly
# a 1.5 s pause in the real audio.
PARAGRAPH_GAP_SECONDS = 0.8
# Imported files are fed to the segmenter in 1 s blocks so its internal buffer
# stays bounded regardless of file length.
IMPORT_BLOCK_SECONDS = 1.0
# Live chunks are short so text appears while talking. A file has no such
# deadline: longer chunks mean fewer encoder passes and fewer prefills, and
# still stay inside the 15 s ceiling the encoder is known to handle.
IMPORT_MAX_SECONDS = 15.0


def _max_segment_seconds() -> float:
    """Long continuous speech is chunked; smaller is faster but cuts more."""
    raw = os.environ.get("SPEECHTRANSCRIBE_MAX_SECONDS", "5")
    try:
        value = float(raw)
    except ValueError:
        return 5.0
    return min(15.0, max(2.0, value))


def split_imported_audio(
    audio: np.ndarray,
    max_seconds: float,
    *,
    rate: int = RECORD_RATE,
) -> list[tuple[float, np.ndarray, float]]:
    """Cut imported audio at pauses, like the live path, with timestamps."""
    return split_at_pauses(audio, max_seconds, rate=rate, block_seconds=IMPORT_BLOCK_SECONDS)


@dataclass
class Utterance:
    id: int
    text: str = ""
    language: str | None = None
    final: bool = False
    seconds: float = 0.0
    start: float = 0.0
    # Segment start -> transcript for that segment. Segments are merged into
    # one paragraph (no line break) until a long pause closes it.
    segments: dict[float, str] = field(default_factory=dict)
    controls: dict[str, ft.Control] = field(default_factory=dict)


@dataclass
class Job:
    utterance_id: int
    audio: np.ndarray
    final: bool
    seconds: float
    segment_start: float = 0.0


class SpeechApp:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self.engine = None
        self.model_ready = False
        self.static_web = bool(getattr(page, "pyodide", False))

        self.utterances: dict[int, Utterance] = {}
        self.order: list[int] = []
        self._next_id = 0
        self._jobs: Deque[Job] = Deque()
        self._wake = asyncio.Event()
        self._current_job: Job | None = None

        self.recording = False
        self.transcribing = False
        self.recording_started = 0.0
        self._timer_task = None
        self._language_name = ""
        self._paragraph_uid: int | None = None
        self._last_segment_end = 0.0
        # Previous decode of the same segment, re-used as a speculative draft.
        self._drafts: dict[tuple[int, float, str], list[int]] = {}
        # Auto-detected language is pinned for the rest of the paragraph.
        self._sticky_uid: int | None = None
        self._sticky_language: str | None = None
        self._debug = os.environ.get("SPEECHTRANSCRIBE_DEBUG") == "1"
        # Services register with the page when constructed; build them once
        # instead of on every copy/save/import.
        self._clipboard_service: ft.Clipboard | None = None
        self._picker_service: ft.FilePicker | None = None

        self.segmenter = Segmenter(
            RECORD_RATE,
            on_partial=self._on_partial,
            on_final=self._on_final,
            max_seconds=_max_segment_seconds(),
        )
        self.recorder = far.AudioRecorder(on_stream=self._on_stream)

        self._build_ui()
        page.run_task(self._worker)
        page.run_task(self._load_model)
        page.on_disconnect = self._on_page_closed
        page.on_close = self._on_page_closed

    def _on_page_closed(self, _e=None) -> None:
        """Release the ONNX sessions when the page/window goes away."""
        self.recording = False
        if self.engine is not None and not self.transcribing:
            try:
                self.engine.close()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                pass

    def _clipboard(self) -> ft.Clipboard:
        service = getattr(self, "_clipboard_service", None)
        if service is None:
            service = ft.Clipboard()
            self._clipboard_service = service
        return service

    def _file_picker(self) -> ft.FilePicker:
        service = getattr(self, "_picker_service", None)
        if service is None:
            service = ft.FilePicker()
            self._picker_service = service
        return service

    # -- UI ----------------------------------------------------------------

    def _build_ui(self) -> None:
        page = self.page
        page.title = "SpeechTranscribe"
        page.theme_mode = ft.ThemeMode.SYSTEM
        page.theme = ft.Theme(color_scheme_seed="#2563EB", use_material3=True)
        page.dark_theme = ft.Theme(color_scheme_seed="#2563EB", use_material3=True)
        page.padding = 0
        if not page.web:
            page.window.width = 760
            page.window.height = 900
            page.window.min_width = 360
            page.window.min_height = 520

        self.language_dd = ft.Dropdown(
            value="",
            options=[ft.DropdownOption(key=name, text=label) for label, name in LANGUAGES],
            dense=True,
            text_size=13,
            on_select=self._on_language,
            width=190,
        )

        self.status_dot = ft.Container(
            width=8, height=8, border_radius=4, bgcolor=ft.Colors.OUTLINE
        )
        self.status_text = ft.Text("Loading model…", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.timer_text = ft.Text("0:00", size=12, color=ft.Colors.ON_SURFACE_VARIANT)

        self.copy_btn = ft.IconButton(
            ft.Icons.CONTENT_COPY, icon_size=18, tooltip="Copy the whole transcript",
            on_click=self._copy_all,
        )
        self.clear_btn = ft.IconButton(
            ft.Icons.DELETE_SWEEP, icon_size=18, tooltip="Clear transcript",
            on_click=self._clear_all,
        )
        self.save_btn = ft.PopupMenuButton(
            icon=ft.Icons.SAVE_ALT,
            icon_size=18,
            disabled=True,
            tooltip="Save the whole session transcript",
            items=[
                ft.PopupMenuItem(
                    content=ft.Text("Save as text (.txt)"),
                    icon=ft.Icons.TEXT_SNIPPET,
                    on_click=lambda _e: self.page.run_task(self._save, "txt"),
                ),
                ft.PopupMenuItem(
                    content=ft.Text("Save as subtitles (.srt)"),
                    icon=ft.Icons.SUBTITLES,
                    on_click=lambda _e: self.page.run_task(self._save, "srt"),
                ),
            ],
        )

        self.empty_hint = ft.Column(
            [
                ft.Icon(ft.Icons.GRAPHIC_EQ, size=44, color=ft.Colors.with_opacity(0.35, ft.Colors.ON_SURFACE)),
                ft.Text("Press record and start talking", size=15,
                        color=ft.Colors.with_opacity(0.55, ft.Colors.ON_SURFACE)),
                ft.Text("30+ languages, auto-detected", size=12,
                        color=ft.Colors.with_opacity(0.35, ft.Colors.ON_SURFACE)),
            ],
            spacing=8,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self.list_view = ft.ListView(
            expand=True,
            spacing=10,
            padding=ft.Padding.symmetric(horizontal=16, vertical=0),
            auto_scroll=True,
            auto_scroll_animation=ft.Animation(
                duration=180, curve=ft.AnimationCurve.EASE_OUT
            ),
        )
        # SelectionArea makes the whole transcript selectable across rows, not
        # just within a single Text control.
        self.selection = ft.SelectionArea(content=self.list_view, expand=True)
        self.empty_container = ft.Container(
            content=self.empty_hint,
            expand=True,
            alignment=ft.Alignment.CENTER,
        )

        self.record_btn = ft.FilledButton(
            content=ft.Row(
                [ft.Icon(ft.Icons.MIC), ft.Text("Record", size=16, weight=ft.FontWeight.W_600)],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=8,
            ),
            height=54,
            expand=True,
            disabled=True,
            on_click=self._toggle_recording,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=16)),
        )

        self.import_btn = ft.OutlinedButton(
            content=ft.Row([ft.Icon(ft.Icons.UPLOAD_FILE, size=18), ft.Text("Audio file")], spacing=6),
            height=54,
            on_click=self._import_file,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=16)),
        )

        header = ft.Container(
            content=ft.Row(
                [
                    ft.Column(
                        [
                            ft.Row(
                                [
                                    self.status_dot,
                                    self.status_text,
                                    ft.Container(expand=True),
                                    self.timer_text,
                                ],
                                spacing=8,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                        ],
                        expand=True,
                        spacing=0,
                    ),
                    self.language_dd,
                ],
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding.only(left=20, right=16, top=8, bottom=12),
        )

        body = ft.Column(
            [
                header,
                ft.Container(
                    content=ft.Stack(
                        [self.selection, self.empty_container],
                        expand=True,
                    ),
                    expand=True,
                    padding=ft.Padding.symmetric(horizontal=16, vertical=0),
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Row(
                                [self.record_btn, self.import_btn],
                                spacing=10,
                            ),
                            ft.Row(
                                [
                                    ft.Text(
                                        "Int4 model · runs on this device",
                                        size=10,
                                        color=ft.Colors.with_opacity(0.35, ft.Colors.ON_SURFACE),
                                    ),
                                    ft.Container(expand=True),
                                    self.copy_btn,
                                    self.save_btn,
                                    self.clear_btn,
                                ],
                                spacing=2,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                        ],
                        spacing=6,
                    ),
                    padding=ft.Padding.only(left=16, right=16, top=10, bottom=10),
                ),
            ],
            expand=True,
            spacing=0,
        )

        page.add(
            ft.SafeArea(
                content=ft.Container(
                    content=ft.Container(content=body, expand=True, width=820),
                    expand=True,
                    alignment=ft.Alignment.TOP_CENTER,
                ),
                expand=True,
            )
        )
        self._show_empty(True)

    def _show_empty(self, show: bool) -> None:
        self.empty_container.visible = show
        self.selection.visible = not show

    def _update_save_enabled(self) -> None:
        """Saving is only allowed once the whole session has been transcribed."""
        has_text = any(self.utterances[uid].text.strip() for uid in self.order)
        busy = self.recording or self.transcribing or bool(self._jobs)
        self.save_btn.disabled = busy or not has_text

    def _set_status(self, text: str, color: str | None = None) -> None:
        self.status_text.value = text
        self.status_dot.bgcolor = color or ft.Colors.OUTLINE
        self.page.update()

    def _snack(self, message: str) -> None:
        self.page.show_dialog(ft.SnackBar(ft.Text(message), duration=ft.Duration(seconds=3)))

    # -- model -------------------------------------------------------------

    async def _load_model(self) -> None:
        if self.static_web:
            self._set_status(
                "Static web build cannot run the model — use desktop, mobile, "
                "or `flet run --web` (dynamic) instead.",
                ft.Colors.ERROR,
            )
            return

        try:
            from asr import Qwen3Asr

            self.engine = Qwen3Asr()
            await asyncio.to_thread(self.engine.load)
            self.model_ready = True
            self.record_btn.disabled = False
            self._set_status("Ready", ft.Colors.PRIMARY)
        except Exception as exc:  # noqa: BLE001 - surface it instead of dying
            self._set_status("Model unavailable", ft.Colors.ERROR)
            self._snack(str(exc))

    # -- recording ---------------------------------------------------------

    async def _toggle_recording(self, _e: ft.Event[ft.FilledButton]) -> None:
        if self.recording:
            await self._stop_recording()
        else:
            await self._start_recording()

    async def _start_recording(self) -> None:
        try:
            if not await self.recorder.has_permission():
                self._snack("Microphone permission is required.")
                return
        except Exception as exc:  # noqa: BLE001
            self._snack(f"Microphone unavailable: {exc}")
            return

        self.segmenter.reset()
        self.recording = True
        self.transcribing = False
        self.recording_started = time.monotonic()
        self._paragraph_uid = None
        self._last_segment_end = 0.0
        self._drafts.clear()
        self._sticky_uid = None
        self._sticky_language = None
        self._update_save_enabled()
        self.record_btn.content.controls[0].icon = ft.Icons.STOP
        self.record_btn.content.controls[1].value = "Stop"
        self.record_btn.content.controls[0].color = ft.Colors.ON_ERROR
        self.record_btn.content.controls[1].color = ft.Colors.ON_ERROR
        self.record_btn.bgcolor = ft.Colors.ERROR
        self.import_btn.disabled = True
        self._set_status("Listening…", ft.Colors.RED_400)

        try:
            await self.recorder.start_recording(
                configuration=far.AudioRecorderConfiguration(
                    encoder=far.AudioEncoder.PCM16BITS,
                    sample_rate=RECORD_RATE,
                    channels=1,
                )
            )
        except Exception as exc:  # noqa: BLE001
            self._snack(f"Could not start recording: {exc}")
            await self._reset_record_button()
            return

        self._timer_task = self.page.run_task(self._tick)

    async def _stop_recording(self) -> None:
        self.recording = False
        try:
            await self.recorder.stop_recording()
        except Exception:  # noqa: BLE001
            pass
        self.segmenter.flush()
        await self._reset_record_button()
        self._update_save_enabled()
        if self._jobs:
            self._set_status("Transcribing…", ft.Colors.AMBER)
        else:
            self._set_status(
                "Ready" if self.model_ready else "Loading model…",
                ft.Colors.PRIMARY if self.model_ready else None,
            )

    async def _reset_record_button(self) -> None:
        self.record_btn.content.controls[0].icon = ft.Icons.MIC
        self.record_btn.content.controls[1].value = "Record"
        self.record_btn.content.controls[0].color = None
        self.record_btn.content.controls[1].color = None
        self.record_btn.bgcolor = None
        self.import_btn.disabled = False
        self.page.update()

    async def _tick(self) -> None:
        while self.recording:
            elapsed = int(time.monotonic() - self.recording_started)
            self.timer_text.value = f"{elapsed // 60}:{elapsed % 60:02d}"
            self.page.update()
            await asyncio.sleep(0.25)

    # -- audio stream ------------------------------------------------------

    def _on_stream(self, e: far.AudioRecorderStreamEvent) -> None:
        if not self.recording:
            return
        samples = np.frombuffer(e.chunk, dtype="<i2").astype(np.float32) / 32768.0
        self.segmenter.feed(samples)

    def _on_partial(self, audio: np.ndarray, seconds: float, start: float) -> None:
        uid = self._paragraph_for(start)
        self._submit(Job(uid, audio, final=False, seconds=seconds, segment_start=start))

    def _on_final(self, audio: np.ndarray, seconds: float, start: float) -> None:
        uid = self._paragraph_for(start)
        self._last_segment_end = max(self._last_segment_end, start + seconds)
        self._submit(Job(uid, audio, final=True, seconds=seconds, segment_start=start))

    def _paragraph_for(self, segment_start: float) -> int:
        """Reuse the open paragraph unless the pause was long enough to split."""
        uid = self._paragraph_uid
        if (
            uid is not None
            and uid in self.utterances
            and segment_start - self._last_segment_end <= PARAGRAPH_GAP_SECONDS
        ):
            return uid
        self._paragraph_uid = self._new_utterance(start=segment_start)
        self._last_segment_end = segment_start
        self._sticky_uid = self._paragraph_uid
        self._sticky_language = None
        return self._paragraph_uid

    # -- language / drafts -------------------------------------------------

    def _language_for(self, job: Job) -> str | None:
        """Pinned language, else the language detected earlier in this paragraph."""
        if self._language_name:
            return self._language_name
        if self._sticky_uid == job.utterance_id:
            return self._sticky_language
        return None

    def _draft_for(self, job: Job, language: str | None) -> list[int] | None:
        return self._drafts.get((job.utterance_id, job.segment_start, language or ""))

    def _remember_result(self, job: Job, language: str | None, result) -> None:
        """Keep a partial's ids as the next draft; drop them once final."""
        if job.final:
            prefix = (job.utterance_id, job.segment_start)
            for key in [k for k in self._drafts if k[:2] == prefix]:
                del self._drafts[key]
        elif result.text and result.token_ids:
            key = (job.utterance_id, job.segment_start, language or "")
            self._drafts[key] = list(result.token_ids)

    # -- transcription worker ---------------------------------------------

    def _submit(self, job: Job) -> None:
        # A running partial for this paragraph is wasted work once a final is
        # queued; abort it so the real transcript lands sooner.
        current = self._current_job
        if (
            job.final
            and current is not None
            and not current.final
            and current.utterance_id == job.utterance_id
            and self.engine is not None
        ):
            try:
                self.engine.cancel_current()
            except Exception:  # noqa: BLE001 - cancellation is best effort
                pass

        # Partial results are disposable: drop queued partials for this
        # paragraph. A final supersedes them, and a newer partial supersedes
        # an older one. Finals are never dropped.
        self._jobs = type(self._jobs)(
            j for j in self._jobs if j.final or j.utterance_id != job.utterance_id
        )
        self._jobs.append(job)
        self._wake.set()

    async def _worker(self) -> None:
        while True:
            if not self._jobs:
                self._wake.clear()
                await self._wake.wait()
                continue

            job = self._jobs.popleft()
            if not job.final and any(
                j.final and j.utterance_id == job.utterance_id for j in self._jobs
            ):
                # A final for this paragraph is already waiting; decoding the
                # partial first would only delay the real result.
                continue
            self.transcribing = True
            self._current_job = job
            self._update_save_enabled()
            self.page.update()
            try:
                if self.engine is None:
                    continue
                language = self._language_for(job)
                draft = self._draft_for(job, language)
                try:
                    started = time.monotonic()
                    result = await asyncio.to_thread(
                        self.engine.transcribe,
                        job.audio,
                        language,
                        draft_ids=draft,
                    )
                    elapsed = time.monotonic() - started
                except Exception as exc:  # noqa: BLE001
                    self._set_status(f"Transcription failed: {exc}", ft.Colors.ERROR)
                    continue

                if result.cancelled:
                    continue

                self._remember_result(job, language, result)
                if self._debug:
                    print(
                        f"[asr] {'final' if job.final else 'partial'} "
                        f"{result.audio_seconds:.2f}s "
                        f"enc={result.encode_seconds:.2f} "
                        f"pre={result.prefill_seconds:.2f} "
                        f"dec={result.decode_seconds:.2f} "
                        f"tok={len(result.token_ids)} lang={result.language}",
                        flush=True,
                    )

                if not job.final and result.text:
                    # Slow phone: don't queue partials faster than they decode.
                    self.segmenter.partial_seconds = min(
                        4.0, max(1.2, elapsed * 1.5)
                    )

                if result.text or job.final:
                    self._apply(job, result.text, result.language, job.seconds)
                if job.final and not self.recording and not self._jobs:
                    self._set_status("Ready", ft.Colors.PRIMARY)
            finally:
                self._current_job = None
                self.transcribing = False
                self._update_save_enabled()
                self.page.update()

    def _apply(self, job: Job, text: str, language: str | None, seconds: float) -> None:
        utt = self.utterances.get(job.utterance_id)
        if utt is None:
            utt = self.utterances[self._new_utterance(start=job.segment_start)]

        if text:
            utt.segments[job.segment_start] = text
        else:
            utt.segments.pop(job.segment_start, None)

        utt.text = normalize_text(
            " ".join(part for _start, part in sorted(utt.segments.items()) if part)
        )
        if text:
            utt.language = language
        utt.final = job.final
        if (
            job.final
            and text
            and language
            and seconds >= 2.0
            and not self._language_name
            and self._sticky_uid == job.utterance_id
            and self._sticky_language is None
        ):
            # Short, fast chunks misdetect more; keep the first good guess.
            self._sticky_language = language
        utt.seconds = max(
            utt.seconds, (job.segment_start - utt.start) + seconds
        )
        if not utt.text and job.final:
            # Keep the row if more audio for it is still queued: the empty
            # chunk may just be a pause inside a long import or paragraph.
            pending = any(j.utterance_id == utt.id for j in self._jobs)
            if not pending:
                self._drop_utterance(utt)
                self._show_empty(not self.order)
                self.page.update()
                return
        self._render_utterance(utt)
        self._show_empty(False)
        self.page.update()

    def _drop_utterance(self, utt: Utterance) -> None:
        row = utt.controls.get("row")
        if row is not None and row in self.list_view.controls:
            self.list_view.controls.remove(row)
        self.utterances.pop(utt.id, None)
        if utt.id in self.order:
            self.order.remove(utt.id)
        if self._paragraph_uid == utt.id:
            self._paragraph_uid = None

    # -- transcript rendering ---------------------------------------------

    def _new_utterance(self, start: float = 0.0) -> int:
        uid = self._next_id
        self._next_id += 1
        utt = Utterance(id=uid, start=start)
        self.utterances[uid] = utt
        self.order.append(uid)
        self._build_row(utt)
        self._render_utterance(utt)
        if len(self.order) > MAX_UTTERANCES:
            old = self.order.pop(0)
            dropped = self.utterances.pop(old, None)
            if dropped and dropped.controls.get("row") in self.list_view.controls:
                self.list_view.controls.remove(dropped.controls["row"])
        return uid

    def _build_row(self, utt: Utterance) -> None:
        badge_text = ft.Text("…", size=10, weight=ft.FontWeight.BOLD, color=ft.Colors.ON_PRIMARY)
        badge_box = ft.Container(
            content=badge_text,
            bgcolor=ft.Colors.PRIMARY,
            border_radius=6,
            padding=ft.Padding.symmetric(horizontal=6, vertical=2),
        )
        body = ft.Text("", size=16, selectable=True)
        meta = ft.Text("", size=10, color=ft.Colors.with_opacity(0.4, ft.Colors.ON_SURFACE))

        container = ft.Container(
            content=ft.Column(
                [
                    ft.Row([badge_box, meta], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    body,
                ],
                spacing=6,
            ),
            padding=14,
            border_radius=14,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
            animate_opacity=200,
        )
        utt.controls = {"badge": badge_text, "body": body, "meta": meta, "row": container}
        self.list_view.controls.append(container)

    def _render_utterance(self, utt: Utterance) -> None:
        if not utt.controls:
            self._build_row(utt)
        utt.controls["badge"].value = badge(utt.language)
        utt.controls["body"].value = utt.text or "…"
        utt.controls["body"].italic = not utt.final
        utt.controls["body"].color = (
            ft.Colors.ON_SURFACE if utt.final else ft.Colors.with_opacity(0.6, ft.Colors.ON_SURFACE)
        )
        utt.controls["meta"].value = (
            f"{utt.seconds:.1f}s" + ("" if utt.final else " · live")
        )

    # -- actions -----------------------------------------------------------

    def _on_language(self, e: ft.Event[ft.Dropdown]) -> None:
        self._language_name = e.control.value or ""
        # A new choice invalidates the auto-detected language for this paragraph.
        self._sticky_language = None

    async def _copy_all(self, _e) -> None:
        text = self._export_text()
        if not text.strip():
            self._snack("Nothing to copy yet.")
            return
        await self._clipboard().set(text)
        self._snack("Transcript copied.")

    async def _clear_all(self, _e) -> None:
        self.utterances.clear()
        self.order.clear()
        self.list_view.controls.clear()
        self._paragraph_uid = None
        self._last_segment_end = 0.0
        self._drafts.clear()
        self._sticky_uid = None
        self._sticky_language = None
        self._update_save_enabled()
        self._show_empty(True)
        self._set_status("Ready" if self.model_ready else "Loading model…",
                         ft.Colors.PRIMARY if self.model_ready else None)

    # -- export ------------------------------------------------------------

    def _export_text(self) -> str:
        """Plain transcript: one paragraph per utterance."""
        lines = [
            self.utterances[uid].text.strip()
            for uid in self.order
            if self.utterances[uid].text.strip()
        ]
        return "\n\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _srt_timestamp(seconds: float) -> str:
        total_ms = max(0, int(round(seconds * 1000)))
        hours, total_ms = divmod(total_ms, 3_600_000)
        minutes, total_ms = divmod(total_ms, 60_000)
        secs, millis = divmod(total_ms, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _export_srt(self) -> str:
        """SubRip using each chunk's position in the recording.

        A paragraph can hold several chunks (live speech, or a long imported
        file), so emit one cue per chunk rather than one cue covering the whole
        paragraph. Utterances without chunk data fall back to their own span.
        """
        blocks: list[str] = []
        for uid in self.order:
            utt = self.utterances[uid]
            starts = sorted(start for start, text in utt.segments.items() if text.strip())
            if not starts:
                text = utt.text.strip()
                if not text:
                    continue
                spans = [(utt.start, utt.start + max(utt.seconds, 0.2), text)]
            else:
                spans = []
                for index, start in enumerate(starts):
                    text = normalize_text(utt.segments[start])
                    if index + 1 < len(starts):
                        end = starts[index + 1]
                    else:
                        end = utt.start + utt.seconds
                    spans.append((start, max(start + 0.2, end), text))
            for start, end, text in spans:
                blocks.append(
                    f"{len(blocks) + 1}\n"
                    f"{self._srt_timestamp(start)} --> {self._srt_timestamp(end)}\n"
                    f"{text}"
                )
        return "\n\n".join(blocks) + ("\n" if blocks else "")

    async def _save(self, fmt: str) -> None:
        payload = self._export_srt() if fmt == "srt" else self._export_text()
        if not payload.strip():
            self._snack("Nothing to save yet.")
            return

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            path = await self._file_picker().save_file(
                dialog_title=f"Save transcript (.{fmt})",
                file_name=f"transcript-{stamp}.{fmt}",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=[fmt],
                src_bytes=payload.encode("utf-8"),
            )
        except Exception as exc:  # noqa: BLE001
            self._snack(f"Could not save: {exc}")
            return

        if path:
            self._snack(f"Saved {Path(path).name}")
        else:
            self._snack("Save cancelled.")

    async def _import_file(self, _e) -> None:
        if self.engine is None:
            self._snack("Model is not loaded.")
            return
        if self.recording:
            self._snack("Stop recording before importing a file.")
            return
        files = await self._file_picker().pick_files(
            dialog_title="Open audio",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["wav"],
            allow_multiple=False,
            with_data=True,
        )
        if not files:
            return
        picked = files[0]
        if not picked.bytes:
            self._snack("Could not read the file.")
            return

        try:
            audio, rate = decode_wav(picked.bytes)
            audio = np.ascontiguousarray(
                resample(audio, rate, RECORD_RATE), dtype=np.float32
            )
        except Exception as exc:  # noqa: BLE001
            self._snack(f"Unsupported WAV: {exc}")
            return

        # A long file used to go to the encoder in one piece: memory grew with
        # its length and the token budget (~12 tokens/s, capped at 256) cut the
        # transcript short. Split it at pauses instead, exactly like live audio.
        # Chunks are longer than the live ones because nothing is waiting on
        # them; a smaller configured live chunk still scales imports down.
        import_seconds = min(
            IMPORT_MAX_SECONDS, max(5.0, _max_segment_seconds() * 3)
        )
        self._set_status("Preparing…", ft.Colors.AMBER)
        chunks = await asyncio.to_thread(
            split_imported_audio, audio, import_seconds
        )
        if not chunks:
            self._snack("No speech detected in that file.")
            self._set_status("Ready", ft.Colors.PRIMARY)
            return

        # One row: the segments are merged in timestamp order as they finish,
        # so the row never disappears from under a queued job.
        uid = self._new_utterance(start=0.0)
        self._set_status(
            f"Transcribing {len(chunks)} chunk{'s' if len(chunks) != 1 else ''}…",
            ft.Colors.AMBER,
        )
        for start, chunk, seconds in chunks:
            self._submit(
                Job(uid, chunk, final=True, seconds=seconds, segment_start=start)
            )
        self._update_save_enabled()
        self._show_empty(False)
        self.page.update()
