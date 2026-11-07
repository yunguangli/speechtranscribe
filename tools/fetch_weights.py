#!/usr/bin/env python3
"""Fetch the Qwen3-ASR-0.6B int4 ONNX weights into ``src/assets/models/``.

The weights are ~790 MB and are intentionally not committed to git. Run this
once before the first launch (or before ``flet build``)::

    python tools/fetch_weights.py            # download missing files
    python tools/fetch_weights.py --check    # verify only, no network
    python tools/fetch_weights.py --force    # re-download everything

Files come from ``rhasspy/qwen3-asr-0.6b-onnx-int4-merged`` (Apache-2.0). The
"merged" export is ~700 MB smaller than the split one and keeps the decoder KV
cache for the system prompt, which is what makes short utterances fast.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "rhasspy/qwen3-asr-0.6b-onnx-int4-merged"
REVISION = "main"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}"

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "src" / "assets" / "models" / "qwen3-asr-0.6b-int4"

# name -> (size in bytes, sha256 of the published file). The digests are the
# ones huggingface.co reports for this revision, so a corrupted or substituted
# download cannot be mistaken for the real export.
FILES: dict[str, tuple[int, str]] = {
    "config.json": (
        1_250,
        "df31c4689abe9d782366fffd2454b546291d0205d082b3fd01b99fb76a45b11f",
    ),
    "tokenizer.json": (
        11_429_377,
        "bd2a97b55c8f7f9c328c73ee9b9178771037e9f566dfca8e238a063d41cbac92",
    ),
    "embed_tokens.bin": (
        311_164_928,
        "e80150119fa5f7e56e85aed64c3a02d5c78eb7a37cfdcb973d0987316f15bee2",
    ),
    "encoder.int4.onnx": (
        269_239,
        "fa42c87613923c9df9576f1bb40adbe6bf14fe5e0f249f1780d9da03d4aa4b78",
    ),
    "encoder.int4.onnx.data": (
        121_314_432,
        "25597bfb2765ac8d82fa725cbe9fe9b8322456a8db093df941def95ebbe41edb",
    ),
    "decoder_merged.int4.onnx": (
        938_846,
        "c98ce48230eabc98456c4555108ec554b1cdf5838661c883c168f32d77fb19d4",
    ),
    "decoder_merged.int4.onnx.data": (
        340_130_816,
        "5b5c847632e2d74910c81285a4cf6d2479c4d57f897242e8bb3eba772dc3208b",
    ),
}

LICENSE_TEXT = """\
Qwen3-ASR-0.6B
Copyright (c) 2026 Alibaba Cloud / Qwen team
Licensed under the Apache License, Version 2.0.
https://huggingface.co/Qwen/Qwen3-ASR-0.6B

ONNX export: https://huggingface.co/rhasspy/qwen3-asr-0.6b-onnx-int4-merged
"""


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, ValueError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, expected: int, sha256: str, force: bool = False) -> None:
    dest = MODEL_DIR / name
    url = f"{BASE_URL}/{name}"

    if dest.exists() and not force:
        if dest.stat().st_size == expected and _sha256(dest) == sha256:
            print(f"  ok   {name} ({_human(dest.stat().st_size)})")
            return
        print(f"  checksum mismatch for {name}, re-downloading")

    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() and not force else 0
    total = expected or _remote_size(url) or 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if have else "wb"

    print(f"  get  {name}  {_human(have)} / {_human(total)}")
    started = time.time()
    last = 0.0
    with urllib.request.urlopen(req, timeout=60) as resp, open(part, mode) as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            have += len(chunk)
            now = time.time()
            if now - last > 2:
                last = now
                rate = have / max(now - started, 0.001)
                pct = f"{100 * have / total:5.1f}%" if total else "  ?  "
                print(
                    f"\r       {pct} {_human(have)}  {_human(rate)}/s",
                    end="",
                    flush=True,
                )
    print()

    if part.stat().st_size != expected:
        sys.exit(
            f"ERROR: {name} is {part.stat().st_size} bytes, expected {expected}.\n"
            f"Inspect {FILE_BYTES_HINT}"
        )
    digest = _sha256(part)
    if digest != sha256:
        part.unlink(missing_ok=True)
        sys.exit(
            f"ERROR: {name} has the wrong checksum.\n"
            f"  expected {sha256}\n  got      {digest}\n"
            f"The mirror may have been re-exported; inspect {FILE_BYTES_HINT}"
        )
    os.replace(part, dest)


FILE_BYTES_HINT = f"https://huggingface.co/{REPO}/tree/main"


def verify() -> bool:
    ok = True
    for name, (expected, sha256) in FILES.items():
        path = MODEL_DIR / name
        if not path.exists():
            print(f"  MISSING {name}")
            ok = False
        elif path.stat().st_size != expected:
            print(f"  BAD     {name} ({path.stat().st_size} != {expected})")
            ok = False
        elif _sha256(path) != sha256:
            print(f"  BAD     {name} (checksum mismatch)")
            ok = False
        else:
            print(f"  ok      {name} ({_human(path.stat().st_size)})")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without downloading")
    parser.add_argument("--force", action="store_true", help="re-download everything")
    args = parser.parse_args()

    if args.check:
        print(f"{MODEL_DIR}:")
        return 0 if verify() else 1

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(MODEL_DIR).free
    need = sum(size for size, _sha in FILES.values())
    if free < need + (256 << 20):
        sys.exit(f"Need ~{_human(need)} free, only {_human(free)} available.")

    print(f"Downloading {REPO} -> {MODEL_DIR}")
    for name, (expected, sha256) in FILES.items():
        download(name, expected, sha256, force=args.force)

    (MODEL_DIR / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8")
    (MODEL_DIR / ".ready").write_text("ok\n", encoding="utf-8")

    print("\nVerifying:")
    return 0 if verify() else 1


if __name__ == "__main__":
    raise SystemExit(main())
