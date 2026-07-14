"""The shared transcription path: normalize → bias → POST → type.

Ported from the original voice-input.sh. Both the PTT daemon and the keybind
fallback funnel a WAV through here; `transcribe()` is the whole story, the rest
are its steps.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import httpx

from heard.config import Config
from heard.context import build_bias
from heard.runtime import notify, state_dir


def normalize(wav: Path) -> Path:
    """Normalize gain before sending — Whisper degrades badly on quiet input
    and mic levels vary. speechnorm is speech-tuned and single-pass-safe on
    short clips; l=1 limits so loud speech won't clip. Fall back to the raw
    clip if ffmpeg is missing or fails."""
    out = state_dir() / "rec.norm.wav"
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(wav),
                "-af",
                "speechnorm=e=12.5:r=0.0001:l=1",
                str(out),
            ],
            capture_output=True,
        )
    except OSError:
        return wav
    return out if proc.returncode == 0 and out.is_file() else wav


def post(cfg: Config, audio: Path) -> str:
    """POST the audio to the STT endpoint with the resolved vocabulary bias."""
    bias = build_bias(cfg)
    data = {"model": cfg.model, "response_format": "text"}
    if cfg.language:
        data["language"] = cfg.language
    if bias.prompt:
        data["prompt"] = bias.prompt
    if bias.hotwords:
        data["hotwords"] = bias.hotwords
    # Decode knobs the OpenAI-compatible endpoint accepts per request. Sent
    # explicitly so the server's own defaults can't drift under us. vad_filter
    # runs Silero VAD before decoding — it drops non-speech spans, which curbs
    # the repetition/early-stop that otherwise truncates a clip to its first
    # few words on pausey speech.
    data["vad_filter"] = "true" if cfg.vad_filter else "false"
    data["temperature"] = str(cfg.temperature)
    with audio.open("rb") as fh:
        resp = httpx.post(
            cfg.endpoint,
            data=data,
            files={"file": (audio.name, fh, "audio/wav")},
            timeout=60.0,
        )
    resp.raise_for_status()
    return resp.text


def type_text(text: str, submit: bool) -> None:
    subprocess.run(["wtype", "--", text], check=False)
    if submit:
        subprocess.run(["wtype", "-k", "Return"], check=False)


def transcribe(cfg: Config, wav: Path) -> int:
    """Normalize, transcribe, and type `wav`. Returns a process exit code."""
    if not wav.is_file():
        return 0
    notify(1500, "🎤 Transcribing…")

    audio = normalize(wav)
    try:
        raw = post(cfg, audio)
    except httpx.HTTPStatusError as e:
        body = e.response.text.strip() if e.response is not None else ""
        notify(5000, "heard: transcription failed", body or str(e))
        return 1
    except httpx.HTTPError as e:
        notify(5000, "heard: transcription failed", str(e))
        return 1

    text = raw.strip()
    if not text:
        notify(2000, "heard", "no speech detected")
        return 0

    type_text(text, cfg.submit)
    return 0
