"""Small shared helpers: desktop toasts and the tmpfs state directory."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def state_dir() -> Path:
    """Per-session scratch under $XDG_RUNTIME_DIR (tmpfs, cleared on logout)."""
    runtime = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
    d = Path(runtime) / "heard"
    d.mkdir(parents=True, exist_ok=True)
    return d


def notify(timeout_ms: int, summary: str, body: str = "") -> None:
    """Fire-and-forget mako/notify-send toast; never raise if it's missing."""
    try:
        subprocess.Popen(
            ["notify-send", "-a", "heard", "-t", str(timeout_ms), summary, body]
        )
    except OSError:
        pass
