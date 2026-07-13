"""End-to-end test harness.

These tests drive the real `heard` CLI as a subprocess — real config loading,
a real HTTP round-trip, real argparse dispatch — with the whole external world
faked so we can observe exactly what a user would experience:

  * a fake STT server standing in for speaches, which records every request so
    we can assert on the model, prompt, and on-screen `hotwords` it received;
  * stub `wtype` / `ffmpeg` / `notify-send` / `herdr` / `pw-record` binaries on
    PATH, so "the transcript was typed into my window" becomes an assertion on
    the wtype stub's log, and "it read my focused pane" becomes controllable
    pane text.

Nothing here touches a mic, a GPU, a compositor, or the network.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

# The installed console script (pyproject `heard = "heard.cli:main"`). Resolved
# from the real PATH before we shadow it with the stub bin dir.
HEARD_BIN = shutil.which("heard")


# --- fake STT server ------------------------------------------------------


class _Recorder:
    """Shared state between the HTTP handler and the test."""

    def __init__(self) -> None:
        self.transcript = "hello world"
        self.status = 200
        self.bodies: list[bytes] = []


def _make_handler(rec: _Recorder):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            rec.bodies.append(self.rfile.read(n))
            payload = rec.transcript.encode()
            self.send_response(rec.status)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):  # silence
            pass

    return Handler


@dataclass
class FakeStt:
    server: HTTPServer
    rec: _Recorder

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://127.0.0.1:{port}/v1/audio/transcriptions"

    def set_transcript(self, text: str) -> None:
        self.rec.transcript = text

    def set_status(self, code: int) -> None:
        self.rec.status = code

    @property
    def request_count(self) -> int:
        return len(self.rec.bodies)

    def last_body(self) -> str:
        assert self.rec.bodies, "STT server received no request"
        return self.rec.bodies[-1].decode("latin1")


@pytest.fixture
def fake_stt():
    rec = _Recorder()
    server = HTTPServer(("127.0.0.1", 0), _make_handler(rec))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield FakeStt(server, rec)
    finally:
        server.shutdown()


# --- stub binaries --------------------------------------------------------

_WTYPE_STUB = """\
#!/usr/bin/env python3
import os, sys
with open(os.environ["HEARD_TEST_WTYPE_LOG"], "a") as f:
    f.write("\\t".join(sys.argv[1:]) + "\\n")
"""

# Pretend to normalize: copy the input WAV (after -i) to the output (last arg).
_FFMPEG_STUB = """\
#!/usr/bin/env python3
import shutil, sys
a = sys.argv[1:]
shutil.copyfile(a[a.index("-i") + 1], a[-1])
"""

_NOTIFY_STUB = "#!/usr/bin/env python3\n"  # no-op

# Emulate the herdr CLI surface heard uses: pane list / pane read / pane current.
_HERDR_STUB = """\
#!/usr/bin/env python3
import json, os, sys
a = sys.argv[1:]
if a[:2] == ["pane", "list"]:
    print(json.dumps({"result": {"type": "pane_list", "panes": [
        {"pane_id": "t:p1", "focused": False},
        {"pane_id": "t:p2", "focused": True},
    ]}}))
elif a[:2] == ["pane", "read"]:
    print(os.environ.get("HEARD_TEST_PANE_TEXT", ""))
elif a[:2] == ["pane", "current"]:
    print(json.dumps({"result": {"pane": {"pane_id": "t:p1"}}}))
else:
    sys.exit(1)
"""

# Write a small silent WAV immediately, then idle until SIGTERM (like pw-record
# streaming to a file until killed on stop).
_PW_RECORD_STUB = """\
#!/usr/bin/env python3
import signal, struct, sys, time, wave
out = sys.argv[-1]
with wave.open(out, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(struct.pack("<" + "h" * 8000, *([0] * 8000)))
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
while True:
    time.sleep(0.02)
"""


def _write_stub(bindir: Path, name: str, body: str) -> None:
    p = bindir / name
    p.write_text(body)
    p.chmod(0o755)


# --- the harness object driven by tests -----------------------------------


@dataclass
class Harness:
    root: Path
    bindir: Path
    runtime: Path
    config_path: Path
    wtype_log: Path
    env: dict
    stt: FakeStt
    pane_text: str = ""

    def set_pane_text(self, text: str) -> None:
        self.pane_text = text
        self.env["HEARD_TEST_PANE_TEXT"] = text

    def write_config(self, **overrides) -> None:
        cfg = {
            "endpoint": self.stt.url,
            "model": "test-whisper",
            "language": "en",
            "submit": True,
            "prompt": "Base prompt.",
            "hotwords": "git podman",
        }
        context = {
            "enabled": True,
            "source": "visible",
            "lines": 50,
            "max_hotwords": 48,
            "prompt_terms": 24,
            "cache_ttl": 8.0,
        }
        context.update(overrides.pop("context", {}))
        cfg.update(overrides)

        def toml_val(v):
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, (int, float)):
                return str(v)
            return json.dumps(v)  # quoted string

        lines = [f"{k} = {toml_val(v)}" for k, v in cfg.items()]
        lines.append("\n[context]")
        lines += [f"{k} = {toml_val(v)}" for k, v in context.items()]
        self.config_path.write_text("\n".join(lines) + "\n")

    def make_wav(self, name: str = "in.wav", samples: int = 8000) -> Path:
        p = self.root / name
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(struct.pack("<" + "h" * samples, *([0] * samples)))
        return p

    def run(
        self, *args: str, extra_env: dict | None = None, timeout: float = 30
    ) -> subprocess.CompletedProcess:
        env = dict(self.env)
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [HEARD_BIN, *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def wtype_calls(self) -> list[list[str]]:
        if not self.wtype_log.exists():
            return []
        out = []
        for line in self.wtype_log.read_text().splitlines():
            out.append(line.split("\t"))
        return out

    def typed_text(self) -> str | None:
        """The text handed to `wtype -- <text>`, as the user would see typed."""
        for call in self.wtype_calls():
            if call and call[0] == "--":
                return call[1] if len(call) > 1 else ""
        return None

    def pressed_return(self) -> bool:
        return ["-k", "Return"] in self.wtype_calls()

    def wait_for(self, path: Path, timeout: float = 3.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if path.exists():
                return True
            time.sleep(0.02)
        return False


@pytest.fixture
def harness(tmp_path, fake_stt):
    if HEARD_BIN is None:
        pytest.skip("`heard` console script not installed; run under `uv run`")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _write_stub(bindir, "wtype", _WTYPE_STUB)
    _write_stub(bindir, "ffmpeg", _FFMPEG_STUB)
    _write_stub(bindir, "notify-send", _NOTIFY_STUB)
    _write_stub(bindir, "herdr", _HERDR_STUB)
    _write_stub(bindir, "pw-record", _PW_RECORD_STUB)

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    config_path = tmp_path / "config.toml"
    wtype_log = tmp_path / "wtype.log"

    env = dict(os.environ)
    env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    env["HEARD_CONFIG"] = str(config_path)
    env["XDG_RUNTIME_DIR"] = str(runtime)
    env["HERDR_ENV"] = "1"
    env["HEARD_TEST_WTYPE_LOG"] = str(wtype_log)
    env["HEARD_TEST_PANE_TEXT"] = ""

    h = Harness(
        root=tmp_path,
        bindir=bindir,
        runtime=runtime,
        config_path=config_path,
        wtype_log=wtype_log,
        env=env,
        stt=fake_stt,
    )
    h.write_config()
    return h
