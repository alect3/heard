"""heard CLI.

Subcommands:
  ptt                 run the hold-CapsLock push-to-talk daemon (primary input)
  transcribe <wav>    normalize, bias, POST, and type a WAV (shared path)
  start / stop        pw-record capture fallback (the Mod+Shift+Space keybind)
  toggle              start if idle, else stop
  context [--show]    debug: print the bias resolved from the focused pane
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

from heard import context
from heard.config import Config, load
from heard.runtime import state_dir
from heard.transcribe import transcribe


def pidfile() -> Path:
    return state_dir() / "rec.pid"


def is_recording() -> bool:
    pf = pidfile()
    if not pf.is_file():
        return False
    try:
        os.kill(int(pf.read_text()), 0)
        return True
    except OSError, ValueError:
        return False


def start_recording() -> int:
    if is_recording():
        return 0
    from heard.runtime import notify

    notify(1200, "🎤 Recording…", "release to transcribe")
    wav = state_dir() / "rec.wav"
    # Native PipeWire capture — parecord (PulseAudio compat) returns silence on
    # a fresh boot here. --latency 50ms keeps the buffer small so audio lands in
    # the file in ~real time (a big buffer would be lost when we kill on stop).
    #
    # Detach: the recorder outlives this process (stop is a separate
    # invocation), so it must not inherit our stdio — an inherited pipe would
    # keep a capturing parent (or `$(...)`) blocked until the recorder dies.
    proc = subprocess.Popen(
        [
            "pw-record",
            "--rate",
            "16000",
            "--channels",
            "1",
            "--format",
            "s16",
            "--latency",
            "50ms",
            str(wav),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pidfile().write_text(str(proc.pid))
    return 0


def stop_recording(cfg: Config) -> int:
    if not is_recording():
        return 0
    pf = pidfile()
    pid = int(pf.read_text())
    pf.unlink(missing_ok=True)
    try:
        os.kill(pid, signal.SIGTERM)
        os.waitpid(pid, 0)
    except OSError, ChildProcessError:
        pass
    return transcribe(cfg, state_dir() / "rec.wav")


def show_context(cfg: Config, show_terms: bool) -> int:
    bias = context.compute_bias(cfg)
    print(f"focused pane: {bias.pane_id or '(none / herdr unavailable)'}")
    print(f"\nhotwords ({len(bias.hotwords.split())}):\n  {bias.hotwords or '(none)'}")
    print(f"\nprompt:\n  {bias.prompt or '(none)'}")
    if show_terms and bias.on_screen:
        print(f"\non-screen terms ({len(bias.on_screen)}):")
        print("  " + ", ".join(bias.on_screen))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heard", description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("ptt", help="run the push-to-talk daemon")

    p_tr = sub.add_parser("transcribe", help="transcribe and type a WAV")
    p_tr.add_argument("wav", type=Path)

    sub.add_parser("start", help="start fallback capture")
    sub.add_parser("stop", help="stop fallback capture and transcribe")
    sub.add_parser("toggle", help="toggle fallback capture")

    p_ctx = sub.add_parser("context", help="show the bias for the focused pane")
    p_ctx.add_argument("--show", action="store_true", help="also list on-screen terms")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load()
    cmd = args.cmd or "toggle"

    if cmd == "ptt":
        from heard.ptt import run

        run(cfg)
        return 0
    if cmd == "transcribe":
        return transcribe(cfg, args.wav)
    if cmd == "start":
        return start_recording()
    if cmd == "stop":
        return stop_recording(cfg)
    if cmd == "toggle":
        return stop_recording(cfg) if is_recording() else start_recording()
    if cmd == "context":
        return show_context(cfg, args.show)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
