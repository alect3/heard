"""Push-to-talk key listener with pre-roll — ported from voice-input-ptt.py.

Watches the PTT key (CapsLock) at the evdev layer, BELOW the compositor (niri
has no on-release keybind action). xkb `caps:none` neutralises CapsLock's
normal function; evdev still reports its raw press/release, which is all we use.

Pre-roll: pw-cat streams raw PCM continuously into a small rolling buffer. On
keydown we seed the utterance with that buffer, so audio from just BEFORE the
press is kept — this eliminates the stream-setup gap that otherwise clips the
first syllable. We also fire a herdr context prefetch on keydown so the pane
read overlaps the recording and adds no latency. On keyup we grab a short tail,
write a WAV, and transcribe it.

Capture uses pw-cat (native PipeWire), NOT parecord — the PulseAudio compat
path returns silence on a fresh boot here. The capture stream is respawned if
it dies, which also rides out the startup race where niri's spawn-at-startup
launches this before the audio stack is ready.

Launched by niri `spawn-at-startup` so it inherits the graphical-session env
(WAYLAND_DISPLAY / PIPEWIRE / DBUS). Requires the `input` group to read
/dev/input.
"""

from __future__ import annotations

import selectors
import subprocess
import threading
import time
import wave
from collections import deque

import evdev
from evdev import ecodes

from heard import context
from heard.config import Config
from heard.runtime import notify, state_dir
from heard.transcribe import transcribe

PTT_KEY = ecodes.KEY_CAPSLOCK

RATE = 16000
CHANNELS = 1
SAMPWIDTH = 2  # s16
CHUNK = 2048
PREROLL_MS = 400  # audio kept from before the press
TAIL_MS = 150  # drain the in-flight latency buffer after release
MIN_MS = 200  # ignore accidental taps shorter than this

PREROLL_BYTES = RATE * CHANNELS * SAMPWIDTH * PREROLL_MS // 1000
MIN_BYTES = RATE * CHANNELS * SAMPWIDTH * MIN_MS // 1000

CAPTURE_CMD = [
    "pw-cat",
    "--record",
    "--raw",
    "--rate",
    str(RATE),
    "--channels",
    str(CHANNELS),
    "--format",
    "s16",
    "-",
]


class PushToTalk:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.ring: deque[bytes] = deque()
        self.ring_bytes = 0
        self.recording = False
        self.captured = bytearray()
        self.wav_path = state_dir() / "ptt.wav"

    # --- capture stream --------------------------------------------------
    def reader(self) -> None:
        """Own the capture stream: feed the pre-roll ring (and the active
        utterance while recording). Respawn pw-cat if it exits — this rides
        out the boot-time race where the audio stack isn't ready yet."""
        while True:
            stream = subprocess.Popen(CAPTURE_CMD, stdout=subprocess.PIPE)
            while True:
                chunk = stream.stdout.read(CHUNK)
                if not chunk:
                    break
                with self.lock:
                    if self.recording:
                        self.captured.extend(chunk)
                    self.ring.append(chunk)
                    self.ring_bytes += len(chunk)
                    while (
                        len(self.ring) > 1
                        and self.ring_bytes - len(self.ring[0]) >= PREROLL_BYTES
                    ):
                        self.ring_bytes -= len(self.ring.popleft())
            try:
                stream.kill()
            except OSError:
                pass
            time.sleep(1)  # capture died -> back off and respawn

    # --- key transitions -------------------------------------------------
    def start(self) -> None:
        """Keydown: seed the utterance with pre-roll and prefetch context."""
        with self.lock:
            self.captured.clear()
            self.captured.extend(b"".join(self.ring))  # seed with pre-roll
            self.recording = True
        # Read the focused herdr pane while the user talks — ready by keyup.
        threading.Thread(target=self.prefetch, daemon=True).start()
        notify(1200, "🎤 Recording…", "release to transcribe")

    def prefetch(self) -> None:
        try:
            context.prefetch(self.cfg)
        except Exception:
            pass  # context is best-effort; never break dictation over it

    def stop(self) -> None:
        """Keyup: let the tail arrive, snapshot the audio, hand it off. Runs in
        its own thread so the event loop stays responsive."""
        time.sleep(TAIL_MS / 1000)
        with self.lock:
            self.recording = False
            data = bytes(self.captured)
            self.captured.clear()
        if len(data) < MIN_BYTES:
            return
        with wave.open(str(self.wav_path), "wb") as w:
            w.setnchannels(CHANNELS)
            w.setsampwidth(SAMPWIDTH)
            w.setframerate(RATE)
            w.writeframes(data)
        transcribe(self.cfg, self.wav_path)

    # --- event loop ------------------------------------------------------
    def run(self) -> None:
        threading.Thread(target=self.reader, daemon=True).start()

        devices = ptt_keyboards()
        if not devices:
            raise SystemExit("heard ptt: no device exposes the PTT key")
        selector = selectors.DefaultSelector()
        for dev in devices:
            selector.register(dev, selectors.EVENT_READ)

        while True:
            for key, _ in selector.select():
                for event in key.fileobj.read():
                    if event.type != ecodes.EV_KEY or event.code != PTT_KEY:
                        continue
                    if event.value == 1:  # key down
                        self.start()
                    elif event.value == 0:  # key up (2 == autorepeat, ignored)
                        threading.Thread(target=self.stop, daemon=True).start()


def ptt_keyboards() -> list[evdev.InputDevice]:
    devices = []
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
        except PermissionError, OSError:
            continue
        if PTT_KEY in dev.capabilities().get(ecodes.EV_KEY, []):
            devices.append(dev)
    return devices


def run(cfg: Config) -> None:
    PushToTalk(cfg).run()
