# heard

Push-to-talk voice dictation for the Wayland desktop, with **herdr-aware
on-screen vocabulary bias**.

Hold CapsLock, talk, release — the transcript is typed into whatever window is
focused. What makes it different from plain Whisper dictation: while you talk,
`heard` reads the focused [herdr](https://herdr.dev) pane and feeds the
identifiers, filenames, flags, and jargon *currently on your screen* to the STT
model as term-boost `hotwords`. Dictate `"check the opossum-ec wrapper"` at a
terminal showing that file and it transcribes the identifier instead of
guessing at English homophones.

Extracted from a SaltStack `voice-input` formula into a standalone project.

## How it works

```
CapsLock down ─┬─▶ pre-roll ring seeds the utterance (no clipped first syllable)
               └─▶ herdr: read focused pane ─▶ extract on-screen vocab ─▶ cache
CapsLock up   ───▶ write WAV ─▶ ffmpeg speechnorm ─▶ POST to STT endpoint
                                 (prompt + on-screen hotwords) ─▶ wtype types it
```

`heard` is **client-only**: the PTT daemon, the transcription path, and the
herdr context builder, in one `uv`/`uvx`-installable Python package. It talks to
an already-running OpenAI-compatible transcription endpoint (point `endpoint` at
your [speaches](https://speaches.ai)/faster-whisper server, or anything speaking
the `/v1/audio/transcriptions` API) — it does not launch or manage the model.

The context step is entirely best-effort: outside herdr (`HERDR_ENV != 1`), or
if the `herdr` CLI is missing or slow, it silently falls back to the static
`prompt`/`hotwords` from config. Dictation never breaks over it.

## Requirements

Runtime binaries (Arch package names): `wtype`, `ffmpeg`, `pipewire` (`pw-cat`,
`pw-record`), `libnotify` (`notify-send`), and `herdr` for context. The PTT
daemon reads `/dev/input`, so the user must be in the `input` group.

## Install

```sh
uv sync                       # dev checkout
uv run heard context --show   # smoke-test the herdr context resolver
```

Or run without a checkout once published:

```sh
uvx --from git+https://…/heard heard ptt
```

Copy `config.example.toml` to `~/.config/heard/config.toml` and edit.

## Usage

```sh
heard ptt                 # the hold-CapsLock daemon (primary; niri spawn-at-startup)
heard toggle              # keybind fallback (start/stop capture)
heard transcribe <wav>    # transcribe + type a WAV (the shared path)
heard context [--show]    # debug: print the bias resolved from the focused pane
```

### Desktop glue (niri)

`caps:none` in the xkb options neutralises CapsLock's normal function (evdev
still reports the raw press/release); then:

```kdl
spawn-at-startup "uv" "run" "--project" "/home/alec/projects/heard" "heard" "ptt"
// fallback toggle:
Mod+Shift+Space { spawn "uv" "run" "--project" "/home/alec/projects/heard" "heard" "toggle"; }
```

## Transcription endpoint

`heard` expects an already-running OpenAI-compatible STT server and just POSTs
to it. Set `endpoint` (and `model`) in the config to wherever yours lives —
`http://127.0.0.1:8083/v1/audio/transcriptions` by default. Standing up
[speaches](https://speaches.ai) or another faster-whisper server is out of
scope for this project.

## Configuration

All config is read at **runtime** from `~/.config/heard/config.toml` (override
path with `$HEARD_CONFIG`), with `HEARD_*` env vars taking final precedence — so
one installed artifact serves every machine. See `config.example.toml` for the
full annotated set. Key `[context]` knobs: `enabled`, `source`
(`visible`/`recent`), `max_hotwords`, `prompt_terms`.

## Development

```sh
uv sync --extra dev
uv run pytest          # unit + end-to-end tests
uv run black --check . # formatting
uv run ruff check .    # linting
```

Tests cover the pure vocabulary/bias logic plus end-to-end flows that drive the
real CLI with the STT server, `wtype`, `ffmpeg`, `herdr`, and `pw-record` all
faked, asserting on what would be typed and what the model was asked to
transcribe. GitHub Actions runs black, ruff, and pytest on every push and PR
(`.github/workflows/ci.yml`).
